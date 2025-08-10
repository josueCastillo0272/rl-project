from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np
import torch


from pedalboard import Pedalboard as PB
from pedalboard import Gain as PBGain
from pedalboard import Distortion as PBDistortion
from pedalboard import Delay as PBDelay
from pedalboard import Reverb as PBReverb
from pedalboard import Phaser as PBPhaser


def _to_mono_np(x: torch.Tensor) -> np.ndarray:
    """
    Accepts 1D mono waveform tensor (T,) and returns float32 numpy (T,).
    """
    if x.ndim != 1:
        raise ValueError(f"Expected mono 1D tensor, got shape {tuple(x.shape)}")
    return x.detach().cpu().to(torch.float32).numpy()

def _to_tensor(y: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert numpy mono (T,) back to torch on original device/dtype.
    """
    t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    return t.to(device=device, dtype=dtype)

def _lin_gain_to_db(g: float) -> float:
    """
    Convert linear gain factor to decibels.
    """
    g = max(g, 1e-12)
    return 20.0 * math.log10(g)



@dataclass(slots=True)
class Pedal(ABC):
    """
    Abstract audio effect unit.

    Attributes:
        name: Human-readable name of the pedal
    """

    @abstractmethod
    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process a mono waveform.

        Args:
            waveform: 1D tensor of shape (T,).

        Returns:
            Processed waveform as a 1D tensor of shape (T,).
        """
        ...

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.apply(waveform)


@dataclass(slots=True)
class SequencePedal(Pedal):
    """Apply multiple pedals in order (left to right)."""
    pedals: Sequence[Pedal] = ()
    name: str = "sequence"

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        out = waveform
        for pedal in self.pedals:
            out = pedal(out)
        return out



@dataclass(slots=True)
class Overdrive(Pedal):
    """
    Overdrive via pedalboard's Distortion (tanh) with optional pre-gain and dry/wet.
    This preserves your original (color, mix, gain) interface:

        - gain (linear)  -> mapped to pre-gain in dB
        - color (0..~1)  -> mapped to Distortion.drive_db (approx: 0..30 dB)
        - mix   (0..1)   -> dry/wet

    Note: pedalboard.Distortion uses tanh waveshaping internally.
    """
    name: str = "overdrive"
    color: float = 0.5    # mapped to drive_db ≈ 60 * color dB (tweak as you like)
    mix: float = 1.0      # wet/dry blend [0..1]
    gain: float = 1.0     # linear pre-gain (>1 pushes drive harder)

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        device, dtype = waveform.device, waveform.dtype
        x_np = _to_mono_np(waveform)

        # Build a small board: pre-gain -> distortion
        pre_gain_db = _lin_gain_to_db(float(self.gain))
        drive_db = float(self.color) * 100.0  # heuristic mapping; adjust to taste

        board = PB([
            PBGain(gain_db=pre_gain_db),
            PBDistortion(drive_db=drive_db),
        ])

        # Process at an arbitrary sample rate since Distortion is rate-agnostic.
        wet_np = board(x_np, sample_rate=48000)

        # Manual wet/dry blend (keep original peak relationship)
        m = float(np.clip(self.mix, 0.0, 1.0))
        out_np = (1.0 - m) * x_np + m * wet_np

        return _to_tensor(out_np, device, dtype)

@dataclass(slots=True)
class Delay(Pedal):
    """
    Simple feedback delay using pedalboard.Delay.

    Args:
        delay_time: delay in seconds
        feedback:   feedback in [0, 1)
        effect_level: wet/dry in [0, 1] (mapped to Delay.mix)
        sr: sample rate used for processing
    """
    delay_time: float
    feedback: float
    effect_level: float
    name: str = "delay"
    sr: int = 22050

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        device, dtype = waveform.device, waveform.dtype
        x_np = _to_mono_np(waveform)

        delay = PBDelay(
            delay_seconds=max(0.0, float(self.delay_time)),
            feedback=float(self.feedback),
            mix=float(np.clip(self.effect_level, 0.0, 1.0)),
        )
        wet_np = delay(x_np, sample_rate=float(self.sr))
        return _to_tensor(wet_np, device, dtype)


@dataclass(slots=True)
class Reverb(Pedal):
    """
    FreeVerb-style reverb via pedalboard.Reverb.

    Args:
        room_size: 0..1
        decay:     0..1 (mapped to damping)
        mix:       wet level (0..1). Dry is set to (1 - mix).
        sr:        sample rate for processing
    """
    room_size: float = 0.5
    decay: float = 0.5
    mix: float = 0.3
    sr: int = 22050
    name: str = "reverb"

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        device, dtype = waveform.device, waveform.dtype
        x_np = _to_mono_np(waveform)

        wet_level = float(np.clip(self.mix, 0.0, 1.0))
        dry_level = 1.0 - wet_level

        reverb = PBReverb(
            room_size=float(np.clip(self.room_size, 0.0, 1.0)),
            damping=float(np.clip(self.decay, 0.0, 1.0)),
            wet_level=wet_level,
            dry_level=dry_level,
            width=1.0,
            freeze_mode=0.0,
        )
        y_np = reverb(x_np, sample_rate=float(self.sr))
        return _to_tensor(y_np, device, dtype)

    __call__ = apply

@dataclass(slots=True)
class Phaser(Pedal):
    """
    6‑stage phaser via pedalboard.Phaser.

    Controls (typical phaser params):
      - rate_hz:    LFO speed in Hz
      - depth:      0..1 modulation depth
      - feedback:   -1..1 feedback amount (more = sharper notches)
      - centre_frequency_hz: center of the sweep in Hz
      - mix:        wet/dry in [0,1]
      - sr:         processing sample rate (Hz)
    """
    rate_hz: float = 0.8              # nice slow swirl
    depth: float = 0.9                # wide sweep
    feedback: float = 0.7             # strong but stable
    centre_frequency_hz: float = 1000.0  # sweep centered in upper mids
    mix: float = 0.6                   # more wet than dry
    sr: int = 22050
    name: str = "phaser"

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1:
            raise ValueError(f"{self.name} expects mono 1D tensor (T,), got {tuple(waveform.shape)}")

        device, dtype = waveform.device, waveform.dtype

        # -> numpy float32 (mono)
        x_np = waveform.detach().cpu().to(torch.float32).numpy()

        # Clamp common ranges to be safe
        depth = float(np.clip(self.depth, 0.0, 1.0))
        mix = float(np.clip(self.mix, 0.0, 1.0))
        feedback = float(np.clip(self.feedback, -0.99, 0.99))  # avoid instability

        ph = PBPhaser(
            rate_hz=float(self.rate_hz),
            depth=depth,
            feedback=feedback,
            centre_frequency_hz=float(self.centre_frequency_hz),
            mix=mix,
        )

        y_np = ph(x_np, sample_rate=float(self.sr))

        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32)).to(device=device, dtype=dtype)
        return y