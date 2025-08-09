from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
import math


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
    """ Apply multiple pedals in order (left to right). """
    pedals: Sequence[Pedal] = ()
    
    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        out = waveform
        for pedal in self.pedals:
            out = pedal(out)
        return out



@dataclass(slots=True)
class Overdrive(Pedal):
    """
    Soft-clipping overdrive using a tanh waveshaper.
    y = mix * (tanh(color * gain * x) / tanh(color)) + (1-mix)*x

    Args:
        gain: Applied before the waveshaper
        color: Curvature / steepness of tanh (higher => earlier/harder clip)
        mix: Wet/dry blend in [0,1]; 1.0 is fully distorted
    """
    name: str = "overdrive"
    color: float = 0.5
    mix: float = 1.0
    gain: float = 1.0

    def apply(self, waveform: torch.Tensor,) -> torch.Tensor:
        device, dtype = waveform.device, waveform.dtype
        g = torch.as_tensor(self.gain, device=device, dtype=dtype)
        c = torch.as_tensor(max(0.0, self.color), device=device, dtype=dtype)
        m = torch.as_tensor(self.mix, device=device, dtype=dtype).clamp_(0.0, 1.0)

        # Pre-gain
        driven = waveform * g

        # Soft clipping
        eps = torch.finfo(dtype).eps # Smallest number you can add to 1.0
        denom = torch.tanh(c).clamp_min(eps) # Makes sure denominator doesn't blow up (tanh(0) = 0)
        shaped = torch.tanh(c*driven)/denom # tanh(c*x)/tanh(c)


        return (1-m) * waveform + m*(shaped)




@dataclass(slots=True)
class Delay(Pedal):
    """
    Simple feedback delay
    Args:
        delay_time: Delay time in seconds.
        feedback: Feedback amoutn in [0,1). Closer to 1 = more repeats.
        effect_level: Wet/dry mix in [0,1]; 0 = dry only, 1 = wet only.
        sr: Sample rate (Hz).
    """
    delay_time: float
    feedback: float
    effect_level: float
    name: str = "delay"
    sr: int = 44100
    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        device, dtype = waveform.device, waveform.dtype
        x = waveform
        T = x.numel()

        D = int(round(max(0.0, self.delay_time) * float(self.sr)))
        fb = float(self.feedback)
        m  = torch.as_tensor(self.effect_level, device=device, dtype=dtype).clamp_(0.0, 1.0)

        # No delay possible or no shift
        if T == 0 or D <= 0 or D >= T:
            return x

        # Compute echo count so last echo is below a tail threshold (e.g., -60 dB)
        tail_db = -60.0
        fb_mag = abs(fb)
        if fb_mag < 1e-6:
            K = 1
        else:
            tail_lin = 10.0 ** (tail_db / 20.0)  # -60 dB -> 0.001
            K = max(1, int(math.ceil(math.log(tail_lin) / math.log(fb_mag))))

        wet = torch.zeros_like(x, device=device, dtype=dtype)

        # First echo coefficient = 1.0 (important!), then multiply by fb each tap
        coeff = 1.0
        for k in range(1, K + 1):
            shift = D * k
            if shift >= T:
                break
            wet[shift:] += torch.as_tensor(coeff, device=device, dtype=dtype) * x[:-shift]
            coeff *= fb  # subsequent echoes get smaller by feedback

        # Blend
        return (1 - m) * x + m * wet

class Reverb(Pedal):
    ...

class Modulation(Pedal):
    ...



