from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch



@dataclass(slots=True)
class Pedal(ABC):
    """
    Abstract audio effect unit.

    Attributes:
        name: Human-readable name of the pedal
    """
    name: str = "pedal"

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
    gain: float
    color: float = 0.5
    mix: float = 1.0
    name: str = "overdrive"

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
    sr: int = 44100
    name: str = "delay"
    def apply(self, waveform: torch.Tensor) -> torch.Tensor:

        device, dtype  = waveform.device, waveform.dtype
        T = waveform.numel()
        D = int(round(self.delay_time * self.sr))
        fb = float(self.feedback)
        m = torch.as_tensor(self.effect_level, device=device, dtype = dtype)

        # Delay not possible
        if T == 0 or D >= T:
            return waveform
        
        wet = torch.zeroes_like(waveform, device=device,dtype=dtype)
        temp = waveform.clone()

        # Accumulate echoes with feedback
        while True:
            temp = torch.cat([torch.zeroes(D, device=device, dtype=dtype), temp[:-D]]) * fb
            if temp.abs().max() < 1e-5: break # Stop when signal is tiny 
            wet += temp

        return (1-m) * waveform + m*wet

class Reverb(Pedal):
    ...

class Modulation(Pedal):
    ...



