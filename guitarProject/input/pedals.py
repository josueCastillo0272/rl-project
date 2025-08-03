
from abc import ABC, abstractmethod


class Pedal(ABC):
    ...
    # method for processing a tensor
    
class SequencePedal(Pedal):
    first: Pedal
    second: Pedal
    
    def process(inp):
        return second.apply(first.process(inp))
    

class Overdrive(Pedal):
    def __init__(self, gain: float):
        self.gain = gain

    def apply(self, waveform):
        ...


    
class Delay(Pedal):
    ...

class Reverb(Pedal):
    ...

class Modulation(Pedal):
    ...



