from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import pyo

from .pedals import Pedal


class GuitarActionType(Enum):
    PICK = auto()
    STRUM = auto()
    HAMMER_ON = auto()
    PULL_OFF = auto()
    # Add more advanced techniques as needed


@dataclass
class GuitarAction:
    string: int  # 1 = high E, 6 = low E
    fret: Optional[int]  # None for open string or strum without fret
    action_type: GuitarActionType
    time: float  # Time in milliseconds or beats
    velocity: Optional[float] = None  # For strum intensity, etc.
    # Add more fields as needed for advanced techniques


@dataclass
class GuitarInputSequence:
    tuning: List[int] = [40, 45, 50, 55, 59, 64]  # MIDI values
    actions: List[GuitarAction] = field(default_factory=list)
    pedal: Pedal

    def add_action(self, action: GuitarAction):
        self.actions.append(action)

    def score_playability(self) -> float:
        """
        Placeholder for a function that scores how playable the sequence is.
        Returns:
            float: Playability score (higher is better)
        """
        return 0.0
           
    # TODO: move this into a simulator class
    def to_wav(self) -> None:
        server = pyo.Server(audio="offline").boot()
        server.recordOptions(dur=4.0, filename="simple_guitar.wav", fileformat=0)

        server.start()
        
        for action in self.actions:
            fret = action.fret if action.fret else 0
            freq = pyo.midiToHz(fret + self.tuning[action.string])
            
            env = pyo.Adsr(
                attack=0.01,
                decay=0.1,
                sustain=0.5,
                release=0.3,
                dur=2,
                mul=0.1
            )
            
            osc = pyo.Sine(freq=freq, mul=env).out(dur=2)

        # Let the render complete
        server.shutdown()
        print("Saved to simple_guitar.wav")
        
