from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class GuitarActionType(Enum):
    FRET = auto()
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
    actions: List[GuitarAction] = field(default_factory=list)

    def add_action(self, action: GuitarAction):
        self.actions.append(action)

    def score_playability(self) -> float:
        """
        Placeholder for a function that scores how playable the sequence is.
        Returns:
            float: Playability score (higher is better)
        """
        pass
