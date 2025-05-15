from enum import IntEnum
import numpy as np

class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def movement(self) -> np.ndarray:
        match self:
            case Direction.RIGHT:
                return np.array([1, 0])
            case Direction.DOWN:
                return np.array([0, 1])
            case Direction.LEFT:
                return np.array([-1, 0])
            case Direction.UP:
                return np.array([0, -1])