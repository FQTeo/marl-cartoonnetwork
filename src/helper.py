import numpy as np
from src.types import Action, Direction

# def get_actions() -> list[Action]:

def get_bit(value: int, bit_index: int):
    """Get the value at a particular bit of the provided int"""
    return (value >> bit_index) & 1


def move(location: list[int], action: Action, direction: Direction) -> list[int]:
    pos = np.array(location)

    # Define relative movement offsets for each direction
    dir_vector = direction.movement
    left_vector = np.array([-dir_vector[1], dir_vector[0]])   # 90° counter-clockwise
    right_vector = np.array([dir_vector[1], -dir_vector[0]])  # 90° clockwise

    match action:
        case Action.FORWARD:
            pos += dir_vector
        case Action.BACKWARD:
            pos -= dir_vector
        case Action.LEFT:
            pos += left_vector
        case Action.RIGHT:
            pos += right_vector
        case Action.STAY:
            pass

    return pos.tolist()
