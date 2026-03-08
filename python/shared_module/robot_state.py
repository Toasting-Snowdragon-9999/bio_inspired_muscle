from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class Gait(Enum):
    WALK   = (0.0, np.pi/2, np.pi, 3*np.pi/2)
    TROT   = (0.0, np.pi,   0.0,   np.pi)
    BOUND  = (0.0, 0.0,     np.pi, np.pi)
    PACE   = (0.0, np.pi,   np.pi, 0.0)
    GALLOP = (0.0, 0.0,     np.pi*0.8, np.pi*0.8)
    NONE   = (0.0, 0.0,     0.0,   0.0)

    def __str__(self):
        return self.name


class Mode(Enum):
    STILL = 1
    MOVING = 2
    TRANSITION = 3
    DOWN = 4

    def __str__(self):
        return self.name

@dataclass
class State:
    mode: Mode
    gait: Gait
    frequency: float = 0.0

    def __str__(self):
        return f"State(mode={self.mode}, gait={self.gait}, frequency={self.frequency})"


class Joint(Enum):
    # Front legs
    FR_HIP = auto()
    FR_THIGH = auto()
    FR_CALF = auto()
    FL_HIP = auto()
    FL_THIGH = auto()
    FL_CALF = auto()
    # Rear legs
    RR_HIP = auto()
    RR_THIGH = auto()
    RR_CALF = auto()
    RL_HIP = auto()
    RL_THIGH = auto()
    RL_CALF = auto()

    def __str__(self):
        return self.name


@dataclass
class RobotState:
    """
    Represents the state of the robot at a given time step.
    """
    current_state: State
    previous_state: State
    next_state: State


class RobotInterface:
    """
    Manages the robot's state over time, allowing for updates and retrievals.
    TODO: Add safeguard to prevent invalid state transitions (e.g., WALK -> BOUND without TROT).
    """

    def __init__(self, starting_state: State):
        self.robot_state = RobotState(current_state=starting_state, previous_state=None, next_state=None)
        self._joint_positions: dict[Joint, float] = {}
        self._joint_velocities: dict[Joint, float] = {}
        self._body_position: list[float] = []
        self._body_orientation: list[float] = []
        self._target_positions: dict[Joint, float] = {}
        self._initialized = True
        self._dt = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def current_gait(self) -> Gait:
        return self.robot_state.current_state.gait if self.robot_state else None

    @property
    def current_mode(self) -> Mode:
        return self.robot_state.current_state.mode if self.robot_state else None

    @property
    def current_state(self) -> State:
        return self.robot_state.current_state if self.robot_state else None

    @property
    def prev_state(self) -> State:
        return self.robot_state.previous_state if self.robot_state else None
    
    @property
    def next_state(self) -> State:
        return self.robot_state.next_state if self.robot_state else None

    @property
    def joint_positions(self) -> dict[Joint, float]:
        return self._joint_positions
    
    @property
    def joint_velocities(self) -> dict[Joint, float]:
        return self._joint_velocities

    @property
    def body_position(self) -> list[float]:
        return self._body_position

    @property
    def body_orientation(self) -> list[float]:
        return self._body_orientation
    
    @joint_positions.setter
    def joint_positions(self, joint_positions: dict[Joint, float]):
        self._joint_positions.update(joint_positions)

    @joint_velocities.setter
    def joint_velocities(self, joint_velocities: dict[Joint, float]):
        self._joint_velocities.update(joint_velocities)

    @body_position.setter
    def body_position(self, body_position: list[float]):
        self._body_position = body_position

    @body_orientation.setter
    def body_orientation(self, body_orientation: list[float]):
        self._body_orientation = body_orientation

    @property
    def target_positions(self) -> dict[Joint, float]:
        return self._target_positions

    @target_positions.setter
    def target_positions(self, target_positions: dict[Joint, float]):
        self._target_positions.update(target_positions)

    def update_mode(self, new_mode: Mode):
        if self.robot_state:
            new_state = State(mode=new_mode, gait=self.robot_state.current_state.gait, frequency=self.robot_state.current_state.frequency)
            self.update_state(new_state)
        else:
            raise ValueError("Cannot update mode without an existing robot state. Please initialize the robot state first.")

    def update_gait(self, new_gait: Gait):
        if self.robot_state:
            new_state = State(mode=self.robot_state.current_state.mode, gait=new_gait, frequency=self.robot_state.current_state.frequency)
            self.update_state(new_state)
        else:
            raise ValueError("Cannot update gait without an existing robot state. Please initialize the robot state first.")

    def update_state(self, new_state: State = None):
        """Update the robot's state. If new_state is None, it will attempt to use the next_state 
        from the current RobotState, if this fails it will raise a ValueError."""
        if new_state is None:
            new_state = self.robot_state.next_state if self.robot_state else None
            self.robot_state.next_state = None
            if new_state is None:
                raise ValueError("No new state provided and no next state available.")
        
        if self.robot_state:
            self.robot_state.previous_state = self.robot_state.current_state

        self.robot_state.current_state = new_state

    def set_next_state(self, next_state: State):
        if self.robot_state:
            self.robot_state.next_state = next_state

    def get_current_state(self) -> State:
        return self.robot_state.current_state if self.robot_state else None


def main():
    robot = RobotInterface(State(mode=Mode.STILL, gait=Gait.WALK, frequency=0.5))
    robot.update_state(State(mode=Mode.MOVING, gait=Gait.WALK, frequency=0.5))
    print("Current State:", robot.get_current_state())
    robot.set_next_state(State(mode=Mode.MOVING, gait=Gait.TROT, frequency=0.7))
    robot.update_state()
    print("Current State after transition:", robot.get_current_state())

    print("Gait phases", robot.get_current_state().gait.value)

if __name__ == "__main__":
    main()