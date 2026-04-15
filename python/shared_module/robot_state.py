from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

class Gait(Enum):
    # Define gait phases for each foot in the order: FL, FR, RL, RR

    # Primary gaits
    WALK   = (np.pi/2, 3*np.pi/2, 0.0, np.pi) 
    # WALK   = (0.0, np.pi/2, 3*np.pi/2, np.pi)  #  FL, FR,  RR, RL
    TROT   = (0.0, np.pi,   0.0,   np.pi)
    CANTER = (0.0, np.pi,   np.pi, 0.0)
    # BOUND  = (0.0, 0.0,     np.pi, np.pi)
    # GALLOP = (0.0, 0.0,     np.pi*0.8, np.pi*0.8)
    GALLOP = (2*np.pi*0.7, np.pi,     0.0, 2*np.pi*0.2)


    # Transitional     
    PACE   = (0.0, np.pi,   np.pi, 0.0)
    AMBLE = (0.0, np.pi/2, np.pi, np.pi/2) # TODO

    NONE   = (0.0, 0.0,     0.0,   0.0)

    def __str__(self):
        return self.name

    def __value__(self):
        return self._value_

class Mode(Enum):
    MOVING = 1
    TRANSITION = 2

    def __str__(self):
        return self.name

class TrajectoryMethod(Enum):
    """Selects which foot trajectory shape the TrajectoryBuilder will produce.
    Set via robot_interface.trajectory_method = TrajectoryMethod.X at any time."""
    EGG    = auto()   # classic egg/ellipse — simple, no extra parameters needed
    OVAL   = auto()   # asymmetric oval    — requires oval_offsets dict on TrajectoryBuilder
    BEZIER = auto()   # Bézier swing+stance — requires bezier_control_points (or uses defaults)
    ELLIPSOID = auto()

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

class Foot(Enum):
    FL = 'FL_foot'
    FR = 'FR_foot'
    RL = 'RL_foot'
    RR = 'RR_foot'
    
    def __str__(self):
        return self.name

class Hip(Enum):
    FL = 'FL_hip'
    FR = 'FR_hip'
    RL = 'RL_hip'
    RR = 'RR_hip' 

    def __str__(self):
        return self.name

class Thigh(Enum):
    FL = 'FL_thigh'
    FR = 'FR_thigh'
    RL = 'RL_thigh'
    RR = 'RR_thigh'

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

    def __init__(self, starting_state: State, trajectory_method: TrajectoryMethod = TrajectoryMethod.EGG, duty_factor: float = 0.5):
        self.robot_state = RobotState(current_state=starting_state, previous_state=None, next_state=None)
        self._joint_positions: dict[Joint, float] = {}
        self._joint_velocities: dict[Joint, float] = {}
        self._body_position: list[float] = []
        self._body_orientation: list[float] = []
        self._target_positions: dict[Joint, float] = {}
        self._target_velocities: dict[Joint, float] = {}
        self._target_torques: dict[Joint, float] = {}
        self._foot_positions: dict[Foot, list[float]] = {}
        self._stance_positions: dict[Foot, list[float]] = {}
        self._hip_position: dict[Hip, list[float]] = {}
        self._thigh_position: dict[Thigh, list[float]] = {}
        self._trajectory_method: TrajectoryMethod = trajectory_method  # default foot path shape
        self._duty_factor: float = duty_factor  # fraction of the cycle spent in stance (0–1)
        self._initialized = True
        self._dt = None

    @property
    def trajectory_method(self) -> 'TrajectoryMethod':
        return self._trajectory_method

    @trajectory_method.setter
    def trajectory_method(self, method: 'TrajectoryMethod'):
        self._trajectory_method = method

    @property
    def duty_factor(self) -> float:
        """Fraction of the gait cycle spent in stance (e.g. 0.5 = 50% stance).
        Change at any time: robot_interface.duty_factor = 0.6"""
        return self._duty_factor

    @duty_factor.setter
    def duty_factor(self, value: float):
        self._duty_factor = float(value)

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
    def foot_positions(self) -> dict[Foot, list[float]]:
        return self._foot_positions

    @foot_positions.setter
    def foot_positions(self, foot_positions: dict[Foot, list[float]]):
        self._foot_positions.update(foot_positions)

    @property
    def stance_positions(self) -> dict[Foot, list[float]]:
        return self._stance_positions

    @stance_positions.setter
    def stance_positions(self, stance_positions: dict[Foot, list[float]]):
        self._stance_positions.update(stance_positions)

    @property
    def hip_position(self) -> dict[Hip, list[float]]:
        return self._hip_position

    @hip_position.setter
    def hip_position(self, hip_position: dict[Hip, list[float]]):
        self._hip_position.update(hip_position)

    @property
    def thigh_position(self) -> dict[Thigh, list[float]]:
        return self._thigh_position

    @thigh_position.setter
    def thigh_position(self, thigh_position: dict[Thigh, list[float]]):
        self._thigh_position.update(thigh_position)

    @property
    def target_positions(self) -> dict[Joint, float]:
        return self._target_positions

    @target_positions.setter
    def target_positions(self, target_positions: dict[Joint, float]):
        self._target_positions.update(target_positions)

    @property
    def target_velocities(self) -> dict[Joint, float]:
        return self._target_velocities
    
    @target_velocities.setter
    def target_velocities(self, target_velocities: dict[Joint, float]):
        self._target_velocities.update(target_velocities)

    @property
    def target_torques(self) -> dict[Joint, float]:
        return self._target_torques
    
    @target_torques.setter
    def target_torques(self, target_torques: dict[Joint, float]):
        self._target_torques.update(target_torques)

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