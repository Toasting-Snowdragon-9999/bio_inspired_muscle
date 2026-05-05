from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

# Define gait phases for each foot in the order: FL, FR, RL, RR
class Gait:
    _registry = {}

    def __init__(self, name, phases):
        self.name = name
        self._value_ = phases
        Gait._registry[name] = self

    def __str__(self):
        return self.name

    @property
    def value(self):
        return self._value_

    def __eq__(self, other):
        if isinstance(other, Gait):
            return self._value_ == other._value_
        return False

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Gait.{self.name}"

    def __class_getitem__(cls, name):
        return cls._registry[name]

Gait.WALK   = Gait("WALK",   (3*np.pi/2, np.pi/2, 0.0, np.pi))
Gait.TROT   = Gait("TROT",   (0.0, np.pi, 0.0, np.pi))
Gait.AMBLE  = Gait("AMBLE",  (np.pi/2, 3*np.pi/2, 0.0, np.pi))
Gait.CANTER = Gait("CANTER", (1.3*np.pi, 0.85*np.pi, 0.0, 0.6*np.pi))
Gait.GALLOP = Gait("GALLOP", (0.0, 0.2*np.pi, 0.8*np.pi, np.pi))

# GAIT_NOMINAL_FREQUENCY = {
#     Gait.WALK: 1.4,
#     Gait.TROT: 1.95,
#     Gait.AMBLE: 1.7,
#     Gait.CANTER: 2.4,
#     Gait.GALLOP: 3.0,
# }

class Mode(Enum):
    MOVING = 1
    TRANSITION = 2
    STILL = 3

    def __str__(self):
        return self.name

class TrajectoryMethod(Enum):
    """Selects which foot trajectory shape the TrajectoryBuilder will produce.
    Set via robot_interface.trajectory_method = TrajectoryMethod.X at any time."""
    EGG       = auto()   # classic egg/ellipse — simple, no extra parameters needed
    OVAL      = auto()   # asymmetric oval    — requires oval_offsets dict on TrajectoryBuilder
    BEZIER    = auto()   # Bézier swing+stance — requires bezier_control_points (or uses defaults)
    ELLIPSOID = auto()   # rotatable ellipse with per-foot x-displacement — requires ellipsoid_config on TrajectoryBuilder

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
        self._enable_cpg = True
        self._cpg_alpha = 0.0
        self._cpg_transition_speed = 0.5
        self._body_velocity = 0.0
        self._target_speed = 0.5
        self._expected_footfall: dict[Foot, int] = {}  # Updated by CPG output for use in PD control and logging
        # Actual per-foot ground contact (1 = stance, 0 = swing). Synced every
        # mj_step from MuJoCo contact pairs in MujocoSim._sync_robot_interface.
        # Mirror partner of `_expected_footfall`: comparing the two yields the
        # gait_error term used by the RL reward (see sim/gait_env.py).
        self._contact: dict[Foot, int] = {}
        # Neutral default so the fuzzy controller has a sane input until a real
        # stability estimator is wired up. 0.75 falls in the 'stable' band.
        self._stability_metric = 0.75
        self._current_traj_params = None
        self._next_traj_params = None
    
    @property
    def trajectory_method(self) -> 'TrajectoryMethod':
        return self._trajectory_method

    @trajectory_method.setter
    def trajectory_method(self, method: 'TrajectoryMethod'):
        self._trajectory_method = method

    @property
    def stability_metric(self) -> float:
        return self._stability_metric
    
    @stability_metric.setter
    def stability_metric(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Stability metric must be between 0.0 and 1.0")
        self._stability_metric = value

    @property
    def current_traj_params(self):
        return self._current_traj_params
    
    @current_traj_params.setter
    def current_traj_params(self, params):
        self._current_traj_params = params

    @property
    def duty_factor(self) -> float:
        """Fraction of the gait cycle spent in stance (e.g. 0.5 = 50% stance).
        Change at any time: robot_interface.duty_factor = 0.6"""
        return self._duty_factor

    @duty_factor.setter
    def duty_factor(self, value: float):
        self._duty_factor = float(value)

    @property
    def expected_footfall(self):
        return self._expected_footfall

    @expected_footfall.setter
    def expected_footfall(self, value: dict[Foot, int]):
        self._expected_footfall = value

    @property
    def contact(self) -> dict[Foot, int]:
        """Actual per-foot ground contact (1 = stance, 0 = swing). Populated
        every mj_step by the simulator from MuJoCo contact pairs."""
        return self._contact

    @contact.setter
    def contact(self, value: dict[Foot, int]):
        self._contact = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def current_state(self) -> State:
        return self.robot_state.current_state if self.robot_state else None

    @current_state.setter
    def current_state(self, state: State):
        self.update_state(state)
    
    @property
    def next_state(self) -> State:
        return self.robot_state.next_state if self.robot_state else None
    
    @next_state.setter
    def next_state(self, state: State):
        if self.robot_state:
            self.robot_state.next_state = state
        else:
            raise ValueError("Cannot set next state without an existing robot state. Please initialize the robot state first.")

    @property
    def current_gait(self) -> Gait:
        return self.robot_state.current_state.gait if self.robot_state else None
    
    @current_gait.setter
    def current_gait(self, gait: Gait):
        self.update_gait(gait)
    
    @property
    def next_gait(self) -> Gait:
        return self.robot_state.next_state.gait if self.robot_state else None
    
    @next_gait.setter
    def next_gait(self, gait: Gait):
        if self.robot_state:
            self.robot_state.next_state = State(mode=self.robot_state.current_state.mode, gait=gait, frequency=self.robot_state.current_state.frequency)
        else:
            raise ValueError("Cannot set next gait without an existing robot state. Please initialize the robot state first.")

    @property
    def current_mode(self) -> Mode:
        return self.robot_state.current_state.mode if self.robot_state else None

    @current_mode.setter
    def current_mode(self, mode: Mode):
        self.update_mode(mode)

    @property
    def frequency(self) -> float:
        """Current CPG frequency in Hz. Settable: robot_interface.frequency = 2.0"""
        return self.robot_state.current_state.frequency if self.robot_state else 0.0

    @frequency.setter
    def frequency(self, value: float):
        self.robot_state.current_state.frequency = float(value)

    @property
    def next_frequency(self) -> float:
        """Next CPG frequency in Hz. Settable: robot_interface.next_frequency = 2.0"""
        return self.robot_state.next_state.frequency if self.robot_state else 0.0

    @next_frequency.setter
    def next_frequency(self, value: float):
        self.robot_state.next_state.frequency = float(value)

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
        return self._body_orientatio

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

    @property
    def enable_cpg(self) -> bool:
        return self._enable_cpg

    @enable_cpg.setter
    def enable_cpg(self, value: bool):
        self._enable_cpg = value

    @property
    def cpg_alpha(self) -> float:
        return self._cpg_alpha
    
    @cpg_alpha.setter
    def cpg_alpha(self, value: float):
        self._cpg_alpha = value

    @property
    def cpg_transition_speed(self) -> float:
        return self._cpg_transition_speed
    
    @cpg_transition_speed.setter
    def cpg_transition_speed(self, value: float):
        self._cpg_transition_speed = value
    
    @property
    def body_velocity(self) -> float:
        return self._body_velocity

    @body_velocity.setter
    def body_velocity(self, value: float):
        self._body_velocity = value
    
    @property
    def target_speed(self) -> float:
        return self._target_speed
    
    @target_speed.setter
    def target_speed(self, value: float):
        self._target_speed = value

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
        if self.robot_state is None:
            raise ValueError("Cannot update state without an existing robot state. Please initialize the robot state first.")
        
        if new_state is None:
            new_state = self.robot_state.next_state if self.robot_state else None

            if new_state is None:
                new_state = self.robot_state.current_state if self.robot_state else None
        
        self.robot_state.previous_state = self.robot_state.current_state
        self.robot_state.current_state = new_state
        self.robot_state.next_state = None

    def set_next_state(self, next_state: State):
        if self.robot_state:
            self.robot_state.next_state = next_state


def main():
    robot = RobotInterface(State(mode=Mode.STILL, gait=Gait.WALK, frequency=0.5))
    robot.update_state(State(mode=Mode.MOVING, gait=Gait.WALK, frequency=0.5))
    print("Current State:", robot.current_state)
    robot.set_next_state(State(mode=Mode.MOVING, gait=Gait.TROT, frequency=0.7))
    robot.update_state()
    print("Current State after transition:", robot.current_state)

    print("Gait phases", robot.current_state.gait.value)

if __name__ == "__main__":
    main()