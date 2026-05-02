import os,sys
from typing import Any
from enum import Enum, auto

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Gait, Mode, RobotInterface
from logger.logger_config import logger

MAX_VELOCITY = 1.0
MIN_VELOCITY = 0.0

# class GaitIndex(Enum):
#     WALK = 0
#     AMBLE = 1
#     TROT = 2
#     CANTER = 3
#     GALLOP = 4
#     MAX_GAIT_INDEX = auto()

# gait_to_index = {
#     Gait.WALK: GaitIndex.WALK,
#     Gait.AMBLE: GaitIndex.AMBLE,
#     Gait.TROT: GaitIndex.TROT,
#     Gait.CANTER: GaitIndex.CANTER,
#     Gait.GALLOP: GaitIndex.GALLOP,
#     Gait.NONE: GaitIndex.MAX_GAIT_INDEX
# }

# # index_to_gait = {v: k for k, v in gait_to_index.items()}
# index_to_gait = {
#     GaitIndex.WALK: Gait.WALK,
#     GaitIndex.AMBLE: Gait.AMBLE,
#     GaitIndex.TROT: Gait.TROT,
#     GaitIndex.CANTER: Gait.CANTER,
#     GaitIndex.GALLOP: Gait.GALLOP,
#     GaitIndex.MAX_GAIT_INDEX: Gait.NONE
# }

class VelType(Enum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"

class VelCmd:
    def __init__(self, velocity: float):
        if velocity < MIN_VELOCITY or velocity > MAX_VELOCITY:
            logger.error("Velocity must be between 0 and 1, capping to valid range")
            velocity = max(MIN_VELOCITY, min(MAX_VELOCITY, velocity))
        self._v = velocity
        self._type = self._classify_velocity()

    def _classify_velocity(self) -> VelType:
        if self._v < 0.3:
            return VelType.SLOW
        elif self._v < 0.7:
            return VelType.MEDIUM
        else:
            return VelType.FAST

    @property
    def v(self) -> float:
        return self._v

    @property
    def type(self) -> VelType:
        return self._type

    @v.setter
    def v(self, value: float):
        if value < MIN_VELOCITY or value > MAX_VELOCITY:
            logger.error("Velocity must be between 0 and 1, capping to valid range")
            value = max(MIN_VELOCITY, min(MAX_VELOCITY, value))
        self._v = value 
        self._type = self._classify_velocity()


class GaitPicker:
    def pick_gait(self, vel: VelCmd, current_gait: Gait) -> Gait:
        v = vel.v

        if current_gait == Gait.WALK:
            if v > 0.35:
                return Gait.AMBLE

        elif current_gait == Gait.AMBLE:
            if v < 0.25:
                return Gait.WALK
            elif v > 0.55:
                return Gait.TROT

        elif current_gait == Gait.TROT:
            if v < 0.45:
                return Gait.AMBLE
            elif v > 0.75:
                return Gait.CANTER

        elif current_gait == Gait.CANTER:
            if v < 0.65:
                return Gait.TROT
            elif v > 0.9:
                return Gait.GALLOP

        elif current_gait == Gait.GALLOP:
            if v < 0.85:
                return Gait.CANTER

        return current_gait

class FuzzyGaitSwitch:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        # Define fuzzy variables
        # self.vel_cmd = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'velocity')
        self.stability = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'stability')
        self.speed_error = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'speed_error')

        # self.frequency = ctrl.Consequent(np.arange(0, 4.0, 0.01), 'frequency')
        self.blend_rate = ctrl.Consequent(np.arange(-1.0, 1.01, 0.01), 'blend_rate')
        self.blending_factor = 0.0  # Initialize blending factor

        self.stability['very_unstable'] = fuzz.trapmf(self.stability.universe, [0.0, 0.0, 0.2, 0.4])
        self.stability['unstable']      = fuzz.trimf(self.stability.universe, [0.3, 0.5, 0.7])
        self.stability['stable']        = fuzz.trimf(self.stability.universe, [0.6, 0.75, 0.9])
        self.stability['very_stable']   = fuzz.trapmf(self.stability.universe, [0.8, 0.9, 1.0, 1.0])

        self.speed_error['low']    = fuzz.trapmf(self.speed_error.universe, [0.0, 0.0, 0.2, 0.4])
        self.speed_error['medium'] = fuzz.trimf(self.speed_error.universe, [0.3, 0.5, 0.7])
        self.speed_error['high']   = fuzz.trapmf(self.speed_error.universe, [0.6, 0.8, 1.0, 1.0])

        self.blend_rate['backward_fast'] = fuzz.trapmf(self.blend_rate.universe, [-1.00, -1.00, -0.70, -0.45])
        self.blend_rate['backward_slow'] = fuzz.trimf(self.blend_rate.universe, [-0.65, -0.32, -0.05])
        self.blend_rate['hold'] = fuzz.trimf(self.blend_rate.universe, [-0.10, 0.00, 0.10])
        self.blend_rate['forward_slow'] = fuzz.trimf(self.blend_rate.universe, [0.05, 0.32, 0.65])
        self.blend_rate['forward_fast'] = fuzz.trapmf(self.blend_rate.universe, [0.45, 0.70, 1.00, 1.00])

        # Define the fuzzy rules
        self.high_stability_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['very_stable'], self.blend_rate['hold'])
        self.high_stability_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['very_stable'], self.blend_rate['forward_slow'])
        self.high_stability_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['very_stable'], self.blend_rate['forward_fast'])

        self.stability_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['stable'], self.blend_rate['hold'])
        self.stability_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['stable'], self.blend_rate['forward_slow'])
        self.stability_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['stable'], self.blend_rate['forward_slow'])

        self.unstable_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['unstable'], self.blend_rate['backward_slow'])
        self.unstable_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['unstable'], self.blend_rate['backward_slow'])
        self.unstable_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['unstable'], self.blend_rate['backward_fast'])

        self.very_unstable_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['very_unstable'], self.blend_rate['backward_slow'])
        self.very_unstable_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['very_unstable'], self.blend_rate['backward_fast'])
        self.very_unstable_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['very_unstable'], self.blend_rate['backward_fast'])
        
        # Define the system — collect every Rule defined above into one
        # ControlSystem and store the simulation on `self` so update() can
        # drive it. (Previously this was a local with a typo'd rule list,
        # which is why `self.fuzzy_sim` didn't exist at update time.)
        all_rules = [
            self.high_stability_rule1, self.high_stability_rule2, self.high_stability_rule3,
            self.stability_rule1,      self.stability_rule2,      self.stability_rule3,
            self.unstable_rule1,       self.unstable_rule2,       self.unstable_rule3,
            self.very_unstable_rule1,  self.very_unstable_rule2,  self.very_unstable_rule3,
        ]
        self.fuzzy_ctrl = ctrl.ControlSystem(all_rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)

    def show_membership_functions(self):
        # Lazy import: matplotlib.pyplot is heavy and binds a backend on first
        # import, so keep it out of module scope.
        import matplotlib.pyplot as plt
        self.stability.view()
        self.speed_error.view()
        self.blend_rate.view()
        plt.show()

    def update_blending_factor(self, blend_rate_output: float, dt: float) -> float:
        """Integrate the fuzzy blend_rate into the persistent blending_factor.

        `dt` is the wall time elapsed since the previous call — when this
        method is driven from a decimated tick (e.g. every 30 sim steps),
        pass `decimated_steps * sim_dt`, NOT `sim_dt`, otherwise the
        integration runs ~30x slower than intended.
        """
        self.blending_factor = float(np.clip(self.blending_factor + blend_rate_output * dt, 0.0, 1.0))
        return self.blending_factor

    def interpolate(self, a: float, b: float, s: float) -> float:
        """Linear interpolation between a and b with blending factor s."""
        return (1 - s) * a + s * b

    def blend_gaits(self, old_gait: Gait, new_gait: Gait, blending_factor: float) -> Any:
        trans_gait_values = []
        for a, b in zip(old_gait.value, new_gait.value):
            trans_gait_values.append(self.interpolate(a, b, blending_factor))
        return Gait("BLENDED_GAIT", tuple(trans_gait_values))

    def blend_trajectories(self, old_traj_params, new_traj_params, blending_factor) -> Any:
        pass

    def update(self, dt: float) -> Gait:
        # `dt` is the elapsed time since the last call — when driven from a
        # decimated tick this is the accumulated dt (decimated_steps * sim_dt),
        # NOT the underlying sim timestep.
        current_gait = self.robot_interface.current_gait
        # `next_gait` accessor crashes when next_state is None (it does
        # `next_state.gait` unconditionally), so guard at the source instead.
        next_state = self.robot_interface.robot_state.next_state if self.robot_interface.robot_state else None
        target_gait = next_state.gait if next_state is not None else None
        if target_gait is None or current_gait == target_gait:
            return current_gait
        
        self.robot_interface.current_mode = Mode.TRANSITION
        self.fuzzy_sim.input['speed_error'] = self.robot_interface.target_speed - self.robot_interface.body_velocity
        self.fuzzy_sim.input['stability'] = self.robot_interface.stability_metric

        self.fuzzy_sim.compute()
        blend_rate_output = self.fuzzy_sim.output['blend_rate']
        blending_factor = self.update_blending_factor(blend_rate_output, dt)
        gait_params = self.blend_gaits(self.robot_interface.current_gait, target_gait, blending_factor)
        # traj_params = self.blend_trajectories(, blending_factor)
        return current_gait

class FuzzyController:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        self.gait_picker = GaitPicker()
        self.fuzzy_gait_switch = FuzzyGaitSwitch(robot_interface)

    def update(self, dt: float) -> None:
        """Decimated fuzzy tick. `dt` is the wall time elapsed since the
        previous call (set by the caller), not the sim timestep — the
        downstream integrator needs the real interval."""
        vel_cmd = VelCmd(self.robot_interface.target_speed)
        transition_gait = self.gait_picker.pick_gait(vel_cmd, self.robot_interface.current_gait)
        self.robot_interface.next_gait = transition_gait

        new_gait = self.fuzzy_gait_switch.update(dt)
        self.robot_interface.update_gait(new_gait)