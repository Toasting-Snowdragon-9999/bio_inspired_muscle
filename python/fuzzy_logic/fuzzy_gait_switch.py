import os,sys
from typing import Any
from enum import Enum, auto

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Gait, Mode, RobotInterface, State, ALL_GAITS
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

    def pick_trajectory(self, current_gait: Gait):
        # Placeholder for trajectory picking logic based on velocity and current gait
        # This can be expanded with more complex logic or fuzzy rules as needed
        return None

    def pick_frequency(self, current_gait: Gait):
        # Placeholder for frequency picking logic based on velocity and current gait
        # This can be expanded with more complex logic or fuzzy rules as needed
        return None

    def pick_gait(self, vel: VelCmd, current_gait: Gait) -> Gait:
        v = vel.v

        picked = current_gait
        if current_gait == Gait.WALK:
            if v > 0.35:
                picked = Gait.AMBLE

        elif current_gait == Gait.AMBLE:
            if v < 0.25:
                picked = Gait.WALK
            elif v > 0.55:
                picked = Gait.TROT

        elif current_gait == Gait.TROT:
            if v < 0.45:
                picked = Gait.AMBLE
            elif v > 0.75:
                picked = Gait.CANTER

        elif current_gait == Gait.CANTER:
            if v < 0.65:
                picked = Gait.TROT
            elif v > 0.9:
                picked = Gait.GALLOP

        elif current_gait == Gait.GALLOP:
            if v < 0.85:
                picked = Gait.CANTER

        return picked

    def pick_new_state(self, vel: VelCmd, current_state: State) -> State:
        new_gait = self.pick_gait(vel, current_state.gait)
        new_traj_params = self.pick_trajectory(current_state.gait)
        new_frequency = self.pick_frequency(current_state.gait)
        return State(mode=current_state.mode, gait=new_gait, traj_params=new_traj_params, frequency=new_frequency)

class FuzzyGaitSwitch:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        # Define fuzzy variables
        # self.vel_cmd = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'velocity')
        self.stability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'stability')
        self.speed_error = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'speed_error')

        # self.frequency = ctrl.Consequent(np.arange(0, 4.0, 0.01), 'frequency')
        self.blend_rate = ctrl.Consequent(np.arange(0.0, 1.01, 0.01), 'blend_rate')
        self.blending_factor = 0.0  # Initialize blending factor


        self._origin_gait: 'Gait | None' = None
        self._target_gait: 'Gait | None' = None

        self.stability['very_unstable'] = fuzz.trapmf(self.stability.universe, [0.0, 0.0, 0.2, 0.4])
        self.stability['unstable']      = fuzz.trimf(self.stability.universe, [0.3, 0.5, 0.7])
        self.stability['stable']        = fuzz.trimf(self.stability.universe, [0.6, 0.75, 0.9])
        self.stability['very_stable']   = fuzz.trapmf(self.stability.universe, [0.8, 0.9, 1.0, 1.0])

        self.speed_error['low']    = fuzz.trapmf(self.speed_error.universe, [0.0, 0.0, 0.2, 0.4])
        self.speed_error['medium'] = fuzz.trimf(self.speed_error.universe, [0.3, 0.5, 0.7])
        self.speed_error['high']   = fuzz.trapmf(self.speed_error.universe, [0.6, 0.8, 1.0, 1.0])

        self.blend_rate['backward_fast'] = fuzz.trapmf(self.blend_rate.universe, [0.00, 0.00, 0.15, 0.35])
        self.blend_rate['backward_slow'] = fuzz.trimf(self.blend_rate.universe, [0.25, 0.40, 0.60])
        self.blend_rate['hold'] = fuzz.trimf(self.blend_rate.universe, [0.45, 0.50, 0.65])
        self.blend_rate['forward_slow'] = fuzz.trimf(self.blend_rate.universe, [0.50, 0.65, 0.75])
        self.blend_rate['forward_fast'] = fuzz.trapmf(self.blend_rate.universe, [0.70, 0.80, 1.00, 1.00])

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

        all_rules = [
            self.high_stability_rule1, self.high_stability_rule2, self.high_stability_rule3,
            self.stability_rule1,      self.stability_rule2,      self.stability_rule3,
            self.unstable_rule1,       self.unstable_rule2,       self.unstable_rule3,
            self.very_unstable_rule1,  self.very_unstable_rule2,  self.very_unstable_rule3,
        ]
        self.fuzzy_ctrl = ctrl.ControlSystem(all_rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)

    def show_membership_functions(self):
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
        prev = self.blending_factor
        self.blending_factor = float(np.clip(self.blending_factor + blend_rate_output * dt, 0.0, 1.0))
        return self.blending_factor

    def interpolate(self, a: float, b: float, s: float) -> float:
        """Linear interpolation between a and b with blending factor s."""
        return (1 - s) * a + s * b

    def blend_gaits(self, old_gait: Gait, new_gait: Gait, blending_factor: float) -> Gait:
        trans_gait_values = []
        for a, b in zip(old_gait.value, new_gait.value):
            trans_gait_values.append(self.interpolate(a, b, blending_factor))
        blended = Gait("BLENDED_GAIT", tuple(trans_gait_values))
        return blended

    def blend_trajectories(self, old_traj_params, new_traj_params, blending_factor) -> Any:
        if old_traj_params is None and new_traj_params is None:
            return None
        if new_traj_params is None:
            return old_traj_params
        if old_traj_params is None:
            return new_traj_params

        # Dataclass path (e.g. EllipsoidConfig): blend each declared field.
        from dataclasses import is_dataclass, fields, replace
        if is_dataclass(old_traj_params) and is_dataclass(new_traj_params):
            blended_kwargs = {}
            for f in fields(old_traj_params):
                a = getattr(old_traj_params, f.name)
                b = getattr(new_traj_params, f.name, a)
                blended_kwargs[f.name] = self.interpolate(float(a), float(b), blending_factor)
            return replace(old_traj_params, **blended_kwargs)

        # Dict path: iterate the old keys; missing keys in `new` fall back
        # to old's value so we never KeyError mid-blend.
        if isinstance(old_traj_params, dict):
            blended = {}
            for key, old_val in old_traj_params.items():
                new_val = new_traj_params.get(key, old_val) if isinstance(new_traj_params, dict) else old_val
                blended[key] = self.interpolate(float(old_val), float(new_val), blending_factor)
            return blended

        # Unknown shape — return old to keep the system stable rather than
        # crashing the control loop.
        return old_traj_params

    @property
    def is_transitioning(self) -> bool:
        """``True`` while a gait transition is mid-blend (factor in (0, 1))."""
        return self._target_gait is not None

    def update(self, dt: float) -> tuple:

        ri = self.robot_interface

        # ----- branch 1: already mid-transition. Keep stepping the locked target.
        if self.is_transitioning:
            return self._step_transition(dt)

        current_gait = ri.current_gait
        current_traj = ri.current_traj_params
        current_freq = ri.frequency

        next_state = ri.robot_state.next_state if ri.robot_state else None
        target_gait = next_state.gait if next_state is not None else None

        if target_gait is None or current_gait == target_gait:
    
            if ri.current_mode == Mode.TRANSITION:
                ri.update_mode(Mode.MOVING)
            return current_gait, current_traj, current_freq, True

        self._origin_gait = current_gait
        self._target_gait = target_gait
        self.blending_factor = 0.0
        return self._step_transition(dt)

    def _step_transition(self, dt: float) -> tuple:

        ri = self.robot_interface
        if ri.current_mode != Mode.TRANSITION:
            ri.update_mode(Mode.TRANSITION)

        speed_error_in = 0.0 #ri.target_speed - ri.body_velocity
        stability_in = ri.stability_metric

        self.fuzzy_sim.input['speed_error'] = float(np.clip(abs(speed_error_in), 0.0, 1.0))
        self.fuzzy_sim.input['stability'] = float(np.clip(stability_in, 0.0, 1.0))
        self.fuzzy_sim.compute()
        blend_rate_output = float(self.fuzzy_sim.output['blend_rate'])
        blending_factor = self.update_blending_factor(blend_rate_output, dt)

        gait_blend = self.blend_gaits(self._origin_gait, self._target_gait, blending_factor)
        traj_blend = self.blend_trajectories(
            ri.current_traj_params, ri.next_traj_params, blending_factor,
        )

        next_freq = ri.next_frequency
        freq_blend = self.interpolate(ri.frequency, next_freq, blending_factor) if next_freq else ri.frequency

        if blending_factor >= 1.0:

            committed_gait = self._target_gait
            self._origin_gait = None
            self._target_gait = None
            self.blending_factor = 0.0
            ri.update_mode(Mode.MOVING)
            return committed_gait, ri.next_traj_params or traj_blend, next_freq or freq_blend, True

        return gait_blend, traj_blend, freq_blend, False

class FuzzyController:
    """Orchestrator: GaitPicker chooses a target, FuzzyGaitSwitch blends toward it.

    ``params_for_gait`` and ``freq_for_gait`` are optional callables the
    integrator (typically the GUI frontend) can supply so this controller
    can populate ``robot_interface.next_traj_params`` /
    ``next_frequency`` whenever a target gait is staged. Without them the
    fuzzy switch would have no data to blend toward, leaving the
    transition stuck on whatever the IK builder last wrote. Defaults are
    ``None`` so existing call sites keep working — just without staged
    blending.
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        params_for_gait=None,   # callable: Gait -> EllipsoidConfig|dict|None
        freq_for_gait=None,     # callable: Gait -> float|None
        duty_for_gait=None,     # callable: Gait -> float|None
    ):
        self.robot_interface = robot_interface
        self.gait_picker = GaitPicker()
        self.fuzzy_gait_switch = FuzzyGaitSwitch(robot_interface)
        self.params_for_gait = params_for_gait
        self.freq_for_gait = freq_for_gait
        self.duty_for_gait = duty_for_gait

        self._origin_duty: 'float | None' = None
        self._target_duty: 'float | None' = None

    def update(self, dt: float) -> None:
        ri = self.robot_interface

        if not self.fuzzy_gait_switch.is_transitioning:
            vel_cmd = VelCmd(ri.target_speed)
            transition_gait = self.gait_picker.pick_gait(vel_cmd, ri.current_gait)
            ri.next_gait = transition_gait

            if self.params_for_gait is not None:
                try:
                    ri.next_traj_params = self.params_for_gait(transition_gait)
                except Exception:
                    # Defensive: don't let a missing-gait lookup kill the loop.
                    pass
            if self.freq_for_gait is not None:
                try:
                    staged_freq = self.freq_for_gait(transition_gait)
                    if staged_freq is not None:
                        ri.next_frequency = float(staged_freq)
                except Exception:
                    pass

            if self.duty_for_gait is not None:
                try:
                    target_duty = self.duty_for_gait(transition_gait)
                    if target_duty is not None:
                        self._origin_duty = float(ri.duty_factor)
                        self._target_duty = float(target_duty)
                except Exception:
                    pass

        new_gait, new_traj, new_freq, committed = self.fuzzy_gait_switch.update(dt)


        if new_traj is not None:
            ri.current_traj_params = new_traj
        if new_freq:
            ri.frequency = new_freq


        if committed and new_gait is not None:
            ri.current_gait = new_gait

        if self._target_duty is not None and self._origin_duty is not None:
            if not self.fuzzy_gait_switch.is_transitioning and committed:
                ri.duty_factor = self._target_duty
                self._origin_duty = None
                self._target_duty = None

def test_fuzzy_gait_switch():
    """Observe the WALK -> AMBLE fuzzy gait transition over 1 second.

    No simulator is involved: a RobotInterface is configured with the
    inputs the GaitPicker needs to route WALK -> AMBLE, then
    FuzzyGaitSwitch.update() is driven at a fixed dt for 1.0 s of
    wall-time and every input, fuzzy output, blending_factor, and
    blended phase tuple is logged.

    NOTE: update_gait() is intentionally NOT called between ticks.
    Calling it would replace current_gait with a 'BLENDED_GAIT'
    instance after the first tick; the GaitPicker has no branch for
    that label, so subsequent ticks would short-circuit and no further
    blend progression would be observable.
    """
    logger.debug("=" * 80)
    logger.debug("test_fuzzy_gait_switch: WALK -> AMBLE, 1 second @ dt=0.05")
    logger.debug("=" * 80)

    starting_state = State(mode=Mode.MOVING, gait=Gait.WALK, frequency=1.0)
    robot_interface = RobotInterface(starting_state)
    robot_interface.target_speed = 0.5        # > 0.35 -> GaitPicker routes WALK -> AMBLE
    robot_interface.body_velocity = 0.0       # max speed_error -> larger fuzzy blend_rate
    robot_interface.stability_metric = 0.85   # 'very_stable' band -> forward blend

    logger.debug(
        f"[setup] mode={robot_interface.current_mode}, gait={robot_interface.current_gait}, "
        f"target_speed={robot_interface.target_speed}, body_velocity={robot_interface.body_velocity}, "
        f"stability_metric={robot_interface.stability_metric}"
    )
    logger.debug(
        f"[setup] reference phases: WALK={tuple(round(x, 4) for x in Gait.WALK.value)}, "
        f"AMBLE={tuple(round(x, 4) for x in Gait.AMBLE.value)}"
    )

    picker = GaitPicker()
    vel_cmd = VelCmd(robot_interface.target_speed)
    target = picker.pick_gait(vel_cmd, robot_interface.current_gait)
    robot_interface.next_gait = target
    logger.debug(f"[setup] picker target_gait={target}; robot_interface.next_gait now set")

    fuzzy = FuzzyGaitSwitch(robot_interface)
    logger.debug(f"[setup] FuzzyGaitSwitch instantiated, blending_factor={fuzzy.blending_factor:.4f}")

    dt = 0.05
    total_time = 1.0
    n_steps = int(round(total_time / dt))
    logger.debug(f"[setup] loop config: dt={dt}, total_time={total_time}, n_steps={n_steps}")
    transitioned = False
    i = 0
    while(not transitioned):
        t = (i + 1) * dt
        logger.debug("-" * 80)
        logger.debug(f"[step {i+1:02d}/{n_steps}] t={t:.3f}s")
        # FuzzyGaitSwitch.update now always returns a 4-tuple
        # (gait, traj_params, frequency, committed). The test only
        # inspects the blended gait, so unpack and ignore the rest.
        blended_gait, _blended_traj, _blended_freq, _committed = fuzzy.update(dt)
        logger.debug(
            f"[step {i+1:02d}] post-update: blending_factor={fuzzy.blending_factor:.4f}, "
            f"blended_phases={tuple(round(x, 4) for x in blended_gait.value)}, "
            f"current_mode={robot_interface.current_mode}, current_gait={robot_interface.current_gait}"
        )
        i+=1
        if blended_gait.value == Gait.AMBLE.value:
            transitioned = True

    logger.debug("=" * 80)
    logger.debug(
        f"test_fuzzy_gait_switch DONE: final blending_factor={fuzzy.blending_factor:.4f}"
    )
    logger.debug("=" * 80)

if __name__ == "__main__":
    test_fuzzy_gait_switch()