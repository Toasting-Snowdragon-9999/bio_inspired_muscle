import os,sys
from typing import Any
from enum import Enum, auto

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Gait, Mode, RobotInterface, State, GaitFreq, ALL_GAITS
from logger.logger_config import logger
from cpg.trajectory_builder import EllipsoidConfig

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
    
    @staticmethod
    def pick_freq(current_gait: Gait):
        if current_gait == Gait.WALK:
            return GaitFreq.WALK.value
        elif current_gait == Gait.AMBLE:
            return GaitFreq.AMBLE.value
        elif current_gait == Gait.TROT:
            return GaitFreq.TROT.value
        elif current_gait == Gait.CANTER:
            return GaitFreq.CANTER.value
        elif current_gait == Gait.GALLOP:
            return GaitFreq.GALLOP.value
        else:
            return None
        
    @staticmethod
    def pick_gait(vel: VelCmd, current_gait: Gait) -> Gait:
        v = vel.v

        picked = current_gait
        if current_gait == Gait.WALK:
            if v > 0.25:
                picked = Gait.AMBLE

        elif current_gait == Gait.AMBLE:
            if v < 0.20:
                picked = Gait.WALK
            elif v > 0.30:
                picked = Gait.TROT

        elif current_gait == Gait.TROT:
            if v < 0.25:
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
        new_frequency = self.pick_freq(current_state.gait)
        return State(mode=current_state.mode, gait=new_gait, traj_params=new_traj_params, frequency=new_frequency)

class FuzzyGaitSwitch:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        self.a = None
        # Define fuzzy variables
        # self.vel_cmd = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'velocity')
        self.stability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'stability')
        self.speed_error = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'speed_error')

        # self.frequency = ctrl.Consequent(np.arange(0, 4.0, 0.01), 'frequency')
        self.blend_rate = ctrl.Consequent(np.arange(0.0, 1.01, 0.01), 'blend_rate')
        self.blending_factor = 0.0  # Initialize blending factor

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
    
    def interp_angle(self, a: float, b: float, s: float) -> float:

        delta = (b - a + np.pi) % (2*np.pi) - np.pi

        return a + s * delta

    def blend_trajectories(
        self,
        old_traj_params: EllipsoidConfig,
        new_traj_params: EllipsoidConfig,
        blending_factor: float
    ) -> EllipsoidConfig:

        blended_values = {}

        for key in old_traj_params.keys():
            old_val = old_traj_params[key]
            new_val = new_traj_params[key]

            blended_values[key] = self.interp_angle(
                old_val,
                new_val,
                blending_factor
            )

        return EllipsoidConfig(**blended_values)

    def update(self, dt: float) -> Gait:
        # `dt` is the elapsed time since the last call — when driven from a
        # decimated tick this is the accumulated dt (decimated_steps * sim_dt),
        # NOT the underlying sim timestep.
        current_gait = self.robot_interface.current_gait
        # `next_gait` accessor crashes when next_state is None (it does
        # `next_state.gait` unconditionally), so guard at the source instead.
        target_gait = self.robot_interface.next_gait

        if target_gait is None or current_gait == target_gait:
            self.a = None
            if self.robot_interface.current_mode == Mode.TRANSITION:
                self.robot_interface.update_mode(Mode.MOVING)
            return (
                current_gait,
                self.robot_interface.frequency,
                None
            )
        if self.a is None:
            self.a = self.robot_interface.frequency # capture the frequency at the start of the transition for blending
            self.blending_factor = 0.0 # reset blending factor at the start of a new transition
        self.robot_interface.update_mode(Mode.TRANSITION)
        speed_error_in = self.robot_interface.target_speed - self.robot_interface.body_velocity
        stability_in = self.robot_interface.stability_metric
        self.fuzzy_sim.input['speed_error'] = speed_error_in
        self.fuzzy_sim.input['stability'] = stability_in

        self.fuzzy_sim.compute()
        blend_rate_output = self.fuzzy_sim.output['blend_rate']
        blending_factor = self.update_blending_factor(blend_rate_output, dt)
        gait_params = self.blend_gaits(self.robot_interface.current_gait, target_gait, blending_factor)
        next_freq = GaitPicker.pick_freq(target_gait)
        frequency = self.interpolate(self.a, next_freq, blending_factor)
        return gait_params, frequency, blending_factor

class FuzzyController:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        self.fuzzy_gait_switch = FuzzyGaitSwitch(robot_interface)

    def update(self, dt: float) -> float:
        
        vel_cmd = VelCmd(self.robot_interface.target_speed)
        if vel_cmd.v < 0.05:
            self.robot_interface.enable_cpg(False)
            self.robot_interface.next_gait = Gait.WALK
            return 0.0
        transition_gait = GaitPicker.pick_gait(vel_cmd, self.robot_interface.current_gait)
        self.robot_interface.next_gait = transition_gait
        new_gait, new_freq, blending_factor = self.fuzzy_gait_switch.update(dt)
        new_gait = self.check_gait(new_gait)

        if blending_factor is not None and blending_factor >= 0.999:
            logger.debug(
                f"Transition complete: "
                f"{self.robot_interface.current_gait} "
                f"-> {self.robot_interface.next_gait}"
            )
            self.robot_interface.current_gait = self.robot_interface.next_gait
            self.robot_interface.active_gait = self.robot_interface.next_gait
            self.current_traj_params = self.robot_interface.next_traj_params
            self.robot_interface.active_traj_params = self.robot_interface.next_traj_params
            self.robot_interface.next_gait = None
            self.robot_interface.update_mode(Mode.MOVING)
            return 1.0

        self.robot_interface.active_gait = new_gait
        self.robot_interface.frequency = new_freq
        return blending_factor
    
    def update_trajectory(self, dt: float, blending_factor: float) -> None:
        self.robot_interface.active_traj_params =  self.fuzzy_gait_switch.blend_trajectories(self.robot_interface.current_traj_params, self.robot_interface.next_traj_params, blending_factor)

    def check_gait(self, gait: Gait) -> Gait:
        for known_gait in ALL_GAITS:
            if np.allclose(gait.value, known_gait.value, atol=0.01):
                return known_gait
        return gait

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
    robot_interface.target_speed = 0.26        # > 0.35 -> GaitPicker routes WALK -> AMBLE
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
        blended_gait, frequency, blending_factor = fuzzy.update(dt)
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

def test_fuzzy_controller_transition():
    """
    Full-system test using FuzzyController.

    This test intentionally exercises the REAL controller path:
        FuzzyController
            -> GaitPicker
            -> FuzzyGaitSwitch
            -> check_gait()
            -> current_gait update

    so we can observe whether:
        WALK -> BLENDED_GAIT -> AMBLE

    properly completes.
    """

    logger.debug("=" * 80)
    logger.debug("test_fuzzy_controller_transition")
    logger.debug("Simulate WALK -> AMBLE transition driven by FuzzyController.update()")
    logger.debug(f"Walk gait params: {Gait.WALK.value}")
    logger.debug(f"Amble gait params: {Gait.AMBLE.value}")
    logger.debug("=" * 80)

    starting_state = State(
        mode=Mode.MOVING,
        gait=Gait.WALK,
        frequency=GaitFreq.WALK.value
    )

    robot_interface = RobotInterface(starting_state)

    # Force WALK -> AMBLE transition
    robot_interface.target_speed = 0.26
    robot_interface.body_velocity = 0.0
    robot_interface.stability_metric = 0.85

    controller = FuzzyController(robot_interface)

    logger.debug(
        f"[setup] initial gait={robot_interface.current_gait}, "
        f"target_speed={robot_interface.target_speed}, "
        f"stability={robot_interface.stability_metric}"
    )

    dt = 0.05
    max_steps = 100

    for i in range(max_steps):

        t = (i + 1) * dt

        blending_factor = controller.update(dt)

        gait = robot_interface.active_gait

        logger.debug("-" * 80)
        logger.debug(
            f"[step {i+1:03d}] "
            f"t={t:.2f}s | "
            f"gait={gait.name} | "
            f"blending_factor={blending_factor:.4f} | "
            f"phases={tuple(round(x, 4) for x in gait.value)}"
        )

        # Detect successful snap to AMBLE
        if gait == Gait.AMBLE:
            logger.debug("")
            logger.debug("TRANSITION COMPLETE -> AMBLE")
            logger.debug(f"completed in {t:.2f} seconds")
            break

    else:
        logger.error("")
        logger.error("Transition never completed!")
        logger.error(
            f"final gait={robot_interface.current_gait.name}"
        )

    logger.debug("=" * 80)
    logger.debug("test_fuzzy_controller_transition DONE")
    logger.debug("=" * 80)

if __name__ == "__main__":
    test_fuzzy_controller_transition()