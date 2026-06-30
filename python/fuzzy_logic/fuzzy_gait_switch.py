"""@brief Fuzzy-logic gait switcher for the quadruped locomotion controller.

This module implements the active gait-switching feature. A discrete
``GaitPicker`` chooses a target gait from a commanded velocity (with
hysteresis bands), while ``FuzzyGaitSwitch`` uses a scikit-fuzzy control
system over ``stability`` and ``speed_error`` to produce a ``blend_rate``
that is integrated into a persistent blending factor. The blending factor
smoothly interpolates the CPG phase offsets and trajectory parameters from
the current gait to the next, and ``FuzzyController`` ties the picker and
fuzzy switcher together into the per-tick update used by the simulator.
"""

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
    """@brief Coarse velocity category used to classify a commanded speed."""
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"

class VelCmd:
    """@brief A validated velocity command with its discrete velocity class."""
    def __init__(self, velocity: float):
        """@brief Construct a velocity command, clamping it to the valid range.

        @param velocity: Commanded velocity; clamped to [MIN_VELOCITY, MAX_VELOCITY].
        """
        if velocity < MIN_VELOCITY or velocity > MAX_VELOCITY:
            logger.error("Velocity must be between 0 and 1, capping to valid range")
            velocity = max(MIN_VELOCITY, min(MAX_VELOCITY, velocity))
        self._v = velocity
        self._type = self._classify_velocity()

    def _classify_velocity(self) -> VelType:
        """@brief Map the stored velocity into a coarse VelType band.

        @return The VelType (SLOW, MEDIUM, or FAST) for the current velocity.
        """
        if self._v < 0.3:
            return VelType.SLOW
        elif self._v < 0.7:
            return VelType.MEDIUM
        else:
            return VelType.FAST

    @property
    def v(self) -> float:
        """@brief The stored (clamped) velocity value.

        @return The current velocity as a float.
        """
        return self._v

    @property
    def type(self) -> VelType:
        """@brief The velocity class for the current velocity.

        @return The cached VelType band.
        """
        return self._type

    @v.setter
    def v(self, value: float):
        """@brief Set the velocity (clamping to range) and reclassify its band.

        @param value: New velocity; clamped to [MIN_VELOCITY, MAX_VELOCITY].
        """
        if value < MIN_VELOCITY or value > MAX_VELOCITY:
            logger.error("Velocity must be between 0 and 1, capping to valid range")
            value = max(MIN_VELOCITY, min(MAX_VELOCITY, value))
        self._v = value 
        self._type = self._classify_velocity()


class GaitPicker:
    """@brief Discrete gait selector mapping velocity to gait with hysteresis."""

    @staticmethod
    def pick_freq(current_gait: Gait):
        """@brief Look up the CPG frequency associated with a gait.

        @param current_gait: The gait whose nominal frequency is requested.
        @return The GaitFreq value for the gait, or None if unrecognized.
        """
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
        """@brief Choose the next gait from the commanded velocity, with hysteresis.

        Each gait has asymmetric up/down velocity thresholds so the gait only
        changes one step at a time and resists rapid back-and-forth switching.

        @param vel: The commanded velocity.
        @param current_gait: The gait currently active.
        @return The selected gait (may equal current_gait if no switch is warranted).
        """
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
        """@brief Build a new robot State (gait, trajectory, frequency) for a velocity.

        @param vel: The commanded velocity.
        @param current_state: The current robot state to transition from.
        @return A new State preserving the mode but with the picked gait,
            trajectory parameters, and frequency.
        """
        new_gait = self.pick_gait(vel, current_state.gait)
        new_traj_params = self.pick_trajectory(current_state.gait)
        new_frequency = self.pick_freq(current_state.gait)
        return State(mode=current_state.mode, gait=new_gait, traj_params=new_traj_params, frequency=new_frequency)

class FuzzyGaitSwitch:
    """@brief Fuzzy controller that smoothly blends one gait into another.

    Holds a scikit-fuzzy control system whose antecedents are ``stability``
    and ``speed_error`` and whose consequent is a ``blend_rate``. The
    blend_rate is integrated over time into a persistent blending factor that
    drives interpolation of CPG phase offsets, frequency, and trajectory
    parameters during a gait transition.
    """
    def __init__(self, robot_interface: RobotInterface):
        """@brief Build the fuzzy membership functions, rules, and control system.

        @param robot_interface: Interface providing current gait, target gait,
            speed/stability inputs, and mode control.
        """
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
        """@brief Display the stability, speed_error, and blend_rate membership functions."""
        # Lazy import: matplotlib.pyplot is heavy and binds a backend on first
        # import, so keep it out of module scope.
        import matplotlib.pyplot as plt
        self.stability.view()
        self.speed_error.view()
        self.blend_rate.view()
        plt.show()

    def update_blending_factor(self, blend_rate_output: float, dt: float) -> float:
        """@brief Integrate the fuzzy blend_rate into the persistent blending_factor.

        Integrate the fuzzy blend_rate into the persistent blending_factor.

        @param blend_rate_output: Fuzzy blend_rate output to integrate.
        @param dt: Elapsed time since the previous call (decimated dt if decimated).
        @return The updated blending factor, clipped to [0.0, 1.0].

        `dt` is the wall time elapsed since the previous call — when this
        method is driven from a decimated tick (e.g. every 30 sim steps),
        pass `decimated_steps * sim_dt`, NOT `sim_dt`, otherwise the
        integration runs ~30x slower than intended.
        """
        prev = self.blending_factor
        self.blending_factor = float(np.clip(self.blending_factor + blend_rate_output * dt, 0.0, 1.0))
        return self.blending_factor

    def interpolate(self, a: float, b: float, s: float) -> float:
        """@brief Linear interpolation between a and b with blending factor s.

        Linear interpolation between a and b with blending factor s.

        @param a: Start value (returned when s == 0).
        @param b: End value (returned when s == 1).
        @param s: Blending factor in [0, 1].
        @return The interpolated value (1 - s) * a + s * b.
        """
        return (1 - s) * a + s * b

    def blend_gaits(self, old_gait: Gait, new_gait: Gait, blending_factor: float) -> Gait:
        """@brief Interpolate each phase-offset component between two gaits.

        @param old_gait: Gait being transitioned from.
        @param new_gait: Gait being transitioned to.
        @param blending_factor: Interpolation factor in [0, 1].
        @return A synthetic "BLENDED_GAIT" whose phase tuple is the
            component-wise interpolation of old and new.
        """
        trans_gait_values = []
        for a, b in zip(old_gait.value, new_gait.value):
            trans_gait_values.append(self.interpolate(a, b, blending_factor))
        blended = Gait("BLENDED_GAIT", tuple(trans_gait_values))
        
        return blended
    
    def interp_angle(self, a: float, b: float, s: float) -> float:
        """@brief Interpolate between two angles along the shortest path.

        Wraps the difference into [-pi, pi) so the interpolation takes the
        shorter way around the circle.

        @param a: Start angle in radians.
        @param b: End angle in radians.
        @param s: Blending factor in [0, 1].
        @return The interpolated angle in radians.
        """

        delta = (b - a + np.pi) % (2*np.pi) - np.pi

        return a + s * delta

    def blend_trajectories(
        self,
        old_traj_params: EllipsoidConfig,
        new_traj_params: EllipsoidConfig,
        blending_factor: float
    ) -> EllipsoidConfig:
        """@brief Interpolate every trajectory parameter between two ellipsoid configs.

        @param old_traj_params: Trajectory configuration being transitioned from.
        @param new_traj_params: Trajectory configuration being transitioned to.
        @param blending_factor: Interpolation factor in [0, 1].
        @return A new EllipsoidConfig with each parameter angle-interpolated.
        """

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
        """@brief Advance the gait blend by one tick and return the blended state.

        Evaluates the fuzzy control system for the current speed_error and
        stability, integrates the blend_rate, and interpolates the gait phase
        tuple and frequency toward the target gait. When there is no pending
        transition, returns the current gait and frequency unchanged.

        @param dt: Elapsed time since the last call (accumulated/decimated dt).
        @return A (gait_params, frequency, blending_factor) tuple; the
            blending_factor is None when no transition is in progress.
        """
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
    """@brief Top-level controller wiring the gait picker and fuzzy blender together."""
    def __init__(self, robot_interface: RobotInterface):
        """@brief Construct the controller and its underlying FuzzyGaitSwitch.

        @param robot_interface: Interface exposing the robot state and commands.
        """
        self.robot_interface = robot_interface
        self.fuzzy_gait_switch = FuzzyGaitSwitch(robot_interface)

    def update(self, dt: float) -> float:
        """@brief Run one control tick: pick a target gait and advance the blend.

        Picks a target gait from the commanded speed, drives the fuzzy gait
        switch, and finalizes the transition (snapping to the target gait)
        once the blending factor saturates. Disables the CPG when the
        commanded speed is effectively zero.

        @param dt: Elapsed time since the last call.
        @return The current blending factor (0.0 when idle/stopped,
            1.0 when a transition just completed).
        """

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
        """@brief Blend the active trajectory parameters toward the next gait's.

        @param dt: Elapsed time since the last call (unused but kept for symmetry).
        @param blending_factor: Interpolation factor in [0, 1].
        """
        self.robot_interface.active_traj_params =  self.fuzzy_gait_switch.blend_trajectories(self.robot_interface.current_traj_params, self.robot_interface.next_traj_params, blending_factor)

    def check_gait(self, gait: Gait) -> Gait:
        """@brief Snap a blended gait back to a known gait when close enough.

        @param gait: A (possibly blended) gait to test against known gaits.
        @return The matching known gait if the phase tuple is within
            tolerance, otherwise the input gait unchanged.
        """
        for known_gait in ALL_GAITS:
            if np.allclose(gait.value, known_gait.value, atol=0.01):
                return known_gait
        return gait

def test_fuzzy_gait_switch():
    """@brief Observe the WALK -> AMBLE fuzzy gait transition over 1 second.

    Observe the WALK -> AMBLE fuzzy gait transition over 1 second.

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
    """@brief Full-system test of WALK -> AMBLE driven by FuzzyController.update().

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