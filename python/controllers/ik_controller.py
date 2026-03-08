"""
IKController — foot-trajectory controller using CPG + inverse kinematics.

Architecture:
    CPG (phase oscillators)  →  Cartesian foot positions per leg
                             →  LM-IK  →  joint targets
                             →  RobotInterface.target_positions

MujocoSim reads target_positions from RobotInterface and writes data.ctrl.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inverse_kinematics'))

from cpg.kuramoto_cpg import KuramotoCpg
from inverse_kin import LevenbergMarquardtIK, Leg, forward_kinematics, LEG_CONFIG
from shared_module.robot_state import Joint, RobotInterface


# Oscillator index → Leg enum (same order as Gait phase tuples: FL, FR, RR, RL)
_OSC_TO_LEG = {
    0: Leg.FL,
    1: Leg.FR,
    2: Leg.RR,
    3: Leg.RL,
}

# Home joint configuration used to compute rest foot positions
_HOME_Q = np.array([0.0, 0.9, -1.8])


class IKController:
    """
    Foot-trajectory controller combining Kuramoto CPG and Levenberg-Marquardt IK.

    The CPG produces Cartesian foot positions; IK converts them to joint angles.
    stride_length and step_height are passed to the CPG and control trajectory size.
    Results are written to robot_interface.target_positions every timestep.
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        stride_length: float = 0.06,
        step_height: float = 0.04,
        warmup_seconds: float = 2.0,
    ) -> None:
        self.robot_interface = robot_interface

        # CPG owns the foot trajectory generation
        self.cpg = KuramotoCpg(
            robot_interface,
            stride_length=stride_length,
            step_height=step_height,
            warmup_seconds=warmup_seconds,
        )

        # Per-leg IK solvers and warm-start caches
        self._ik_solvers: dict[Leg, LevenbergMarquardtIK] = {}
        self._prev_q: dict[Leg, np.ndarray] = {}

        for osc_idx, leg in _OSC_TO_LEG.items():
            cfg = LEG_CONFIG[leg]
            self._ik_solvers[leg] = LevenbergMarquardtIK(leg=leg)
            self._prev_q[leg] = _HOME_Q.copy()
            # Register home foot position with the CPG
            home_foot = forward_kinematics(_HOME_Q, cfg['d_y'])
            self.cpg.set_foot_home(osc_idx, home_foot)

    # ── Public interface (called by MujocoSim callback) ──────────

    def run(self) -> None:
        """
        Step CPG, solve IK for each leg, write joint targets to robot_interface.
        Called once per simulation timestep via mjcb_control.
        """
        self.cpg.run()

        foot_targets = self.cpg.get_targets()
        targets: dict[Joint, float] = {}

        for osc_idx, foot_pos in foot_targets.items():
            leg = _OSC_TO_LEG[osc_idx]

            try:
                joint_angles = self._ik_solvers[leg].calculate(
                    goal=foot_pos,
                    init_q=self._prev_q[leg],
                )
                self._prev_q[leg] = np.array(list(joint_angles.values()))
            except RuntimeError:
                # IK diverged — hold previous solution
                joint_angles = {
                    j: float(q)
                    for j, q in zip(LEG_CONFIG[leg]['joints'], self._prev_q[leg])
                }

            targets.update(joint_angles)

        self.robot_interface.target_positions = targets

    # ── Delegate helpers ─────────────────────────────────────────

    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Proxy to CPG for oscillator graph overlay in MujocoSim."""
        return self.cpg.get_oscillator_outputs()

    def get_targets(self) -> dict[Joint, float]:
        """Return the current joint targets from robot_interface."""
        return self.robot_interface.target_positions
