import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cpg.kuramoto_cpg import KuramotoCpg
from inverse_kinematics.inverse_kin import LevenbergMarquardtIK, forward_kinematics, LEG_CONFIG
from shared_module.robot_state import Joint, RobotInterface, Foot
from shared_module.global_constants import NEURON_TO_FOOT_DICT
from cpg.trajectory_builder import TrajectoryBuilder

# Home joint configuration used to compute rest foot positions
_HOME_Q = np.array([0.0, 0.9, -1.8])


class IKController:
    """
    Foot-trajectory controller combining Kuramoto CPG and Levenberg-Marquardt IK.
    The CPG produces Cartesian foot positions; IK converts them to joint angles.
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

        self.traj_builder = TrajectoryBuilder(robot_interface, width=stride_length, height=step_height, z_value_ground_offset=0.0)

        # Per-leg IK solvers and warm-start caches
        self._ik_solvers: dict[Foot, LevenbergMarquardtIK] = {}
        self._prev_q: dict[Foot, np.ndarray] = {}

        for osc_idx, foot in NEURON_TO_FOOT_DICT.items():
            cfg = LEG_CONFIG[foot]
            self._ik_solvers[foot] = LevenbergMarquardtIK(leg=foot)
            self._prev_q[foot] = _HOME_Q.copy()
            # Register home foot position with the CPG
            self.home_foot = forward_kinematics(_HOME_Q, cfg['d_y'])

    def run(self) -> None:
        """
        Step CPG, solve IK for each leg, write joint targets to robot_interface.
        Called once per simulation timestep via mjcb_control.
        """
        self.cpg.run()
        phase_outputs = self.cpg.get_phase_outputs()
        foot_targets = self.traj_builder.build_trajectory(phase_outputs)
        targets: dict[Joint, float] = {}

        for foot, foot_pos in foot_targets.items():

            try:
                joint_angles = self._ik_solvers[foot].calculate(
                    goal=foot_pos,
                    init_q=self._prev_q[foot],
                )
                self._prev_q[foot] = np.array(list(joint_angles.values()))

            except RuntimeError:
                # IK diverged — hold previous solution
                joint_angles = {
                    j: float(q)
                    for j, q in zip(LEG_CONFIG[foot]['joints'], self._prev_q[foot])
                }

            targets.update(joint_angles)

        self.robot_interface.target_positions = targets


    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Proxy to CPG for oscillator graph overlay in MujocoSim."""
        return self.cpg.get_phase_outputs()

    def get_targets(self) -> dict[Joint, float]:
        """Return the current joint targets from robot_interface."""
        return self.robot_interface.target_positions
