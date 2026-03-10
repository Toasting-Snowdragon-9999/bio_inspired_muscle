import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cpg.kuramoto_cpg import KuramotoCpg
from inverse_kinematics.inverse_kin import *
from shared_module.robot_state import Joint, RobotInterface, Foot
from shared_module.global_constants import NEURON_TO_FOOT_DICT
from cpg.trajectory_builder import TrajectoryBuilder

# Home joint configuration used to compute rest foot positions
# _HOME_Q = np.array([0.0, 0.9, -1.8])

home_position = { # IN JOINT ANGLES
    Foot.FL: np.array([ 1.56000000e-03, 8.51601985e-01, -1.80657050e+00]),
    Foot.FR: np.array([ 1.56000000e-03, 8.52830942e-01, -1.80920322e+00]),
    Foot.RL: np.array([ 1.56000000e-03, 1.02456780e+00, -1.75184535e+00]),
    Foot.RR: np.array([ 1.56000000e-03, 1.02612405e+00, -1.75448753e+00]),
}


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
        self.cpg = KuramotoCpg(robot_interface)

        self.traj_builder = TrajectoryBuilder(robot_interface, width=stride_length, height=step_height)

        # ===== IK =====
        self.solvers = {}
        self.prev_q = {}

        for foot in Foot:
            self.solvers[foot] = LevenbergMarquardtIK(foot)
            self.prev_q[foot] = home_position[foot]

        # ===== IK END =====

    def run(self) -> None:
        """
        Step CPG, solve IK for each leg, write joint targets to robot_interface.
        Called once per simulation timestep via mjcb_control.
        """
        self.cpg.run()
        phase_outputs = self.cpg.get_phase_outputs()
        foot_targets = self.traj_builder.build_trajectory(phase_outputs)

        # ===== IK =====
        joint_targets: dict[Joint, float] = {}
        for foot in Foot:
            goal_pos = foot_targets[foot]
            goal_pos = convert_frame(goal_pos, foot) # most important step

            q_result = self.solvers[foot].calculate(goal_pos, init_q=self.prev_q[foot])
            self.prev_q[foot] = q_result # update previous config

            for joint, angle in q_result.items():
                joint_targets[joint] = angle
            
        # ===== IK END =====

        self.robot_interface.target_positions = joint_targets

    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Proxy to CPG for oscillator graph overlay in MujocoSim."""
        return self.cpg.get_oscillator_outputs()

    def get_targets(self) -> dict[Joint, float]:
        """Return the current joint targets from robot_interface."""
        return self.robot_interface.target_positions
