import os
import sys
import numpy as np
from mujoco.glfw import glfw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cpg.kuramoto_cpg import KuramotoCpg
from inverse_kinematics.inverse_kin import *
from shared_module.robot_state import Joint, RobotInterface, Foot
from shared_module.global_constants import FOOT_TO_JOINT_DICT
from cpg.trajectory_builder import TrajectoryBuilder, Coordinate
from pd.muscle_like_pd import MuscleLikePD

# Maximum joint velocity (rad/s) to prevent aggressive torques from large
# Jacobian pseudoinverse outputs during fast CPG phase transitions.
MAX_JOINT_VEL: float = 5.0

# Home joint configuration used to compute rest foot positions
# _HOME_Q = np.array([0.0, 0.9, -1.8])

home_position = { # IN JOINT ANGLES
    Foot.FL: np.array([ 1.56000000e-03, 8.51601985e-01, -1.80657050e+00]),
    Foot.FR: np.array([ 1.56000000e-03, 8.52830942e-01, -1.80920322e+00]),
    Foot.RL: np.array([ 1.56000000e-03, 1.02456780e+00, -1.75184535e+00]),
    Foot.RR: np.array([ 1.56000000e-03, 1.02612405e+00, -1.75448753e+00]),
}

def pretty_print_foot_positions(title: str, foot_positions: dict[Foot, Coordinate]):
    print(f"\n{title}")
    print("-" * 40)

    for foot in Foot:
        pos = foot_positions[foot]
        print(
            f"{foot.name:>3}  "
            f" [{float(pos.x): .4f},  "
            f"{float(pos.y): .4f},  "
            f"{float(pos.z): .4f}]"
        )

    print("-" * 40)

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
        use_adaptive_pd: bool = True
    ) -> None:
        self.robot_interface = robot_interface

        # CPG owns the foot trajectory generation
        self.cpg = KuramotoCpg(robot_interface)

        self.traj_builder = TrajectoryBuilder(robot_interface, width=stride_length, height=step_height)

        self.pd = MuscleLikePD(robot_interface, use_ioac=use_adaptive_pd)
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
        # ===== CPG & trajectory =====
        self.cpg.run()
        phase_outputs = self.cpg.get_phase_outputs()
        phase_velocities = self.cpg.get_phase_velocities()
        foot_targets, foot_velocities = self.traj_builder.build_trajectory(phase_outputs, phase_velocities)
        # Build proper vel dict from dict[Foot, Coordinate] to dict[Foot, np.ndarray]

        pos_targets = {
            foot: np.array([pos.x, pos.y, pos.z]) 
                            for foot, pos in foot_targets.items()
        }
        # ===== CPG & trajectory END =====

        # ===== IK =====
        joint_targets: dict[Joint, float] = {}
        joint_vel_targets: dict[Joint, float] = {}

        for foot in Foot:
            goal_pos = pos_targets[foot]
            goal_pos = convert_frame(goal_pos, foot) # most important step

            q_result = self.solvers[foot].calculate(goal_pos, init_q=self.prev_q[foot])
            joints = FOOT_TO_JOINT_DICT[foot]
            q = []

            for joint, angle in q_result.items():
                joint_targets[joint] = angle
                q.append(angle)
            
            self.prev_q[foot] = np.array(q)

            # Compute vel targets from Jacobian
            foot_vel = foot_velocities[foot]
            v = np.array([foot_vel.x, foot_vel.y, foot_vel.z])

            d_y = LEG_CONFIG[foot]["d_y"]

            J = leg_jacobian(q, d_y)

            dq = np.asarray(np.linalg.pinv(J) @ v).flatten()

            # Clamp joint velocities to prevent aggressive torques during fast phase transitions
            dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL)

            # Abduction (HIP) is not controlled by our 2-DOF sagittal IK,
            # so zero out its position and velocity targets to prevent drift.
            hip_joint = joints[0]
            joint_targets[hip_joint] = 0.0
            dq[0] = 0.0

            for j, vel in zip(joints, dq):
                joint_vel_targets[j] = float(vel)
        # ===== IK END =====
        
        self.robot_interface.target_positions = joint_targets
        self.robot_interface.target_velocities = joint_vel_targets

        # ===== PD Control =====
        output_torque = self.pd.control(joint_targets, joint_vel_targets)
        for foot, torques in output_torque.items():
            for i, joint in enumerate(FOOT_TO_JOINT_DICT[foot]):
                tau = np.asarray(torques).flatten()   # force shape (3,)
                self.robot_interface.target_torques[joint] = tau[i]  # i = 1,2,3 for hip, thigh, knee respectively
        # ===== PD Control END =====

    def keyboard_callback(self, window, key, scancode, act, mods):
        if act != glfw.PRESS:
            return

    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Proxy to CPG for oscillator graph overlay in MujocoSim."""
        return self.cpg.get_oscillator_outputs()

    def get_targets(self) -> dict[Joint, float]:
        """Return the current joint targets from robot_interface."""
        return self.robot_interface.target_positions
