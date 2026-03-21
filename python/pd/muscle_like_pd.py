import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.global_constants import FOOT_TO_JOINT_DICT, ACTUATOR_TORQUE_LIMIT
from shared_module.robot_state import RobotInterface, Foot, Joint
from pd.adaptive_imp import ada_imp_ctrl


"""
currently it is per leg controller 
could also crosscouple the legs for where 
an error in one joint can change the stiffness of another joint.

class MuscleLikePDController: 
    def __init__(self, robot_interface: RobotInterface) -> None: 
        self.robot_interface = robot_interface
        self.dof = 12  # 3 joints per leg * 4 legs
        self.impedance_controller = ada_imp_ctrl(self.dof)
"""

DOF_PER_LEG = 3  # hip (abduction), thigh, calf


class MuscleLikePD:
    """Minimal interface between RobotInterface and the adaptive impedance controller.

    For each leg, reads current joint state from robot_interface, feeds
    position/velocity errors into ada_imp_ctrl to obtain stiffness (K) and
    damping (B) matrices, then computes feedback torque:
        tau = K @ (q_d - q) + B @ (dq_d - dq)

    When use_ioac is False, ada_imp_ctrl returns fixed diagonal K/B matrices
    (the non-adaptive fallback gains defined in adaptive_imp.py).
    When use_ioac is True, K and B are adapted online per Xiong & Fang 2023.
    Both paths go through ada_imp_ctrl — a fixed PD is just K=diag(kp), B=diag(kd).
    """

    def __init__(self, robot_interface: RobotInterface, use_ioac: bool = True) -> None:
        self.robot_interface = robot_interface
        # One impedance controller per leg (3 DOF each: hip, thigh, calf)
        self.impedance_controller: dict[Foot, ada_imp_ctrl] = {
            foot: ada_imp_ctrl(DOF_PER_LEG, use_ioac=use_ioac)
            for foot in Foot
        }

    def control(
        self,
        q_d: dict[Joint, float],
        dq_d: dict[Joint, float],
    ) -> dict[Foot, np.ndarray]:
        """Compute per-leg torques using the adaptive impedance controller.

        Parameters
        ----------
        q_d  : target joint positions  (from IK solver)
        dq_d : target joint velocities (from Jacobian)

        Returns
        -------
        dict mapping each Foot to a (3,) torque array [hip, thigh, calf].
        """
        # Current joint state from sensors (via robot_interface)
        q = self.robot_interface.joint_positions
        dq = self.robot_interface.joint_velocities

        outputs: dict[Foot, np.ndarray] = {}

        for foot, joints in FOOT_TO_JOINT_DICT.items():
            # Build per-leg vectors (3,)
            q_vec   = np.array([q[j]    for j in joints])
            qd_vec  = np.array([q_d[j]  for j in joints])
            dq_vec  = np.array([dq[j]   for j in joints])
            dqd_vec = np.array([dq_d[j] for j in joints])

            # Position and velocity errors
            e  = qd_vec - q_vec
            de = dqd_vec - dq_vec

            # Get stiffness (K) and damping (B) from impedance controller.
            # When use_ioac=False this returns fixed diagonal matrices (standard PD).
            # When use_ioac=True K and B are adapted online (Xiong & Fang 2023).
            K, B = self.impedance_controller[foot].update_impedance(
                q_vec, qd_vec, dq_vec, dqd_vec
            )

            # Feedback torque: tau = K @ e + B @ de
            # @ is matrix multiplication, works with vectors as well
            tau = (K @ e) + (B @ de)
            tau = np.asarray(tau, dtype=float).reshape(-1)

            # Clip to actuator torque limits
            tau = np.clip(tau, -ACTUATOR_TORQUE_LIMIT, ACTUATOR_TORQUE_LIMIT)

            outputs[foot] = tau

        return outputs
