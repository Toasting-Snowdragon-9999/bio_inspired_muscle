import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import (
    FOOT_TO_JOINT_DICT,
    ACTUATOR_TORQUE_LIMIT
)

from shared_module.robot_state import (
    RobotInterface,
    Foot,
    Joint
)

from pd.adaptive_imp import ada_imp_ctrl


DOF_PER_LEG = 3


class MuscleLikePD:
    """
    Per-leg adaptive impedance controller for Unitree Go2.

    Each leg has:
        [hip, thigh, calf]

    OAIC adapts:
        - stiffness matrix K
        - damping matrix B

    Torque:
        tau = K(q_d - q) + B(dq_d - dq)
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        params: tuple[float, float, float] = (0.2, 5.0, 0.05),
        use_oiac: bool = True
    ) -> None:

        self.robot_interface = robot_interface

        # One adaptive impedance controller per leg
        self.impedance_controller: dict[Foot, ada_imp_ctrl] = {
            foot: ada_imp_ctrl(
                DOF_PER_LEG,
                params=params,
                use_oiac=use_oiac
            )
            for foot in Foot
        }

    def control(
        self,
        q_d: dict[Joint, float],
        dq_d: dict[Joint, float],
    ) -> dict[Foot, np.ndarray]:

        q = self.robot_interface.joint_positions
        dq = self.robot_interface.joint_velocities

        outputs: dict[Foot, np.ndarray] = {}

        for foot, joints in FOOT_TO_JOINT_DICT.items():
            q_vec = np.array(
                [[q[j]] for j in joints],
                dtype=float
            )

            qd_vec = np.array(
                [[q_d[j]] for j in joints],
                dtype=float
            )

            dq_vec = np.array(
                [[dq[j]] for j in joints],
                dtype=float
            )

            dqd_vec = np.array(
                [[dq_d[j]] for j in joints],
                dtype=float
            )

            K, B = self.impedance_controller[foot].update_impedance(
                q=q_vec,
                q_d=qd_vec,
                dq=dq_vec,
                dq_d=dqd_vec
            )

            e = self.impedance_controller[foot].gen_pos_err
            de = self.impedance_controller[foot].gen_vel_err
            # e = qd_vec - q_vec        # Correct old 
            # de = dqd_vec - dq_vec     # Correct old 
            # e = q_vec - qd_vec      # Crazy bot
            # de = dq_vec - dqd_vec   # no 
    
            tau = (K @ e) + (B @ de)
            tau = tau.flatten()
            tau = np.clip(
                tau,
                -ACTUATOR_TORQUE_LIMIT,
                ACTUATOR_TORQUE_LIMIT
            )

            outputs[foot] = tau

        return outputs