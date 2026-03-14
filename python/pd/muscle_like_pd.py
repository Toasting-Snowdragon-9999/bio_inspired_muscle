import os, sys
import numpy as np



sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.global_constants import *
from shared_module.robot_state import RobotInterface, Foot, Joint
from pd.adaptive_impedance import ada_imp_ctrl


"""
currently it is pr leg controller 
could also crossccouple the legs for where 
an error in one joint can change the stiffness of another joint.

class MuscleLikePDController: 
    def __init__(self, robot_interface: RobotInterface) -> None: 
        self.robot_interface = robot_interface
        self.dof = 12  # 3 joints per leg * 4 legs
        self.impedance_controller = ada_imp_ctrl(self.dof)
"""

class MuscleLikePD:
    def __init__(
        self,
        robot_interface: RobotInterface,
        kp_fixed: float = 90.0,
        kd_fixed: float = 15.0,
    ) -> None:
        self.robot_interface = robot_interface
        self.dof = 3  # 3 joints per leg * 4 legs
        self.kp_fixed = float(kp_fixed)
        self.kd_fixed = float(kd_fixed)
        self.freeze_adaptation = False
        self.impedance_controller = {
            foot: ada_imp_ctrl(
                self.dof,
                kp0=np.array([20.0, 28.0, 35.0]),
                kd0=np.array([1.5, 2.2, 3.0]),
                kp_bounds=(5.0, 120.0),
                kd_bounds=(0.5, 25.0),
                adapt_rate=0.02,
                tau_limit=np.array([23.7, 23.7, 45.43]),
                use_full_matrix=False,
            )
            for foot in Foot
        }
        self._sat_events = {foot: np.zeros(self.dof, dtype=np.int64) for foot in Foot}
        self._step_count = 0

    def set_freeze_adaptation(self, enabled: bool) -> None:
        self.freeze_adaptation = bool(enabled)

    def toggle_freeze_adaptation(self) -> bool:
        self.freeze_adaptation = not self.freeze_adaptation
        return self.freeze_adaptation

    def control(self, q_d: dict[Joint, float], dq_d: dict[Joint, float]) -> dict[Foot, np.ndarray]:
        q    = self.robot_interface.joint_positions
        # q_d  = self.robot_interface.target_positions
        dq   = self.robot_interface.joint_velocities
        # dq_d = self.robot_interface.target_velocities

        outputs = {}

        for foot, joints in FOOT_TO_JOINT_DICT.items():

            q_vec   = np.array([q[j]   for j in joints])
            qd_vec  = np.array([q_d[j] for j in joints])
            dq_vec  = np.array([dq[j]  for j in joints])
            dqd_vec = np.array([dq_d[j]for j in joints])

            e = qd_vec - q_vec
            de = dqd_vec - dq_vec

            if self.freeze_adaptation:
                # True ordinary fixed PD fallback.
                tau = self.kp_fixed * e + self.kd_fixed * de
            else:
                # Paper-style adaptive feedback: tau_ff=0, tau=tau_fb.
                K, B = self.impedance_controller[foot].update_impedance(
                    q_vec, qd_vec, dq_vec, dqd_vec, freeze_adaptation=False
                )
                tau = (K @ e) + (B @ de)

            tau = np.asarray(tau, dtype=float).reshape(-1)

            limits = self.impedance_controller[foot].tau_limit
            tau = np.clip(tau, -limits, limits)

            sat_mask = np.isclose(np.abs(tau), limits, rtol=0.0, atol=1e-6)
            self._sat_events[foot] += sat_mask.astype(np.int64)
            outputs[foot] = tau

        self._step_count += 1
        return outputs

    def get_saturation_stats(self) -> dict[Foot, np.ndarray]:
        if self._step_count <= 0:
            return {foot: np.zeros(self.dof) for foot in Foot}
        return {
            foot: self._sat_events[foot] / float(self._step_count)
            for foot in Foot
        }
