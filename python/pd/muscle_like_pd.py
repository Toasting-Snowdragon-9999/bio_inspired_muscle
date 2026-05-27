import os
import sys
import numpy as np
import copy
from collections import deque
from scipy.optimize import minimize

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
        self.initial_params = params
        # One adaptive impedance controller per leg
        self.impedance_controller: dict[Foot, ada_imp_ctrl] = {
            foot: ada_imp_ctrl(
                DOF_PER_LEG,
                params=params,
                use_oiac=use_oiac
            )
            for foot in Foot
        }
        self.prev_tau = {foot: None for foot in Foot}
        self.state_buffer = {
            foot: deque(maxlen=3000)
            for foot in Foot
        }
        self.iteration = 0

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
    
            tau = (K @ e) + (B @ de)
            tau = tau.flatten()
            tau = np.clip(
                tau,
                -ACTUATOR_TORQUE_LIMIT,
                ACTUATOR_TORQUE_LIMIT
            )
            # self.state_buffer[foot].append({
            #     "q": q_vec.copy(),
            #     "dq": dq_vec.copy(),
            #     "q_d": qd_vec.copy(),
            #     "dq_d": dqd_vec.copy()
            # })
            outputs[foot] = tau

        # if (self.robot_interface.new_gait_cycle):
            
        #     x = self.optimize_params()
        #     print(f"Optimized params for {foot.name}: a={x[0]:.4f}, b={x[1]:.4f}, k={x[2]:.4f} function called {self.iteration} times with size of state buffer {len(self.state_buffer[foot])}")
        #     self.update_params(x)
        #     self.iteration = 0

        return outputs
    
    def update_params(self, new_params: tuple[float, float, float]):
        for foot in Foot:
            self.impedance_controller[foot].a, self.impedance_controller[foot].b, self.impedance_controller[foot].k = new_params

    def cost_function(self, params):

        a, b, k = params

        w = [1000.0, 100.0, 1e-4, 1e-5, 100.0]

        total_cost = 0.0
        

        for foot in Foot:

            prev_states = self.state_buffer[foot]

            ctrl = copy.deepcopy(self.impedance_controller[foot])

            ctrl.a = a
            ctrl.b = b
            ctrl.k = k

            prev_tau = None
            prev_dq = None

            for state in prev_states:

                q = state["q"]
                dq = state["dq"]
                q_d = state["q_d"]
                dq_d = state["dq_d"]

                K, B = ctrl.update_impedance(
                    q, q_d,
                    dq, dq_d
                )

                e = ctrl.gen_pos_err
                de = ctrl.gen_vel_err

                tau = (K @ e) + (B @ de)

                power_cost = np.sum(
                    np.abs(
                        tau.flatten() * dq.flatten()
                    )
                )

                acc_cost = 0.0
                qdd= 0.0
                if prev_dq is not None:
                    qdd = (dq - prev_dq) / self.robot_interface.dt
                    acc_cost = np.sum(qdd**2)

                smooth_cost = 0.0

                if prev_tau is not None:
                    smooth_cost = np.sum(
                        (tau - prev_tau)**2
                    )

                pos_cost = np.sum(e**2)
                vel_cost = np.sum(de**2)
                acc_cost = np.sum(qdd**2)

                total_cost += (
                    w[0] * pos_cost
                    + w[1] * vel_cost
                    + w[2] * power_cost
                    + w[3] * smooth_cost
                    + w[4] * acc_cost
                )

                prev_tau = tau.copy()
                prev_dq = dq.copy()

                self.iteration += 1

        total_samples = sum(
            len(self.state_buffer[foot])
            for foot in Foot
        )

        return total_cost / max(total_samples, 1)

    def optimize_params(self):

        bounds = [
            (0.01, 100.0),
            (0.1, 200.0),
            (0.01, 500.0)
        ]

        options={
            "maxiter": 20000,
            "ftol": 1e-4
        }

        result = minimize(
            self.cost_function,
            self.initial_params,
            # args=(foot_id,),
            method="SLSQP",
            bounds=bounds, 
            options=options
        )

        return result.x