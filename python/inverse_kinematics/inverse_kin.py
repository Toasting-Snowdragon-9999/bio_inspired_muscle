# import os, sys
# import numpy as np
# from enum import Enum, auto

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from shared_module.robot_state import Joint
# from shared_module.global_constants import (
#     ABDUCTION_POS_RANGE, FRONT_HIP_POS_RANGE, BACK_HIP_POS_RANGE, KNEE_POS_RANGE
# )


# class Leg(Enum):
#     """Identifies which leg the IK solver operates on."""
#     FL = auto()
#     FR = auto()
#     RR = auto()
#     RL = auto()


# # ── Go2 leg geometry (metres) ──────────────────────────────────
# L_THIGH = 0.213   # thigh → calf link length (Z)
# L_CALF  = 0.213   # calf  → foot link length (Z)

# # Per-leg configuration: lateral hip-to-thigh offset, Joint enum mapping, hip range
# LEG_CONFIG = {
#     Leg.FL: {'d_y':  0.141490825432425, 'joints': (Joint.FL_HIP, Joint.FL_THIGH, Joint.FL_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
#     Leg.FR: {'d_y': -0.141490825432425, 'joints': (Joint.FR_HIP, Joint.FR_THIGH, Joint.FR_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
#     Leg.RL: {'d_y':  0.141490825432425, 'joints': (Joint.RL_HIP, Joint.RL_THIGH, Joint.RL_CALF), 'hip_range': BACK_HIP_POS_RANGE},
#     Leg.RR: {'d_y': -0.141490825432425, 'joints': (Joint.RR_HIP, Joint.RR_THIGH, Joint.RR_CALF), 'hip_range': BACK_HIP_POS_RANGE},
# }


# # ── Analytical forward kinematics & Jacobian for Go2 3-DOF leg ─

# def forward_kinematics(q: np.ndarray, d_y: float, hip_sign: float = 1.0) -> np.ndarray:
#     """
#     Compute foot position [x, y, z] relative to hip origin for a Go2 leg.

#     Kinematic chain:
#         abduction(X-axis) → lateral offset d_y → hip_pitch(Y-axis) → L_thigh
#                           → knee_pitch(Y-axis) → L_calf → foot

#     Args:
#         q:   joint angles [q_abduction, q_hip, q_knee]
#         d_y: signed lateral hip-to-thigh offset (+left, -right)
#     """
#     q1, q2, q3 = q
#     s_q1,  c_q1    = np.sin(q1),      np.cos(q1)
#     # s_q2,  c_q2    = np.sin(q2),      np.cos(q2)
#     s_q2,  c_q2 = np.sin(hip_sign * q2), np.cos(hip_sign * q2)
#     # s_q23, c_q23   = np.sin(q2 + q3), np.cos(q2 + q3)
#     s_q23, c_q23 = np.sin(hip_sign * q2 + q3), np.cos(hip_sign * q2 + q3)

#     x = -L_THIGH * s_q2 - L_CALF * s_q23
#     y =  d_y * c_q1 + (L_THIGH * c_q2 + L_CALF * c_q23) * s_q1
#     z =  d_y * s_q1 - (L_THIGH * c_q2 + L_CALF * c_q23) * c_q1
#     return np.array([x, y, z])

# def convert_frame(pos: np.ndarray) -> np.ndarray:
#     """
#     Convert MuJoCo foot position frame to IK model frame.
#     MuJoCo appears to use +Z downward while the IK model uses -Z downward.
#     """
#     p = pos.copy()
#     p[2] *= -1
#     return p

# def leg_jacobian(q: np.ndarray, d_y: float, hip_sign: float = 1.0) -> np.ndarray:
#     """
#     Compute 3x3 analytical positional Jacobian  J = ∂p/∂q  for a Go2 leg.

#     Args:
#         q:   joint angles [q_abduction, q_hip, q_knee]
#         d_y: signed lateral hip-to-thigh offset
#     """
#     q1, q2, q3 = q
#     s_q1,  c_q1    = np.sin(q1),      np.cos(q1)
#     # s_q2,  c_q2    = np.sin(q2),      np.cos(q2)
#     # s_q23, c_q23   = np.sin(q2 + q3), np.cos(q2 + q3)
#     s_q2,  c_q2  = np.sin(hip_sign * q2), np.cos(hip_sign * q2)
#     s_q23, c_q23 = np.sin(hip_sign * q2 + q3), np.cos(hip_sign * q2 + q3)

#     Lc = L_THIGH * c_q2 + L_CALF * c_q23   # sum of cosine projections / horizontal projection
#     Ls = L_THIGH * s_q2 + L_CALF * s_q23   # sum of sine projections   / vertical   projection

#     # return np.array([ # how each joint angle changes the cartesian foot position. maps: dp = J(q)dq
#     #     [0.0,                       -L_THIGH * c_q2 - L_CALF * c_q23, -L_CALF * c_q23        ], # how x is affected
#     #     [-d_y * s_q1 + Lc * c_q1,   -Ls * s_q1,                       -L_CALF * s_q23 * s_q1 ], # how y is affected / lateral  movement
#     #     [ d_y * c_q1 + Lc * s_q1,    Ls * c_q1,                        L_CALF * s_q23 * c_q1 ], # how z is affected / vertical movement
#     #     # abduction joint            # hip joint                       # knee joint
#     # ])
#     # return np.array([
#     #     [0.0,                       hip_sign * (-L_THIGH * c_q2 - L_CALF * c_q23), hip_sign * (-L_CALF * c_q23)],
#     #     [-d_y * s_q1 + Lc * c_q1,   hip_sign * (-Ls * s_q1),                         hip_sign * (-L_CALF * s_q23 * s_q1)],
#     #     [ d_y * c_q1 + Lc * s_q1,   hip_sign * (Ls * c_q1),                          hip_sign * ( L_CALF * s_q23 * c_q1)],
#     # ])
#     return np.array([
#         [0.0, hip_sign * (-L_THIGH * c_q2 - L_CALF * c_q23), -L_CALF * c_q23],
#         [-d_y * s_q1 + Lc * c_q1, hip_sign * (-Ls * s_q1), -L_CALF * s_q23 * s_q1],
#         [ d_y * c_q1 + Lc * s_q1, hip_sign * ( Ls * c_q1),  L_CALF * s_q23 * c_q1],
#     ])


# class LevenbergMarquardtIK:
#     """
#     Levenberg-Marquardt IK for a single Go2 leg.

#     Hybrid of Gauss-Newton (large error) and gradient descent (small error).
#     Uses analytical forward kinematics and Jacobian — no MuJoCo dependency.

#     Input:  target foot position [x, y, z] relative to hip origin.
#     Output: dict[Joint, float] for the 3 joints of that leg.
#     """

#     def __init__(
#         self,
#         leg: Leg,
#         step_size: float = 1.0,
#         tol: float = 1e-4,
#         damping: float = 0.05,
#         max_iter: int = 500,
#     ) -> None:
#         self.leg = leg
#         self.config = LEG_CONFIG[leg]
#         self.d_y = self.config['d_y']
#         self.step_size = step_size
#         self.tol = tol
#         self.damping = damping
#         self.max_iter = max_iter

#         # Joint limits: rows = [abduction, hip_pitch, knee]
#         self.joint_limits = np.array([
#             ABDUCTION_POS_RANGE,
#             self.config['hip_range'],
#             KNEE_POS_RANGE,
#         ])

#     def clip_to_joint_limits(self, q: np.ndarray) -> None:
#         """Clamp joint angles to their limits in-place."""
#         for i in range(3):
#             q[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])

#     # Numerical inverse kinematics (IK) solver, using:
#     # Levenberg-Marquardt pseudocode implementation / damped least-squared method
#     # Goal: find joint angles so that the foot reaches a desired cartesian position
#     def calculate(self, goal: np.ndarray, init_q: np.ndarray = None) -> dict[Joint, float]:
#         """
#         Calculate the desired joint angles for a target foot position.

#         Args:
#             goal:   target foot position [x, y, z] relative to hip origin.
#             init_q: initial guess [q_abd, q_hip, q_knee].  Defaults to home pose (0, 0.9, -1.8).

#         Returns:
#             dict mapping the leg's three Joint enums to their solved angles (radians).

#         Raises:
#             RuntimeError: if the solver does not converge within max_iter.
#         """
#         if init_q is None: # just an initial joint pos - helps convergence - typical go2 stance
#             init_q = np.array([0.0, 0.9, -1.8])

#         # clip to limits
#         q = init_q.copy().astype(float)
#         # rear legs have mirrored hip kinematics
#         hip_sign = -1 if self.leg in (Leg.RL, Leg.RR) else 1

#         q[0] = init_q[0]  # keep abduction fixed
#         self.clip_to_joint_limits(q)

#         # compute forward kinematics
#         current_pose = forward_kinematics(q, self.d_y, hip_sign) # our method
#         # error = np.subtract(goal, current_pose)        # error - between desired and actual pos - vector 1x3
#         error = goal[[0,2]] - current_pose[[0,2]]

#         # now we try to minimize error by computing joint positions again and again
#         for _ in range(self.max_iter):
#             if np.linalg.norm(error) < self.tol: # if error is low enough, break
#                 break

#             # calculate jacobian
#             # J = leg_jacobian(q, self.d_y)[[0, 2], 1:3] # our method dp = J(q)dq
#             J = leg_jacobian(q, self.d_y, hip_sign)[[0, 2], 1:3]

#             # calculate delta of joint q
#             n = J.shape[1]
#             I = np.identity(n)
#             product = J.T @ J + self.damping * I # J^T*J+lambda*I - damping is lambda - the LM system
#             # why damping? because if (J^T*J)^{-1} is near a singularity, it can explode - damping stabilizes

#             # # if the determinant is close to 0 = close to singularity
#             # if np.isclose(np.linalg.det(product), 0):
#             #     j_inv = np.linalg.pinv(product) @ J.T # pseudo-inverse - if close to singularity
#             # else:
#             #     j_inv = np.linalg.inv(product) @ J.T  # inverse
#             j_inv = np.linalg.pinv(product) @ J.T

#             delta_q = j_inv @ error # compute joint correction - delta q = (J^TJ + lambda*I)^{-1}*J^T*error - converts cartesian error into joint motion
#             # e.g.:
#             # error   = [0.02,  0.01, -0.02]
#             # delta q = [0.01, -0.04,  0.02] - +0.01 in abduction, -0.04 in hip, +0.02 in knee

#             # compute next step - update joint positions
#             q[1:] += self.step_size * delta_q
#             # clip to joint limits
#             self.clip_to_joint_limits(q)
#             # compute forward kinematics
#             current_pose = forward_kinematics(q, self.d_y, hip_sign) # our method
#             # calculate new error - cartesian
#             # error = np.subtract(goal, current_pose)
#             error = goal[[0,2]] - current_pose[[0,2]]
#         else:
#             residual = np.linalg.norm(error)
#             if residual >= self.tol: # if max iterations is reached - cannot find joint position
#                 raise RuntimeError(
#                     f"IK did not converge after {self.max_iter} iterations "
#                     f"(residual={residual:.6f}, tol={self.tol})"
#                 )

#         hip_joint, thigh_joint, calf_joint = self.config['joints']
#         return {
#             hip_joint:   float(q[0]),
#             thigh_joint: float(q[1] * hip_sign),
#             calf_joint:  float(q[2]),
#         }
#         # damped least-squares inverse kinematics does:
#         # delta q = (J^T*J + lambda*I)^{-1} * J^T * (p_desired - forward_kinematics(q))

#         # stable
#         # handles singularities
#         # simple to implement
#         # works well for legs

# def test_ik_from_foot_positions(foot_positions: dict[Leg, np.ndarray]):
#     """
#     Test IK using measured foot positions.

#     For each leg:
#         foot position → IK → joint angles → FK → compare error
#     """

#     print("\n================ IK FOOT POSITION TEST ================\n")

#     for leg, target in foot_positions.items():

#         config = LEG_CONFIG[leg]
#         d_y = config["d_y"]

#         print(f"Leg: {leg.name}")
#         print("Target foot position (raw):", target)

#         # convert coordinate frame
#         target_model = convert_frame(target)
#         target_model[1] = d_y

#         print("Target in IK frame:", target_model)

#         solver = LevenbergMarquardtIK(leg)

#         try:
#             # Solve IK
#             if leg in (Leg.RL, Leg.RR):
#                 init = np.array([0.00156, -0.79, -1.50])
#             else:
#                 init = np.array([0.00156, 0.79, -1.50])
#             result = solver.calculate(target_model, init_q=init)

#             # convert dict → vector
#             q = np.array([result[j] for j in config["joints"]])

#             print("Solved joint angles:", q)

#             # Forward kinematics check
#             fk = forward_kinematics(q, d_y)

#             print("FK position:", fk)

#             # Cartesian error
#             error_vec = fk - target_model
#             error_norm = np.linalg.norm(error_vec)

#             print("Cartesian error vector:", error_vec)
#             print("Cartesian error norm:", error_norm)

#             if error_norm < 1e-3:
#                 print("✓ IK solution valid\n")
#             else:
#                 print("✗ Large IK error\n")

#         except RuntimeError as e:
#             print("✗ IK solver failed:", e)
#             print()

# if __name__ == "__main__":
#     # Test with home pose for FL leg
#     foot_targets = {
#         Leg.FL: np.array([0.1795716643722928, 0.14149099862255338, 0.18806348777447116]),
#         Leg.FR: np.array([0.1795716705870714, -0.141490825432425, 0.18806342628408043]),
#         Leg.RL: np.array([-0.26887864336080886, 0.14201085282060438, 0.19723781364348866]),
#         Leg.RR: np.array([-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]),
#     }
#     FR_joint_pos = [0.0015597710497448469, 0.7941666367506252, -1.4975358491294977]

#     test_ik_from_foot_positions(foot_targets)
#     # q_home = np.array([0.0, 0.9, -1.8])
#     # test_ik_round_trip(Leg.FL, q_home)

# import os, sys
# import numpy as np
# from enum import Enum, auto

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from shared_module.robot_state import Joint
# from shared_module.global_constants import (
#     ABDUCTION_POS_RANGE, FRONT_HIP_POS_RANGE, BACK_HIP_POS_RANGE, KNEE_POS_RANGE
# )


# class Leg(Enum):
#     FL = auto()
#     FR = auto()
#     RR = auto()
#     RL = auto()


# # ── Go2 leg geometry (meters) ─────────────────────────────
# L_THIGH = 0.213
# L_CALF = 0.213


# LEG_CONFIG = {
#     Leg.FL: {'d_y':  0.141490825432425, 'joints': (Joint.FL_HIP, Joint.FL_THIGH, Joint.FL_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
#     Leg.FR: {'d_y': -0.141490825432425, 'joints': (Joint.FR_HIP, Joint.FR_THIGH, Joint.FR_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
#     Leg.RL: {'d_y':  0.141490825432425, 'joints': (Joint.RL_HIP, Joint.RL_THIGH, Joint.RL_CALF), 'hip_range': BACK_HIP_POS_RANGE},
#     Leg.RR: {'d_y': -0.141490825432425, 'joints': (Joint.RR_HIP, Joint.RR_THIGH, Joint.RR_CALF), 'hip_range': BACK_HIP_POS_RANGE},
# }


# # ── Forward kinematics ────────────────────────────────────

# def forward_kinematics(q: np.ndarray, d_y: float): # front: +hip_sign, rear: -hip_sign

#     q1, q2, q3 = q

#     s_q1, c_q1 = np.sin(q1), np.cos(q1)

#     # s_q2 = np.sin(hip_sign * q2)
#     # c_q2 = np.cos(hip_sign * q2)

#     # s_q23 = np.sin(hip_sign * q2 + q3)
#     # c_q23 = np.cos(hip_sign * q2 + q3)
#     s_q2 = np.sin(q2)
#     c_q2 = np.cos(q2)

#     s_q23 = np.sin(q2 + q3)
#     c_q23 = np.cos(q2 + q3)

#     x = -L_THIGH * s_q2 - L_CALF * s_q23

#     y = d_y * c_q1 + (L_THIGH * c_q2 + L_CALF * c_q23) * s_q1

#     z = d_y * s_q1 - (L_THIGH * c_q2 + L_CALF * c_q23) * c_q1

#     return np.array([x, y, z])


# # ── Frame conversion (MuJoCo → IK) ────────────────────────

# def convert_frame(pos: np.ndarray):
#     p = pos.copy()
#     p[2] *= -1
#     return p


# # ── Analytical Jacobian ───────────────────────────────────

# def leg_jacobian(q: np.ndarray, d_y: float):

#     q1, q2, q3 = q

#     s_q1, c_q1 = np.sin(q1), np.cos(q1)

#     s_q2 = np.sin(q2)
#     c_q2 = np.cos(q2)

#     s_q23 = np.sin(q2 + q3)
#     c_q23 = np.cos(q2 + q3)

#     Lc = L_THIGH * c_q2 + L_CALF * c_q23
#     Ls = L_THIGH * s_q2 + L_CALF * s_q23

#     return np.array([
#         [0.0,
#          (-L_THIGH * c_q2 - L_CALF * c_q23),
#          -L_CALF * c_q23],

#         [-d_y * s_q1 + Lc * c_q1,
#          (-Ls * s_q1),
#          -L_CALF * s_q23 * s_q1],

#         [d_y * c_q1 + Lc * s_q1,
#          (Ls * c_q1),
#          L_CALF * s_q23 * c_q1]
#     ])


# # ── Levenberg-Marquardt IK ─────────────────────────────────

# class LevenbergMarquardtIK:

#     def __init__(self, leg: Leg,
#                  step_size=1.0,
#                  tol=1e-4,
#                  damping=0.05,
#                  max_iter=500):

#         self.leg = leg
#         self.config = LEG_CONFIG[leg]

#         self.d_y = self.config['d_y']

#         self.step_size = step_size
#         self.tol = tol
#         self.damping = damping
#         self.max_iter = max_iter

#         self.joint_limits = np.array([
#             ABDUCTION_POS_RANGE,
#             self.config['hip_range'],
#             KNEE_POS_RANGE
#         ])

#     def clip_to_joint_limits(self, q):

#         for i in range(3):
#             q[i] = np.clip(q[i],
#                            self.joint_limits[i, 0],
#                            self.joint_limits[i, 1])

#     def calculate(self, goal, init_q=None):

#         if init_q is None:
#             init_q = np.array([0.0, 0.9, -1.8])

#         q = init_q.copy().astype(float)

#         self.clip_to_joint_limits(q)

#         current_pose = forward_kinematics(q, self.d_y)

#         error = goal[[0, 2]] - current_pose[[0, 2]]

#         for _ in range(self.max_iter):

#             if np.linalg.norm(error) < self.tol:
#                 break

#             J = leg_jacobian(q, self.d_y)[[0, 2], 1:3]

#             n = J.shape[1]

#             I = np.identity(n)

#             product = J.T @ J + self.damping * I

#             j_inv = np.linalg.pinv(product) @ J.T

#             delta_q = j_inv @ error

#             q[1:] += self.step_size * delta_q

#             self.clip_to_joint_limits(q)

#             current_pose = forward_kinematics(q, self.d_y)

#             error = goal[[0, 2]] - current_pose[[0, 2]]

#         else:

#             residual = np.linalg.norm(error)

#             if residual >= self.tol:
#                 raise RuntimeError(
#                     f"IK did not converge after {self.max_iter} iterations "
#                     f"(residual={residual:.6f}, tol={self.tol})"
#                 )

#         hip_joint, thigh_joint, calf_joint = self.config['joints']

#         return {
#             hip_joint: float(q[0]),
#             # thigh_joint: float(q[1] * hip_sign),
#             thigh_joint: float(q[1]),
#             calf_joint: float(q[2])
#         }
#         # return {
#         #     hip_joint: float(init_q[0]),
#         #     thigh_joint: float(init_q[1] * hip_sign),
#         #     calf_joint: float(init_q[2])
#         # }



# # ── Test routine ───────────────────────────────────────────

# def test_ik_from_foot_positions(foot_positions):

#     print("\n================ IK FOOT POSITION TEST ================\n")

#     for leg, target in foot_positions.items():

#         config = LEG_CONFIG[leg]
#         d_y = config["d_y"]

#         print(f"Leg: {leg.name}")

#         target_model = convert_frame(target)

#         # Rear hip frames are rotated 180° around Z
#         if leg in (Leg.RL, Leg.RR):
#             target_model[0] *= -1
#             target_model[1] *= -1

#         target_model[1] = d_y

#         solver = LevenbergMarquardtIK(leg)

#         try:

#             if leg in (Leg.RL, Leg.RR):
#                 init = np.array([0.00156, -0.79, -1.50])
#             else:
#                 init = np.array([0.00156, 0.79, -1.50])

#             result = solver.calculate(target_model, init_q=init)

#             q = np.array([result[j] for j in config["joints"]])

#             fk = forward_kinematics(q, d_y)

#             error_vec = fk - target_model
#             error_norm = np.linalg.norm(error_vec)

#             print("Solved joint angles:", q)
#             print("Cartesian error:", error_norm)

#             if error_norm < 1e-3:
#                 print("✓ IK solution valid\n")
#             else:
#                 print("✗ Large IK error\n")

#         except RuntimeError as e:
#             print("✗ IK solver failed:", e)
#             print()


# if __name__ == "__main__":

#     foot_targets = {
#         Leg.FL: np.array([0.1795716643722928, 0.14149099862255338, 0.18806348777447116]),
#         Leg.FR: np.array([0.1795716705870714, -0.141490825432425, 0.18806342628408043]),
#         Leg.RL: np.array([-0.26887864336080886, 0.14201085282060438, 0.19723781364348866]),
#         Leg.RR: np.array([-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]),
#     }
#     joint_targets = {
#         Leg.FL: np.array([0.0015597710497448469, 0.7941666367506252, -1.4975358491294977]),
#         Leg.FR: np.array([0.0015597710497448469, 0.7941666367506252, -1.4975358491294977]),
#         Leg.RL: np.array([0.0015597710497448469, -0.7941666367506252, -1.4975358491294977]),
#         Leg.RR: np.array([0.0015597710497448469, -0.7941666367506252, -1.4975358491294977]),
#     }

#     test_ik_from_foot_positions(foot_targets)

import os, sys
import numpy as np
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.robot_state import Joint, Foot
from shared_module.global_constants import (
    ABDUCTION_POS_RANGE,
    FRONT_HIP_POS_RANGE,
    BACK_HIP_POS_RANGE,
    KNEE_POS_RANGE
)

# ── Go2 leg geometry (meters) ─────────────────────────────

L_THIGH = 0.213
L_CALF = 0.213


LEG_CONFIG = {
    Foot.FL: {'d_y':  0.141490825432425, 'joints': (Joint.FL_HIP, Joint.FL_THIGH, Joint.FL_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
    Foot.FR: {'d_y': -0.141490825432425, 'joints': (Joint.FR_HIP, Joint.FR_THIGH, Joint.FR_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
    Foot.RL: {'d_y':  0.141490825432425, 'joints': (Joint.RL_HIP, Joint.RL_THIGH, Joint.RL_CALF), 'hip_range': BACK_HIP_POS_RANGE},
    Foot.RR: {'d_y': -0.141490825432425, 'joints': (Joint.RR_HIP, Joint.RR_THIGH, Joint.RR_CALF), 'hip_range': BACK_HIP_POS_RANGE},
}
HIP_OFFSETS = {
    Foot.FL: np.array([0.1934,  0.0465, 0.445]),
    Foot.FR: np.array([0.1934, -0.0465, 0.445]),
    Foot.RL: np.array([-0.1934,  0.0465, 0.445]),
    Foot.RR: np.array([-0.1934, -0.0465, 0.445]),
}

# ── Forward kinematics ────────────────────────────────────

def forward_kinematics(q: np.ndarray, d_y: float):

    q1, q2, q3 = q

    s_q1, c_q1 = np.sin(q1), np.cos(q1)

    s_q2 = np.sin(q2)
    c_q2 = np.cos(q2)

    s_q23 = np.sin(q2 + q3)
    c_q23 = np.cos(q2 + q3)

    x = -L_THIGH * s_q2 - L_CALF * s_q23

    y = d_y * c_q1 + (L_THIGH * c_q2 + L_CALF * c_q23) * s_q1

    z = d_y * s_q1 - (L_THIGH * c_q2 + L_CALF * c_q23) * c_q1

    return np.array([x, y, z])


# ── Frame conversion (MuJoCo → IK hip frame) ──────────────

def convert_frame(pos: np.ndarray, leg: Foot):

    p = pos.copy()

    # MuJoCo: Z up
    # IK model: Z down
    # p[2] *= -1

    # Rear legs are mounted facing backward
    if leg in (Foot.RL, Foot.RR):
        # p[0] *= -1
        # p[1] *= -1
        pass

    return p

# ── Analytical Jacobian ───────────────────────────────────

def leg_jacobian(q: np.ndarray, d_y: float) -> np.ndarray:  

    q1, q2, q3 = q

    s_q1, c_q1 = np.sin(q1), np.cos(q1)

    s_q2 = np.sin(q2)
    c_q2 = np.cos(q2)

    s_q23 = np.sin(q2 + q3)
    c_q23 = np.cos(q2 + q3)

    Lc = L_THIGH * c_q2 + L_CALF * c_q23
    Ls = L_THIGH * s_q2 + L_CALF * s_q23

    return np.array([
        [0.0,
         (-L_THIGH * c_q2 - L_CALF * c_q23),
         -L_CALF * c_q23],

        [-d_y * s_q1 + Lc * c_q1,
         (-Ls * s_q1),
         -L_CALF * s_q23 * s_q1],

        [d_y * c_q1 + Lc * s_q1,
         (Ls * c_q1),
         L_CALF * s_q23 * c_q1]
    ])


# ── Levenberg–Marquardt IK ─────────────────────────────────

class LevenbergMarquardtIK:

    def __init__(
        self,
        leg: Foot,
        step_size=1.0,
        tol=1e-4,
        damping=0.05,
        max_iter=500
    ):

        self.leg = leg
        self.config = LEG_CONFIG[leg]

        self.d_y = self.config['d_y']

        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        self.max_iter = max_iter

        self.joint_limits = np.array([
            ABDUCTION_POS_RANGE,
            self.config['hip_range'],
            KNEE_POS_RANGE
        ])

    def clip_to_joint_limits(self, q):

        for i in range(3):
            q[i] = np.clip(
                q[i],
                self.joint_limits[i, 0],
                self.joint_limits[i, 1]
            )

    def calculate(self, goal, init_q=None):

        if init_q is None:
            init_q = np.array([0.0, 0.9, -1.8])

        q = init_q.copy().astype(float)

        self.clip_to_joint_limits(q)

        current_pose = forward_kinematics(q, self.d_y)

        error = goal[[0, 2]] - current_pose[[0, 2]]

        for _ in range(self.max_iter):

            if np.linalg.norm(error) < self.tol:
                break

            J = leg_jacobian(q, self.d_y)[[0, 2], 1:3]

            I = np.identity(J.shape[1])

            product = J.T @ J + self.damping * I

            j_inv = np.linalg.pinv(product) @ J.T

            delta_q = j_inv @ error

            q[1:] += self.step_size * delta_q

            self.clip_to_joint_limits(q)

            current_pose = forward_kinematics(q, self.d_y)

            error = goal[[0, 2]] - current_pose[[0, 2]]

        else:

            residual = np.linalg.norm(error)

            if residual >= self.tol:
                raise RuntimeError(
                    f"IK did not converge after {self.max_iter} iterations "
                    f"(residual={residual:.6f}, tol={self.tol})"
                )

        hip_joint, thigh_joint, calf_joint = self.config['joints']

        return {
            hip_joint: float(q[0]),
            thigh_joint: float(q[1]),
            calf_joint: float(q[2])
        }


# ── Test routine ───────────────────────────────────────────

def test_ik_from_foot_positions(foot_positions):

    print("\n================ IK FOOT POSITION TEST ================\n")

    for leg, target in foot_positions.items():

        config = LEG_CONFIG[leg]
        d_y = config["d_y"]

        print(f"Leg: {leg.name}")

        target_model = convert_frame(target, leg)

        print("Raw stance:", target)

        print("After frame conversion:", target_model)

        # expected pose from your initial guess
        init = np.array([0.00156, -0.79, -1.50]) if leg in (Foot.RL, Foot.RR) else np.array([0.00156, 0.79, -1.50])

        fk_init = forward_kinematics(init, d_y)

        print("FK from initial guess:", fk_init)

        print("Initial error (x,z):", target_model[[0,2]] - fk_init[[0,2]])
        print()

        # keep foot in sagittal plane
        # target_model[1] = d_y

        solver = LevenbergMarquardtIK(leg)

        try:

            if leg in (Foot.RL, Foot.RR):
                init = np.array([0.00156, -0.79, -1.50])
            else:
                init = np.array([0.00156, 0.79, -1.50])

            result = solver.calculate(target_model, init_q=init)

            q = np.array([result[j] for j in config["joints"]])

            fk = forward_kinematics(q, d_y)

            error_vec = fk[[0,2]] - target_model[[0,2]]
            error_norm = np.linalg.norm(error_vec)

            print("Solved joint angles:", q)
            print("Cartesian error:", error_norm)

            if error_norm < 1e-3:
                print("✓ IK solution valid\n")
            else:
                print("✗ Large IK error\n")

        except RuntimeError as e:
            print("✗ IK solver failed:", e)
            print()


if __name__ == "__main__":

    foot_targets = {
        Foot.FL: np.array([0.1795716643722928, 0.14149099862255338, 0.18806348777447116]),
        Foot.FR: np.array([0.1795716705870714, -0.141490825432425, 0.18806342628408043]),
        Foot.RL: np.array([-0.26887864336080886, 0.14201085282060438, 0.19723781364348866]),
        Foot.RR: np.array([-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]),
    }

    foot_positions = {
        'foot_FL': np.array([ 0.1797101023341874,   0.1413506224698631,  99.68767388573536]), 
        'foot_FR': np.array([ 0.17971010766906423, -0.1413504181104758,  99.6876738144711]), 
        'foot_RL': np.array([-0.26880895284567585,  0.14201380537581443, 99.69686760577544]), 
        'foot_RR': np.array([-0.26880894651812637, -0.14201497980323355, 99.6968679652466]),
    }
    hip_positions = {
        'hip_FL': np.array([ 0.19369817604619316,  0.04649998744108871, 100.0]),
        'hip_FR': np.array([ 0.19369817604619316, -0.04650001255891129, 100.0]),
        'hip_RL': np.array([-0.19310182395380682,  0.04649998744108871, 100.0]),
        'hip_RR': np.array([-0.19310182395380682, -0.04650001255891129, 100.0]),
    }
    thigh_positions = {
        'thigh_FL': [ 0.19369817604619316,  0.14199978089864376, 99.9998013808561],
        'thigh_FR': [ 0.19369817604619316, -0.14199980587051006, 99.99980131068997],
        'thigh_RL': [-0.19310182395380682,  0.1419999873418711,  100.00000435322426],
        'thigh_RR': [-0.19310182395380682, -0.14200001244250227, 100.00000471530736],
    }
    base_link_position = [0.00029817604619316766, -1.2558911291640273e-08, 100.0]

    stance_positions = {
        Foot.FL: np.array([-0.01458281454414101, 0.09520361349694849, -0.31155118257490244]),
        Foot.FR: np.array([-0.014582808367942734, -0.0952034741060266, -0.311551225724358]),
        Foot.RL: np.array([-0.07616549739558984, 0.0955063386302885, -0.3023549187505239]),
        Foot.RR: np.array([-0.076165490580238, -0.09550685969947971, -0.30235475609234186]),
    }
    stance_positions_hipf = {
        Foot.FL: np.array([ 0.01359,  0.10217, -0.26305]),
        Foot.FR: np.array([ 0.01359, -0.10217, -0.26305]),
        Foot.RL: np.array([-0.04041,  0.09929, -0.26944]),
        Foot.RR: np.array([-0.04041, -0.09929, -0.26944]),
    }
    ninety_degree_stance_new = {
        Foot.FL: np.array([-0.005036816173289977, 0.24416272947995582, 0.008882972842174985]),
        Foot.FR: np.array([-0.005036819539444062, -0.2441772706603052, 0.008882976190526026]),
        Foot.RL: np.array([-0.45696953150783753, 0.24128273012130325, 0.01339652496606547]),
        Foot.RR: np.array([-0.4569698918564412, -0.24129727017798086, 0.013396895780783069]),
    }
    from_cpg = {
        Foot.FL: np.array([-0.1864, 0.1022, -0.2627]),
        Foot.FR: np.array([0.2136, -0.1022, -0.2631]),
        Foot.RL: np.array([0.1596, 0.0993, -0.2694]),
        Foot.RR: np.array([-0.2404, -0.0993, -0.2691]),
    }
#      FL  x: -0.1864  y:  0.1022  z: -0.2627
#  FR  x:  0.2136  y: -0.1022  z: -0.2631
#  RL  x:  0.1596  y:  0.0993  z: -0.2694
#  RR  x: -0.2404  y: -0.0993  z: -0.2691

    test_ik_from_foot_positions(stance_positions_hipf)