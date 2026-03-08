import os, sys
import numpy as np
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Joint
from shared_module.global_constants import (
    ABDUCTION_POS_RANGE, FRONT_HIP_POS_RANGE, BACK_HIP_POS_RANGE, KNEE_POS_RANGE
)


class Leg(Enum):
    """Identifies which leg the IK solver operates on."""
    FL = auto()
    FR = auto()
    RR = auto()
    RL = auto()


# ── Go2 leg geometry (metres) ──────────────────────────────────
L_THIGH = 0.213   # thigh → calf link length (Z)
L_CALF  = 0.213   # calf  → foot link length (Z)

# Per-leg configuration: lateral hip-to-thigh offset, Joint enum mapping, hip range
LEG_CONFIG = {
    Leg.FL: {'d_y':  0.0955, 'joints': (Joint.FL_HIP, Joint.FL_THIGH, Joint.FL_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
    Leg.FR: {'d_y': -0.0955, 'joints': (Joint.FR_HIP, Joint.FR_THIGH, Joint.FR_CALF), 'hip_range': FRONT_HIP_POS_RANGE},
    Leg.RL: {'d_y':  0.0955, 'joints': (Joint.RL_HIP, Joint.RL_THIGH, Joint.RL_CALF), 'hip_range': BACK_HIP_POS_RANGE},
    Leg.RR: {'d_y': -0.0955, 'joints': (Joint.RR_HIP, Joint.RR_THIGH, Joint.RR_CALF), 'hip_range': BACK_HIP_POS_RANGE},
}


# ── Analytical forward kinematics & Jacobian for Go2 3-DOF leg ─

def forward_kinematics(q: np.ndarray, d_y: float) -> np.ndarray:
    """
    Compute foot position [x, y, z] relative to hip origin for a Go2 leg.

    Kinematic chain:
        abduction(X-axis) → lateral offset d_y → hip_pitch(Y-axis) → L_thigh
                          → knee_pitch(Y-axis) → L_calf → foot

    Args:
        q:   joint angles [q_abduction, q_hip, q_knee]
        d_y: signed lateral hip-to-thigh offset (+left, −right)
    """
    q1, q2, q3 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    x = -L_THIGH * s2 - L_CALF * s23
    y =  d_y * c1 + (L_THIGH * c2 + L_CALF * c23) * s1
    z =  d_y * s1 - (L_THIGH * c2 + L_CALF * c23) * c1
    return np.array([x, y, z])


def leg_jacobian(q: np.ndarray, d_y: float) -> np.ndarray:
    """
    Compute 3×3 positional Jacobian  J = ∂p/∂q  for a Go2 leg.

    Args:
        q:   joint angles [q_abduction, q_hip, q_knee]
        d_y: signed lateral hip-to-thigh offset
    """
    q1, q2, q3 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    Lc = L_THIGH * c2 + L_CALF * c23   # sum of cosine projections
    Ls = L_THIGH * s2 + L_CALF * s23   # sum of sine projections

    return np.array([
        [0.0,                   -L_THIGH * c2 - L_CALF * c23,  -L_CALF * c23      ],
        [-d_y * s1 + Lc * c1,   -Ls * s1,                       -L_CALF * s23 * s1 ],
        [ d_y * c1 + Lc * s1,    Ls * c1,                        L_CALF * s23 * c1 ],
    ])


class LevenbergMarquardtIK:
    """
    Levenberg-Marquardt IK for a single Go2 leg.

    Hybrid of Gauss-Newton (large error) and gradient descent (small error).
    Uses analytical forward kinematics and Jacobian — no MuJoCo dependency.

    Input:  target foot position [x, y, z] relative to hip origin.
    Output: dict[Joint, float] for the 3 joints of that leg.
    """

    def __init__(
        self,
        leg: Leg,
        step_size: float = 1.0,
        tol: float = 1e-4,
        damping: float = 0.01,
        max_iter: int = 500,
    ) -> None:
        self.leg = leg
        self.config = LEG_CONFIG[leg]
        self.d_y = self.config['d_y']
        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        self.max_iter = max_iter

        # Joint limits: rows = [abduction, hip_pitch, knee]
        self.joint_limits = np.array([
            ABDUCTION_POS_RANGE,
            self.config['hip_range'],
            KNEE_POS_RANGE,
        ])

    def check_joint_limits(self, q: np.ndarray) -> None:
        """Clamp joint angles to their limits in-place."""
        for i in range(3):
            q[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])

    # Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal: np.ndarray, init_q: np.ndarray = None) -> dict[Joint, float]:
        """
        Calculate the desired joint angles for a target foot position.

        Args:
            goal:   target foot position [x, y, z] relative to hip origin.
            init_q: initial guess [q_abd, q_hip, q_knee].  Defaults to home pose (0, 0.9, -1.8).

        Returns:
            dict mapping the leg's three Joint enums to their solved angles (radians).

        Raises:
            RuntimeError: if the solver does not converge within max_iter.
        """
        if init_q is None:
            init_q = np.array([0.0, 0.9, -1.8])

        q = init_q.copy().astype(float)
        self.check_joint_limits(q)

        # compute forward kinematics
        current_pose = forward_kinematics(q, self.d_y)
        error = np.subtract(goal, current_pose)

        for _ in range(self.max_iter):
            if np.linalg.norm(error) < self.tol:
                break

            # calculate jacobian
            J = leg_jacobian(q, self.d_y)

            # calculate delta of joint q
            n = J.shape[1]
            I = np.identity(n)
            product = J.T @ J + self.damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ J.T
            else:
                j_inv = np.linalg.inv(product) @ J.T

            delta_q = j_inv @ error

            # compute next step
            q += self.step_size * delta_q
            # check limits
            self.check_joint_limits(q)
            # compute forward kinematics
            current_pose = forward_kinematics(q, self.d_y)
            # calculate new error
            error = np.subtract(goal, current_pose)
        else:
            residual = np.linalg.norm(error)
            if residual >= self.tol:
                raise RuntimeError(
                    f"IK did not converge after {self.max_iter} iterations "
                    f"(residual={residual:.6f}, tol={self.tol})"
                )

        hip_joint, thigh_joint, calf_joint = self.config['joints']
        return {
            hip_joint:   float(q[0]),
            thigh_joint: float(q[1]),
            calf_joint:  float(q[2]),
        }  