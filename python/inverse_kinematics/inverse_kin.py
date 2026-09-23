"""
@brief Inverse-kinematics module for the Unitree Go2 legs.

Provides the analytical forward kinematics and Jacobian for a single leg, a
MuJoCo-to-IK frame conversion helper, a damped-least-squares Levenberg-Marquardt
IK solver mapping Cartesian foot targets to joint angles, and a standalone test
routine for validating IK against reference foot positions.
"""
import os, sys
import numpy as np

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
    """
    @brief Compute the foot position in the hip frame from leg joint angles.
    @param q: Array of the three joint angles (hip abduction, thigh, calf) in radians.
    @param d_y: Lateral hip-to-thigh offset (meters) for the leg, signed by side.
    @return numpy array [x, y, z] of the foot position in the hip frame (meters).
    """
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
    """
    @brief Convert a foot position from the MuJoCo frame into the IK hip frame.
    @param pos: Foot position [x, y, z] in the MuJoCo frame (meters).
    @param leg: The Foot whose frame convention is being applied.
    @return numpy array of the foot position expressed in the IK hip frame.
    """
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
    """
    @brief Compute the analytical 3x3 leg Jacobian mapping joint rates to foot velocity.
    @param q: Array of the three joint angles (hip abduction, thigh, calf) in radians.
    @param d_y: Lateral hip-to-thigh offset (meters) for the leg, signed by side.
    @return 3x3 numpy Jacobian matrix d(foot position)/d(joint angles) in the hip frame.
    """
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

def estimate_body_velocity(q, q_dot, d_y):
    """
    @brief Estimate body velocity from a stance leg's joint state via the leg Jacobian.
    @param q: Array of the three joint angles (hip abduction, thigh, calf) in radians.
    @param q_dot: Array of the corresponding joint velocities (rad/s).
    @param d_y: Lateral hip-to-thigh offset (meters) for the leg, signed by side.
    @return numpy array of the estimated body velocity (negated foot velocity) in the hip frame.
    """
    J = leg_jacobian(q, d_y)
    v_foot = J @ q_dot
    v_body = -v_foot
    return v_body

# ── Levenberg–Marquardt IK ─────────────────────────────────

class LevenbergMarquardtIK:
    """
    @brief Damped-least-squares (Levenberg-Marquardt) inverse-kinematics solver for one Go2 leg.

    Iteratively refines the thigh and calf joint angles so the analytical forward kinematics
    matches a Cartesian foot target in the sagittal (x, z) plane, clamping each iterate to the
    leg's joint limits.
    """

    def __init__(
        self,
        leg: Foot,
        step_size=1.0,
        tol=1e-4,
        damping=0.05,
        max_iter=500
    ):
        """
        @brief Construct the solver for a given leg with its geometry and convergence settings.
        @param leg: The Foot this solver targets; selects leg geometry and joint limits.
        @param step_size: Scale applied to each Levenberg-Marquardt joint-angle update.
        @param tol: Cartesian error norm (meters) below which iteration is considered converged.
        @param damping: Levenberg-Marquardt damping factor added to the normal equations.
        @param max_iter: Maximum number of solver iterations before declaring non-convergence.
        """
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
        """
        @brief Clamp the three joint angles in place to this leg's configured joint limits.
        @param q: Mutable array of the three joint angles (rad); modified in place.
        """
        for i in range(3):
            q[i] = np.clip(
                q[i],
                self.joint_limits[i, 0],
                self.joint_limits[i, 1]
            )

    def calculate(self, goal, init_q=None):
        """
        @brief Solve inverse kinematics for a Cartesian foot target via Levenberg-Marquardt iteration.
        @param goal: Target foot position [x, y, z] in the hip frame; only x and z are tracked.
        @param init_q: Optional initial joint-angle guess (warm start); defaults to a nominal stance pose.
        @return Dict mapping this leg's hip, thigh, and calf Joints to their solved angles (rad).
        """
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
    """
    @brief Run and report the IK solver against a set of reference foot positions.

    For each leg, converts the target to the IK frame, solves IK from a per-leg initial guess,
    recomputes forward kinematics, and prints the resulting joint angles and Cartesian error,
    flagging whether each solution is within tolerance.
    @param foot_positions: Mapping from each Foot to its target Cartesian position [x, y, z].
    """
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
