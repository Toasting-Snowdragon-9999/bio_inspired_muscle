"""
Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller for Decoding Skill Intelligence. 
Biomimetic Intelligence and Robotics, 3(2), [100100]. https://doi.org/10.1016/j.birob.2023.100100

"""
import numpy as np
import numpy.linalg as la

class ada_imp_ctrl():
    """@brief Online impedance adaptation controller (OIAC) computing adaptive stiffness and damping matrices.
    Online impedance adaptation"""
    def __init__(self, dof, params: tuple[float, float, float] = (0.2, 5.0, 0.05),  use_oiac=True):
        """@brief Initialise the OIAC controller state, matrices, and adaptation parameters.
        When Using the MuscleLikePD controller, the default params cant be used
        @param dof: Number of degrees of freedom (joints) this controller adapts.
        @param params: (a, b, k) tuple of OIAC adaptation parameters (a, b control the adaptation factor; k weights velocity error).
            May be None — e.g. load_settings_from_file() returns None when a gait's
            .ini file has no [oiac] section — in which case the defaults above are used.
        @param use_oiac: If True use online adaptive impedance; otherwise use fixed-gain diagonal PD.
        """
        self.DOF = dof

        # Use ndarray everywhere
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        self.ff_tau_mat = np.zeros((self.DOF, 1))

        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))

        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))

        # A caller may pass params=None explicitly (which bypasses the default in
        # the signature), so fall back to the defaults rather than failing to unpack.
        if params is None:
            params = (0.2, 5.0, 0.05)
        self.a, self.b, self.k = params

        self.use_oiac = use_oiac


    def update_impedance(self, q, q_d, dq, dq_d):
        """
        @brief Update and return the stiffness and damping matrices from the current joint state.

        In OIAC mode the matrices are computed from the outer product of the tracking error with the
        position/velocity errors, scaled by the adaptation factor; otherwise fixed-gain diagonal
        matrices are returned.
        @param q: Measured joint positions (column vector / array, reshaped to (DOF, 1)).
        @param q_d: Desired joint positions (column vector / array, reshaped to (DOF, 1)).
        @param dq: Measured joint velocities (column vector / array, reshaped to (DOF, 1)).
        @param dq_d: Desired joint velocities (column vector / array, reshaped to (DOF, 1)).
        @return Tuple (k_mat, b_mat) of the DOFxDOF stiffness and damping matrices.
        """
        if self.use_oiac:
            self.q = np.asarray(q).reshape(-1, 1)
            self.q_d = np.asarray(q_d).reshape(-1, 1)
            self.dq = np.asarray(dq).reshape(-1, 1)
            self.dq_d = np.asarray(dq_d).reshape(-1, 1)

            self.k_mat = (self.gen_track_err @ self.gen_pos_err.T) / self.gen_ad_factor
            self.b_mat = (self.gen_track_err @ self.gen_vel_err.T) / self.gen_ad_factor
        else:
            # Normal non adaptive pd control, with fixed gains.
            # NOTE on signs: gen_pos_err/gen_vel_err in this module are (q_d - q) and
            # (dq_d - dq) — see the properties below. The *consumer* (muscle_like_pd.py)
            # applies the control law as tau = K @ (q_d - q) + B @ (dq_d - dq), so the
            # positive diagonal K, B below are stabilising (they restore the joint toward q_d).
            self.q = np.asarray(q).reshape(-1, 1)
            self.q_d = np.asarray(q_d).reshape(-1, 1)
            self.dq = np.asarray(dq).reshape(-1, 1)
            self.dq_d = np.asarray(dq_d).reshape(-1, 1)
            kp = 40.0
            kd = 2.0  # was 15.0 — too high; with MAX_JOINT_VEL=5 rad/s and 23.7 Nm motor limit,
                      # velocity errors >1.5 rad/s saturate the actuator and cause overshoot spasms.
            self.k_mat = np.diag(np.full(self.DOF, kp))
            self.b_mat = np.diag(np.full(self.DOF, kd))

        return self.k_mat, self.b_mat

    @property
    def gen_pos_err(self):#position error, see Eq. (1)
        """@brief Generalised position error (desired minus measured joint positions), see Eq. (1).
        @return Column vector of the position error q_d - q.
        """
        return (self.q_d - self.q)

    @property
    def gen_vel_err(self):#velocity error, see Eq. (1)
        """@brief Generalised velocity error (desired minus measured joint velocities), see Eq. (1).
        @return Column vector of the velocity error dq_d - dq.
        """
        return (self.dq_d - self.dq)

    @property
    def gen_track_err(self):#tracking error, see Eq. (3)
        """@brief Generalised tracking error combining velocity and position errors, see Eq. (3).
        @return Column vector k * gen_vel_err + gen_pos_err.
        """
        return (self.k * self.gen_vel_err + self.gen_pos_err)

    @property
    def gen_ad_factor(self):#adaptation scalar, see Eq. (3)
        """@brief Scalar adaptation factor scaling the impedance matrices, see Eq. (3).
        @return Clamped adaptation scalar a / (1 + b * ||tracking error||^2), floored at 1e-6.
        """
        ad = self.a/(1.0 + self.b * la.norm(self.gen_track_err) * la.norm(self.gen_track_err))
        return max(ad, 1e-6)  # clamp to prevent blow-up when dividing K and B

    """
    #Pseudocode

    if __name__ == "__main__":
        self.ada_imp = aic.ada_imp_con(dof) # degree of freedom of robot arm

        while(t<t_max):
            ...
            #get robot arm joint angles and velocity
            #NOTE THAT the online impedance adaptation control requires the feedforward control tau_ff, e.g., 
                #(1) linear (tracking) error-based control, e.g., tau_ff = a * e, a is the scalar constant, e  is the error between desired and real joint position.
            #(2) data-driven learning or opmtimization of feedforward joint torque(s) tau_ff

            self.ada_imp.(q, q_d, dq, dq_d) #undate stiffness (self.k_mat) and damping (self.b_mat) matrices 

            tau_fb =  self.k_mat* (q_d-q) + self.k_mat*(dq_d-dq) #compute feedback joint torque(s)

            tau = tau_ff + tau_fb #compute total joint torques

            ...
    """
