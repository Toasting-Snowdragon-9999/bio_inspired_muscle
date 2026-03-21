"""
Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller for Decoding Skill Intelligence. 
Biomimetic Intelligence and Robotics, 3(2), [100100]. https://doi.org/10.1016/j.birob.2023.100100

"""
import numpy as np
import numpy.linalg as la

class ada_imp_ctrl( ):
    """Online impedance adaptation"""
    def __init__(self, dof, use_ioac=True):

        self.DOF = dof

        # Use ndarray everywhere
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        self.ff_tau_mat = np.zeros((self.DOF, 1))

        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))

        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))

        self.a = 0.2
        self.b = 5.0
        self.k = 0.05
        self.use_ioac = use_ioac


    def update_impedance(self, q, q_d, dq, dq_d):
        if self.use_ioac: 
            self.q = np.asarray(q).reshape(-1, 1)
            self.q_d = np.asarray(q_d).reshape(-1, 1)
            self.dq = np.asarray(dq).reshape(-1, 1)
            self.dq_d = np.asarray(dq_d).reshape(-1, 1)

            pos_err = self.q - self.q_d
            vel_err = self.dq - self.dq_d
            track_err = self.k * vel_err + pos_err

            norm_sq = la.norm(track_err)**2
            ad = self.a / (1.0 + self.b * norm_sq)

            ad = max(ad, 1e-6)  # prevent blow-up

            self.k_mat = (track_err @ pos_err.T) / ad
            self.b_mat = (track_err @ vel_err.T) / ad
        else: 
            # Normal non adaptive pd control, with fixed gains.
            kp = 20.0
            kd = 5.0
            self.k_mat = np.diag(np.full(self.DOF, kp))
            self.b_mat = np.diag(np.full(self.DOF, kd))

        return self.k_mat, self.b_mat

    def gen_pos_err(self):#position error, see Eq. (1)
        return (self.q - self.q_d)

    def gen_vel_err(self):#velocity error, see Eq. (1)
        return (self.dq - self.dq_d)

    def gen_track_err(self):#tracking error, see Eq. (3)
        return (self.k * self.gen_vel_err() + self.gen_pos_err())

    def gen_ad_factor(self):#adaptation scalar, see Eq. (3)
        return self.a/(1.0 + self.b * la.norm(self.gen_track_err()) * la.norm(self.gen_track_err()))

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
