import array
import os, sys
import numpy as np
import mujoco as mj
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *

class MusclePdController:

    class DampingException(Exception):
        """Damping coefficient is negativ!"""

    def __init__(self, kp: float = 1.0, kd: float = 1.0) -> None:
        self.neurons_cnt = 4  # one oscillator per hip (FL, FR, RR, RL)
        self.model = None
        self.data = None
        self.dt = None  # read from MuJoCo model at init_controller

        # For hip 
        self.passive_kp = 40.0
        self.passive_kd = 4.0
        
        # For knee
        self.kp = kp
        self.kd = kd
        self.spring_constant = 1.0
        """
        We dont want over damped 
        overdamped        0 < ξ < 1
        Critically damped ξ = 1
        Underdamped       ξ > 1
        Should be citically damped or underdamped 
        """
        self.damping = 1.0
        self.__check_damping()

    def init_controller(self, model, data):
        """Called once by MujocoSim before the simulation loop."""
        self.model = model
        self.data = data
        self.dt = model.opt.timestep  # MuJoCo simulation dt
        self.sim_time = 0.0

        # Load the 'home' keyframe so the robot starts in a stable standing pose
        key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mj.mj_resetDataKeyframe(model, data, key_id)
            mj.mj_forward(model, data)
            print(f"Loaded 'home' keyframe (id={key_id})")
        else:
            print("Warning: 'home' keyframe not found, starting from default pose")

    def __check_damping(self) -> None:
        if self.damping == 1:
            print("System is critically damped")
        elif self.damping > 1:
            print("System is overdamped!!")
        elif self.damping > 0:
            print("system is underdamped")
        else: 
            raise MusclePdController.DampingException("Damping must be >= 0")

        
    def __compute_single_pd(self, theta_ref: float, theta_knee: float, d_theta: float) -> float:
        """Computation for the knee joint"""
        torque_knee = self.spring_constant*(theta_ref - theta_knee) - self.damping * d_theta

    def apply_pd(self, targets: list[float]) -> None: # , d_theta: list[float]
        """Apply PD control to all actuators towards the given targets."""
        for joint_name, actuator_idx in ACTUATOR_DICT.items():
            if joint_name not in targets:
                continue
            pos_idx = SENSOR_POS_DICT[joint_name]
            vel_idx = SENSOR_VEL_DICT[joint_name]

            current_pos = self.data.sensordata[pos_idx]
            current_vel = self.data.sensordata[vel_idx]
            target_pos = targets[joint_name]

            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel

            # Use passive (stiff) gains for hip/thigh, muscle-like gains for calf
            if 'calf' in joint_name:
                kp, kd = self.kp, self.kd
            else:
                kp, kd = self.passive_kp, self.passive_kd

            self.data.ctrl[actuator_idx] = (
                kp * position_error + kd * velocity_error
            )