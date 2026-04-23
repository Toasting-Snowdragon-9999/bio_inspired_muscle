import os
import sys
os.environ.setdefault('MUJOCO_GL', 'glfw')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mujoco_sim import MujocoSim
import glfw
from shared_module.robot_state import RobotInterface, State, Mode, Gait, Joint
from shared_module.global_constants import (
    KNEE_POS_RANGE, FRONT_HIP_POS_RANGE, BACK_HIP_POS_RANGE, ABDUCTION_POS_RANGE, Foot
)



class InteractiveController:

    # Maps (leg, joint) string pair -> Joint enum
    JOINT_MAP = {
        ('front_right', 'hip'):   Joint.FR_HIP,
        ('front_right', 'thigh'): Joint.FR_THIGH,
        ('front_right', 'knee'):  Joint.FR_CALF,
        ('front_left',  'hip'):   Joint.FL_HIP,
        ('front_left',  'thigh'): Joint.FL_THIGH,
        ('front_left',  'knee'):  Joint.FL_CALF,
        ('rear_right',  'hip'):   Joint.RR_HIP,
        ('rear_right',  'thigh'): Joint.RR_THIGH,
        ('rear_right',  'knee'):  Joint.RR_CALF,
        ('rear_left',   'hip'):   Joint.RL_HIP,
        ('rear_left',   'thigh'): Joint.RL_THIGH,
        ('rear_left',   'knee'):  Joint.RL_CALF,
    }

    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        self.prev_fl = 0.0, 0.0, 0.0

        self.current_leg = 'front_right'
        self.current_joint = 'hip'
        self.current_range = FRONT_HIP_POS_RANGE

        self.step_size = 0.05  # radians per key press

        # Target joint positions for standing configuration
        self.target_positions = {
            Joint.FR_HIP: 0.0,  Joint.FR_THIGH: 0.8,  Joint.FR_CALF: -1.5,
            Joint.FL_HIP: 0.0,  Joint.FL_THIGH: 0.8,  Joint.FL_CALF: -1.5,
            Joint.RR_HIP: 0.0,  Joint.RR_THIGH: 1.0,  Joint.RR_CALF: -1.5,
            Joint.RL_HIP: 0.0,  Joint.RL_THIGH: 1.0,  Joint.RL_CALF: -1.5,
        }

        # PD gains
        self.kp = 1.0
        self.kd = 1.0

    def _current_joint_enum(self) -> Joint:
        return self.JOINT_MAP[(self.current_leg, self.current_joint)]

    def keyboard_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input for interactive control."""
        if action == glfw.PRESS or action == glfw.REPEAT:

            if key == glfw.KEY_1:
                self.current_leg = 'front_right'
            elif key == glfw.KEY_2:
                self.current_leg = 'front_left'
            elif key == glfw.KEY_3:
                self.current_leg = 'rear_right'
            elif key == glfw.KEY_4:
                self.current_leg = 'rear_left'

            elif key == glfw.KEY_Q:
                self.current_joint = 'hip'
            elif key == glfw.KEY_W:
                self.current_joint = 'thigh'
            elif key == glfw.KEY_E:
                self.current_joint = 'knee'

            if self.current_joint == 'hip':
                self.current_range = ABDUCTION_POS_RANGE
            elif self.current_joint == 'thigh':
                self.current_range = FRONT_HIP_POS_RANGE if 'front' in self.current_leg else BACK_HIP_POS_RANGE
            elif self.current_joint == 'knee':
                self.current_range = KNEE_POS_RANGE

            joint = self._current_joint_enum()

            if key == glfw.KEY_UP:
                self.target_positions[joint] = min(
                    self.target_positions[joint] + self.step_size, self.current_range[1])
            elif key == glfw.KEY_DOWN:
                self.target_positions[joint] = max(
                    self.target_positions[joint] - self.step_size, self.current_range[0])
            elif key == glfw.KEY_R:
                self.target_positions[joint] = 0.0

    def run(self):
        """PD controller: reads state from robot_interface, writes torques back."""
        joint_positions = self.robot_interface.joint_positions
        joint_velocities = self.robot_interface.joint_velocities

        computed_targets = {}
        for joint in Joint:
            current_pos = joint_positions.get(joint, 0.0)
            current_vel = joint_velocities.get(joint, 0.0)
            target_pos = self.target_positions.get(joint, 0.0)

            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel

            computed_targets[joint] = self.kp * position_error + self.kd * velocity_error

        self.robot_interface.target_positions = computed_targets
        
        fl = self.robot_interface.foot_positions.get(Foot.FL.value)
        if fl:
            fl = np.array(fl)
            prev = np.array(self.prev_fl)
            if np.linalg.norm(fl - prev) > 0.001:
                print(self.robot_interface.foot_positions)
                print(f"FL_foot: x={fl[0]:.4f}  y={fl[1]:.4f}  z={fl[2]:.4f}")
                self.prev_fl = tuple(fl)

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    robot_interface = RobotInterface(State(mode=Mode.MOVING, gait=Gait.NONE))
    controller = InteractiveController(robot_interface)
    sim = MujocoSim(xml_path, robot_interface)
    sim.enable_air_mode(0.5)
    sim.sim(controller=controller, sim_length=-1)


if __name__ == "__main__":
    main()
