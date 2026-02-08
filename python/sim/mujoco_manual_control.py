import os 
from mujoco_sim import MujocoSim
import glfw



class InteractiveController:
    # KNEE_CONTROL_RANGE = {-45.43, 45.43}
    # DEFAULT_CONTROL_RANGE = {-23.7, 23.7}

    KNEE_POS_RANGE      = (-2.7227, -0.83776)  # radians
    FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)  # radians
    BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)  # radians
    ABDUCTION_POS_RANGE = (-1.0472, 1.0472)  # radians

    ACTUATOR_DICT = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_knee': 2,

        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_knee': 5,

        'rear_right_hip': 6,
        'rear_right_thigh': 7,
        'rear_right_knee': 8,

        'rear_left_hip': 9,
        'rear_left_thigh': 10,
        'rear_left_knee': 11,
    }

    SENSOR_POS_DICT = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_knee': 2,

        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_knee': 5,

        'rear_right_hip': 6,
        'rear_right_thigh': 7,
        'rear_right_knee': 8,

        'rear_left_hip': 9,
        'rear_left_thigh': 10,
        'rear_left_knee': 11,
    }

    SENSOR_VEL_DICT = {
        'front_right_hip': 12,
        'front_right_thigh': 13,
        'front_right_knee': 14,

        'front_left_hip': 15,
        'front_left_thigh': 16,
        'front_left_knee': 17,

        'rear_right_hip': 18,
        'rear_right_thigh': 19,
        'rear_right_knee': 20,

        'rear_left_hip': 21,
        'rear_left_thigh': 22,
        'rear_left_knee': 23,
    }

    def __init__(self):
        self.model = None
        self.data = None

        self.current_leg = 'front_right'
        self.current_joint = 'hip'
        self.current_range = self.FRONT_HIP_POS_RANGE

        self.step_size = 0.05  # radians per key press
        
        # Target joint positions for standing configuration (non-interactive joints)
        self.target_positions = {
            # Hip joints (abduction)
            'front_right_hip': 0.0,
            'front_left_hip': 0.0,
            'rear_right_hip': 0.0,
            'rear_left_hip': 0.0,
            
            # Thigh joints
            'front_right_thigh': 0.8,
            'front_left_thigh': 0.8,
            'rear_right_thigh': 1.0,
            'rear_left_thigh': 1.0,
            
            # Calf joints (also named calf)
            'front_right_knee': -1.5,
            'front_left_knee': -1.5,
            'rear_right_knee': -1.5,
            'rear_left_knee': -1.5,
        }
        
        # PD gains
        self.kp = 20.0  # Proportional gain
        self.kd = 2.0   # Derivative gain
        
    def init_controller(self, model, data):
        self.model = model
        self.data = data

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
                self.current_range = self.FRONT_HIP_POS_RANGE if 'front' in self.current_leg else self.BACK_HIP_POS_RANGE
            elif self.current_joint == 'thigh':
                self.current_range = self.ABDUCTION_POS_RANGE 
            elif self.current_joint == 'knee':
                self.current_range = self.KNEE_POS_RANGE

            if key == glfw.KEY_UP:
                # Increase target angle
                self.target_positions[f"{self.current_leg}_{self.current_joint}"] += self.step_size
                self.target_positions[f"{self.current_leg}_{self.current_joint}"] = min(self.target_positions[f"{self.current_leg}_{self.current_joint}"], self.current_range[1])
                
            elif key == glfw.KEY_DOWN:
                # Decrease target angle
                self.target_positions[f"{self.current_leg}_{self.current_joint}"] -= self.step_size
                self.target_positions[f"{self.current_leg}_{self.current_joint}"] = max(self.target_positions[f"{self.current_leg}_{self.current_joint}"], self.current_range[0])
                
            elif key == glfw.KEY_R:
                # Reset to neutral
                self.target_positions[f"{self.current_leg}_{self.current_joint}"] = 0.0

    def run(self, model, data):
        # PD controller to move joints to target positions
        for joint_name, actuator_idx in self.ACTUATOR_DICT.items():
            # Get current position and velocity
            pos_idx = self.SENSOR_POS_DICT[joint_name]
            vel_idx = self.SENSOR_VEL_DICT[joint_name]
            
            current_pos = data.sensordata[pos_idx]
            current_vel = data.sensordata[vel_idx]
            
            # Get target position (use interactive target for FR hip)
 
            target_pos = self.target_positions[joint_name]
            
            # PD control
            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel
            
            control_signal = self.kp * position_error + self.kd * velocity_error
            
            # Apply control (clipping is handled by MuJoCo)
            data.ctrl[actuator_idx] = control_signal

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_terrain.xml')
    sim = MujocoSim(xml_path)
    controller = InteractiveController()
    
    # Optional: Set initial FR hip position programmatically
    # controller.set_fr_hip_target(0.5)  # Set to 0.5 radians
    
    sim.sim(controller=controller, sim_length=-1)  # Run until window closed

if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES:

1. Interactive keyboard control (default):
   - Run the script
   - Use UP/DOWN arrows to control FR hip
   - Press R to reset to neutral
   - Press ESC to exit

2. Programmatic control:
   controller = InteractiveController()
   controller.set_fr_hip_target(0.5)  # Set to specific angle in radians
   
3. Get current target:
   current_target = controller.fr_hip_target
   
4. Adjust step size:
   controller.step_size = 0.1  # Larger steps per key press
"""
