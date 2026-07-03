"""@brief Standalone demo: PD-control a Go2 quadruped to a fixed standing pose on terrain.

Entry-point script that wires a simple position-PD controller (``test_controller``)
into ``MujocoSim`` and runs a rendered simulation.
"""
import os
from mujoco_sim import MujocoSim



class test_controller:
    """@brief Self-contained PD controller that holds the robot in a standing configuration.

    Owns its own actuator/sensor index maps and joint ranges, and drives each
    actuator with a proportional-derivative law toward fixed target joint angles.
    """
    # KNEE_CONTROL_RANGE = {-45.43, 45.43}
    # DEFAULT_CONTROL_RANGE = {-23.7, 23.7}

    KNEE_POS_RANGE      = (-2.7227, -0.83776)  # radians
    FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)  # radians
    BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)  # radians
    ABDUCTION_POS_RANGE = (-1.0472, 1.0472)  # radians

    ACTUATOR_DICT = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_calf': 2,

        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_calf': 5,

        'rear_right_hip': 6,
        'rear_right_thigh': 7,
        'rear_right_calf': 8,

        'rear_left_hip': 9,
        'rear_left_thigh': 10,
        'rear_left_calf': 11,
    }

    SENSOR_POS_DICT = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_calf': 2,

        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_calf': 5,

        'rear_right_hip': 6,
        'rear_right_thigh': 7,
        'rear_right_calf': 8,

        'rear_left_hip': 9,
        'rear_left_thigh': 10,
        'rear_left_calf': 11,
    }

    SENSOR_VEL_DICT = {
        'front_right_hip': 12,
        'front_right_thigh': 13,
        'front_right_calf': 14,

        'front_left_hip': 15,
        'front_left_thigh': 16,
        'front_left_calf': 17,

        'rear_right_hip': 18,
        'rear_right_thigh': 19,
        'rear_right_calf': 20,

        'rear_left_hip': 21,
        'rear_left_thigh': 22,
        'rear_left_calf': 23,
    }

    def __init__(self):
        """@brief Initialise model/data handles, standing target pose, and PD gains."""
        self.model = None
        self.data = None

        # Target joint positions for standing configuration
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
            
            # Calf joints (knee)
            'front_right_calf': -1.5,
            'front_left_calf': -1.5,
            'rear_right_calf': -1.5,
            'rear_left_calf': -1.5,
        }
        
        # PD gains
        self.kp = 20.0  # Proportional gain
        self.kd = 2.0   # Derivative gain

    def init_controller(self, model, data):
        """@brief Store the MuJoCo model and data handles for later use.

        @param model: MuJoCo MjModel for the loaded scene.
        @param data: MuJoCo MjData state container associated with the model.
        """
        self.model = model
        self.data = data

    def run(self, model, data):
        """@brief Apply one PD-control step driving every joint toward its standing target.

        @param model: MuJoCo MjModel (unused directly; index maps are class attributes).
        @param data: MuJoCo MjData providing sensor readings and the ctrl array to write.
        """
        # PD controller to move joints to target standing position
        for joint_name, actuator_idx in self.ACTUATOR_DICT.items():
            # Get current position and velocity
            pos_idx = self.SENSOR_POS_DICT[joint_name]
            vel_idx = self.SENSOR_VEL_DICT[joint_name]
            
            current_pos = data.sensordata[pos_idx]
            current_vel = data.sensordata[vel_idx]
            
            # Get target position
            target_pos = self.target_positions[joint_name]
            
            # PD control
            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel
            
            control_signal = self.kp * position_error + self.kd * velocity_error
            
            # Apply control (clipping is handled by MuJoCo)
            data.ctrl[actuator_idx] = control_signal

def main():
    """@brief Load the terrain scene, build the PD controller, and run the rendered sim."""
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_terrain.xml')
    sim = MujocoSim(xml_path)
    controller = test_controller()
    sim.sim(controller=controller, sim_length=100)  # Run simulation for 10 seconds

if __name__ == "__main__":
    main()
