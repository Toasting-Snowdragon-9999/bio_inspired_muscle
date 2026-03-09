import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *
from shared_module.robot_state import RobotInterface

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

class TrajectoryBuilder:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface
        self.width = 0.2
        self.height = 0.1
        self.z_value_ground_offset = 0.0  # Can be tuned to make ground contact

    def build_trajectory(self, neuron_output) -> Coordinate:
        """Convert CPG neuron outputs (phases) to Cartesian foot positions in the hip-local frame.\n
           Make sure the foot_position in RobotInterface is initialized before calling this."""
        foot_positions = {}
        current_position = self.robot_interface.foot_positions

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            pos = current_position[foot]
            theta = neuron_output[neuron_idx]

            x0 = pos[0]
            y0 = pos[1]
            z0 = pos[2]

            y = y0  # Keep the current y position unchanged
            x = x0 - self.width * np.cos(theta)
            z = z0 + self.height * max(0.0, np.sin(theta)) + self.z_value_ground_offset
            foot_positions[foot] = Coordinate(x, y, z)

        return foot_positions
