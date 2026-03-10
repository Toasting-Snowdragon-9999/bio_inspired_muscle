import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *
from shared_module.robot_state import RobotInterface, Hip, Foot

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

class TrajectoryBuilder:
    def __init__(self, robot_interface: RobotInterface, width: float = 0.1, height: float = 0.05) -> None:
        self.robot_interface = robot_interface
        self.width = width
        self.height = height
        # Stance positions in the Hip frame
        self.stance_positions_hipf = {
            Foot.FL: Coordinate( 0.01359,  0.10217, -0.26305),
            Foot.FR: Coordinate( 0.01359, -0.10217, -0.26305),
            Foot.RL: Coordinate(-0.04041,  0.09929, -0.26944),
            Foot.RR: Coordinate(-0.04041, -0.09929, -0.26944),
        }

    def transform_to_world(self, target_foot_pos: dict[Foot, Coordinate]) -> dict[Foot, Coordinate]:
        pass

    def transform_world_to_hip(self, target_world_foot_pos: dict[Foot, Coordinate]) -> dict[Foot, Coordinate]:
        foot_positions_hip = {}

        for foot, pos in target_world_foot_pos.items():
            hip_world = self.robot_interface.hip_position[Hip[foot.name]]

            foot_positions_hip[foot] = Coordinate(
                x = pos.x - hip_world[0],
                y = pos.y - hip_world[1],
                z = pos.z - hip_world[2],
            )

        return foot_positions_hip

    def transform_relative_world_to_hip(self, foot_offset: dict[Foot, Coordinate]) -> dict[Foot, Coordinate]:
        """Convert foot positions from world coordinates to hip-centered coordinates."""
        foot_positions_hip = {}
        for foot, pos in foot_offset.items():
            stance = self.stance_positions_hipf[foot]
            foot_positions_hip[foot] = Coordinate(
            x = stance.x + pos.x,
            y = stance.y + pos.y,
            z = stance.z + pos.z
        )
        return foot_positions_hip

    def build_trajectory(self, neuron_output) -> Coordinate:
        """Convert CPG neuron outputs (phases) to Cartesian foot positions in the hip-local frame.\n
           Make sure the foot_position in RobotInterface is initialized before calling this."""
        foot_positions = {}

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta = neuron_output[neuron_idx]

            y = 0.0  # Keep the current y position unchanged
            x = - self.width * np.cos(theta)
            z = self.height * max(0.0, np.sin(theta))
            foot_positions[foot] = Coordinate(x, y, z)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)

        return foot_positions
