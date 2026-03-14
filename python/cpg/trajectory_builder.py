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
    def __init__(self, robot_interface: RobotInterface, width: float = 0.1, height: dict[Foot, float] = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05}) -> None:
        self.robot_interface = robot_interface
        self.width = width
        self.height = height
        self.stance_depth = 0.02
        # Stance positions in the Hip frame
        self.stance_positions_hipf = {
            Foot.FL: Coordinate( 0.01359,  0.10217, -0.26305),
            Foot.FR: Coordinate( 0.01359, -0.10217, -0.26305),
            Foot.RL: Coordinate(-0.04041,  0.09929, -0.26944),
            Foot.RR: Coordinate(-0.04041, -0.09929, -0.26944),
        }
        self.stance_positions = {
            Foot.FL: Coordinate(-0.01458281454414101, 0.09520361349694849, -0.31155118257490244),
            Foot.FR: Coordinate(-0.014582808367942734, -0.0952034741060266, -0.311551225724358),
            Foot.RL: Coordinate(-0.07616549739558984, 0.0955063386302885, -0.3023549187505239),
            Foot.RR: Coordinate(-0.076165490580238, -0.09550685969947971, -0.30235475609234186),
        }

    def compute_target_velocities(self, neuron_phases: np.ndarray, neuron_phase_velocities: np.ndarray) -> dict[Foot, Coordinate]:
        foot_velocities = {}
        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():

            theta = neuron_phases[neuron_idx]
            theta_dot = neuron_phase_velocities[neuron_idx]

            dx = self.width * np.sin(theta) * theta_dot

            s = np.sin(theta)
            s_dot = np.cos(theta)

            if s >= 0:
                dz = self.height[foot] * s_dot * theta_dot
            else:
                dz = self.stance_depth * s_dot * theta_dot

            dy = 0.0

            foot_velocities[foot] = Coordinate(dx, dy, dz)

        return foot_velocities
        

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
            stance = self.stance_positions[foot]
            foot_positions_hip[foot] = Coordinate(
            x = stance.x + pos.x,
            y = stance.y + pos.y,
            z = stance.z + pos.z
        )
        return foot_positions_hip

    def build_trajectory(self, neuron_output, neuron_phase_velocities) -> Coordinate:
        """Convert CPG neuron outputs (phases) to Cartesian foot positions in the hip-local frame.\n
           Make sure the foot_position in RobotInterface is initialized before calling this."""
        foot_positions = {}

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta = neuron_output[neuron_idx]

            y = 0.0  # Keep the current y position unchanged
            x = - self.width * np.cos(theta)

            s = np.sin(theta)
            if s >= 0:
                z = self.height[foot] * s
            else:
                z = self.stance_depth * s

            # z = self.height[foot] * max(0.0, np.sin(theta))
            foot_positions[foot] = Coordinate(x, y, z)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)
        foot_velocities = self.compute_target_velocities(neuron_output, neuron_phase_velocities)
        return foot_positions, foot_velocities
