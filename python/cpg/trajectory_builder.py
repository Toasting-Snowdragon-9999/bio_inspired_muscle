from enum import auto
import os, sys
from dataclasses import dataclass

from numpy import angle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *
from shared_module.robot_state import RobotInterface, Hip, Foot, TrajectoryMethod

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

class OvalOffset(Enum):
    X_FFORE = auto()
    X_RFORE = auto()
    X_FHIND = auto()
    X_RHIND = auto()

    Z_FTOP = auto()
    Z_FBOTTOM = auto()
    Z_RTOP = auto()
    Z_RBOTTOM = auto()

class TrajectoryBuilder:
    def __init__(
        self,
        robot_interface: RobotInterface,
        width:  dict[Foot, float] = {Foot.FL: 0.1, Foot.FR: 0.1, Foot.RL: 0.1, Foot.RR: 0.1},
        height: dict[Foot, float] = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05},
        duty_factor: float = 0.5,
        oval_offsets: dict[OvalOffset, float] | None = None
    ) -> None:
        """
        Duty factor is the fraction of stance phase, 0.8 means 80% stance, 20% swing.
        """
        self.robot_interface = robot_interface

        # Normalise width/height: if None or a bare scalar is passed, expand to a
        # per-foot dict so all internal code can always do self.width[foot] safely.
        _default_width  = {Foot.FL: 0.1,  Foot.FR: 0.1,  Foot.RL: 0.1,  Foot.RR: 0.1}
        _default_height = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05}
        self.width  = width  if isinstance(width,  dict) else (_default_width  if width  is None else {f: float(width)  for f in Foot})
        self.height = height if isinstance(height, dict) else (_default_height if height is None else {f: float(height) for f in Foot})
        self.stance_depth = 0.01
        # duty_factor is owned by robot_interface — read it dynamically via
        # self.robot_interface.duty_factor so callers can update it at runtime.
        # The constructor argument is forwarded to the interface if it differs
        # from the interface's current value (backwards-compatibility shim).
        if duty_factor != 0.5:
            self.robot_interface.duty_factor = duty_factor

        self.blend_sharpness = 10.0

        self.oval_offsets = oval_offsets
        # Trajectory method is driven by robot_interface.trajectory_method.
        # Change it at any time: robot_interface.trajectory_method = TrajectoryMethod.OVAL

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

    def apply_duty_factor(self, phase: float) -> float:
        """
        Warp phase according to duty factor.
        Input:  phase in radians [0, 2π)
        Output: warped phase in radians [0, 2π)
        """
        phi = (phase % (2 * np.pi)) / (2 * np.pi)

        duty = self.robot_interface.duty_factor
        if phi < duty:
            gait_phi = 0.5 * (phi / duty)
        else:
            gait_phi = 0.5 + 0.5 * ((phi - duty) / (1 - duty))

        return gait_phi * 2 * np.pi

    def compute_target_velocities(self, neuron_phases: np.ndarray, neuron_phase_velocities: np.ndarray) -> dict[Foot, Coordinate]:
        foot_velocities = {}
        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():

            theta = neuron_phases[neuron_idx]
            theta_dot = neuron_phase_velocities[neuron_idx]

            dx = self.width[foot] * np.sin(theta) * theta_dot

            s = np.sin(theta)
            s_dot = np.cos(theta)

            # Smooth blend between swing (height) and stance (stance_depth) amplitudes
            # to eliminate velocity discontinuity at foot landing (tanh sigmoid: 1 in swing, 0 in stance)
            blend = 0.5 * (1.0 + np.tanh(self.blend_sharpness * s))
            z_amp = blend * self.height[foot] + (1.0 - blend) * self.stance_depth
            dz = z_amp * s_dot * theta_dot

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

    def build_egg_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Convert CPG neuron outputs (phases) to Cartesian foot positions in the hip-local frame.\n
           Make sure the foot_position in RobotInterface is initialized before calling this."""
        foot_positions = {}

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta = neuron_output[neuron_idx]
            theta = self.apply_duty_factor(theta)  # <-- IMPORTANT: apply duty factor to phase before computing trajectory
            y = 0.0  # Keep the current y position unchanged
            x = - self.width[foot] * np.cos(theta)

            s = np.sin(theta)
            # Smooth blend between swing (height) and stance (stance_depth) amplitudes
            # to eliminate velocity discontinuity at foot landing (tanh sigmoid: 1 in swing, 0 in stance)
            blend = 0.5 * (1.0 + np.tanh(self.blend_sharpness * s))
            z_amp = blend * self.height[foot] + (1.0 - blend) * self.stance_depth
            z = z_amp * s

            # z = self.height[foot] * max(0.0, np.sin(theta))
            foot_positions[foot] = Coordinate(x, y, z)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)
        foot_velocities = self.compute_target_velocities(neuron_output, neuron_phase_velocities)
        return foot_positions, foot_velocities

    def build_oval_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        assert self.oval_offsets is not None, "oval_offsets must be set before calling build_oval_trajectory"

        FRONT_FEET = {Foot.FL, Foot.FR}
        foot_positions: dict[Foot, Coordinate] = {}
        foot_velocities: dict[Foot, Coordinate] = {}

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta     = neuron_output[neuron_idx]
            theta = self.apply_duty_factor(theta)  # <-- IMPORTANT: apply duty factor to phase before computing trajectory
            theta_dot = neuron_phase_velocities[neuron_idx]

            c = np.cos(theta)
            s = np.sin(theta)

            is_front = foot in FRONT_FEET
            if is_front:
                x_fore = self.oval_offsets[OvalOffset.X_FFORE]
                x_hind = self.oval_offsets[OvalOffset.X_FHIND]
                z_top  = self.oval_offsets[OvalOffset.Z_FTOP]
                z_bot  = self.oval_offsets[OvalOffset.Z_FBOTTOM]
            else:
                x_fore = self.oval_offsets[OvalOffset.X_RFORE]
                x_hind = self.oval_offsets[OvalOffset.X_RHIND]
                z_top  = self.oval_offsets[OvalOffset.Z_RTOP]
                z_bot  = self.oval_offsets[OvalOffset.Z_RBOTTOM]

            # ── Smooth blends ─────────────────────────────────────────────
            # Front/back transition (based on cos)
            blend_x = 0.5 * (1.0 + np.tanh(self.blend_sharpness * (-c)))
            x_amp = blend_x * x_fore + (1.0 - blend_x) * x_hind

            # Swing/stance transition (based on sin)
            blend_z = 0.5 * (1.0 + np.tanh(self.blend_sharpness * s))
            z_amp = blend_z * z_top + (1.0 - blend_z) * z_bot

            # ── Position ─────────────────────────────────────────────────
            x = -x_amp * c
            z =  z_amp * s

            foot_positions[foot] = Coordinate(x, 0.0, z)

            # ── Derivatives (IMPORTANT: include dA/dθ) ────────────────────
            # tanh'(x) = 1 - tanh²(x)
            sech2_x = 1.0 - np.tanh(self.blend_sharpness * (-c))**2
            sech2_z = 1.0 - np.tanh(self.blend_sharpness * s)**2

            dblend_x_dtheta = 0.5 * self.blend_sharpness * sech2_x * (s)
            dblend_z_dtheta = 0.5 * self.blend_sharpness * sech2_z * (c)

            dx_amp_dtheta = dblend_x_dtheta * (x_fore - x_hind)
            dz_amp_dtheta = dblend_z_dtheta * (z_top  - z_bot)

            # Full derivatives
            dx = (x_amp * s + dx_amp_dtheta * (-c)) * theta_dot
            dz = (z_amp * c + dz_amp_dtheta * s) * theta_dot

            foot_velocities[foot] = Coordinate(dx, 0.0, dz)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)
        return foot_positions, foot_velocities
    
    def build_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Unified entry point — dispatches based on robot_interface.trajectory_method.
        Switch shape at any time: robot_interface.trajectory_method = TrajectoryMethod.BEZIER"""
        method = self.robot_interface.trajectory_method
        if method is TrajectoryMethod.OVAL:
            return self.build_oval_trajectory(neuron_output, neuron_phase_velocities)

        else:  # TrajectoryMethod.EGG (default)
            return self.build_egg_trajectory(neuron_output, neuron_phase_velocities)
