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
        bezier_control_points: dict[Foot, dict] | None = None,
        lift_off_bias: float = 0.3,
        touchdown_bias: float = 0.05,
        transition_blend_width: float = 0.05,
        oval_offsets: dict[OvalOffset, float] | None = None
    ) -> None:
        self.robot_interface = robot_interface

        # Normalise width/height: if None or a bare scalar is passed, expand to a
        # per-foot dict so all internal code can always do self.width[foot] safely.
        _default_width  = {Foot.FL: 0.1,  Foot.FR: 0.1,  Foot.RL: 0.1,  Foot.RR: 0.1}
        _default_height = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05}
        self.width  = width  if isinstance(width,  dict) else (_default_width  if width  is None else {f: float(width)  for f in Foot})
        self.height = height if isinstance(height, dict) else (_default_height if height is None else {f: float(height) for f in Foot})
        self.stance_depth = 0.01

        self.bezier_control_points = bezier_control_points
        self.lift_off_bias = lift_off_bias
        self.touchdown_bias = touchdown_bias
        self.transition_blend_width = transition_blend_width
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

    # def build_oval_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
    #     assert self.oval_offsets is not None, "oval_offsets must be set before calling build_oval_trajectory"

    #     FRONT_FEET = {Foot.FL, Foot.FR}
    #     foot_positions: dict[Foot, Coordinate] = {}
    #     foot_velocities: dict[Foot, Coordinate] = {}

    #     for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
    #         theta     = neuron_output[neuron_idx]
    #         theta_dot = neuron_phase_velocities[neuron_idx]
    #         c_angle   = np.cos(theta)
    #         s_angle   = np.sin(theta)

    #         is_front = foot in FRONT_FEET
    #         if is_front:
    #             x_fore_key = OvalOffset.X_FFORE   # forward reach,  front legs
    #             x_hind_key = OvalOffset.X_FHIND   # rearward reach, front legs
    #             z_top_key  = OvalOffset.Z_FTOP    # swing height,   front legs
    #             z_bot_key  = OvalOffset.Z_FBOTTOM # stance depth,   front legs
    #         else:
    #             x_fore_key = OvalOffset.X_RFORE   # forward reach,  rear legs
    #             x_hind_key = OvalOffset.X_RHIND   # rearward reach, rear legs
    #             z_top_key  = OvalOffset.Z_RTOP    # swing height,   rear legs
    #             z_bot_key  = OvalOffset.Z_RBOTTOM # stance depth,   rear legs

    #         # ── Position ──────────────────────────────────────────────────────
    #         # Convention matches the egg trajectory: x = -A * cos(θ)
    #         #   θ=0   → cos=+1 → x = -A_hind  (foot at rear,  start of swing)
    #         #   θ=π   → cos=-1 → x = +A_fore   (foot at front, end   of swing)
    #         # Key selection: cos ≥ 0 → foot in rear half  → HIND amplitude
    #         #                cos < 0 → foot in front half → FORE amplitude
    #         if c_angle >= 0.0:
    #             x_amp = self.oval_offsets[x_hind_key]  # rearward reach
    #         else:
    #             x_amp = self.oval_offsets[x_fore_key]  # forward reach
    #         x = -x_amp * c_angle   # negated to match egg direction convention

    #         if s_angle > 0.0:
    #             z_amp = self.oval_offsets[z_top_key]
    #         else:
    #             z_amp = self.oval_offsets[z_bot_key]
    #         z = z_amp * s_angle   # upward swing arc / downward stance compression

    #         foot_positions[foot] = Coordinate(x, 0.0, z)
    #         # ── Velocity: d/dt(-x_amp·cos θ) = +x_amp·sin(θ)·θ̇ ─────────────
    #         dx =  x_amp * s_angle * theta_dot
    #         dz =  z_amp * c_angle * theta_dot

    #         foot_velocities[foot] = Coordinate(dx, 0.0, dz)

    #     foot_positions = self.transform_relative_world_to_hip(foot_positions)
    #     return foot_positions, foot_velocities
    
    def build_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Unified entry point — dispatches based on robot_interface.trajectory_method.
        Switch shape at any time: robot_interface.trajectory_method = TrajectoryMethod.BEZIER"""
        method = self.robot_interface.trajectory_method
        if method is TrajectoryMethod.OVAL:
            return self.build_oval_trajectory(neuron_output, neuron_phase_velocities)
        elif method is TrajectoryMethod.BEZIER:
            return self.build_bezier_trajectory(neuron_output, neuron_phase_velocities)
        else:  # TrajectoryMethod.EGG (default)
            return self.build_egg_trajectory(neuron_output, neuron_phase_velocities)
    
    def build_bezier_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        foot_positions: dict[Foot, Coordinate] = {}
        foot_velocities: dict[Foot, Coordinate] = {}

        bw = self.transition_blend_width  # blend half-width in radians

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta = neuron_output[neuron_idx]
            theta_dot = neuron_phase_velocities[neuron_idx]

            # Normalise phase to [0, 2π)
            theta_mod = theta % (2.0 * np.pi)

            swing_pts, stance_pts = self._get_bezier_control_points(foot)

            # Determine which segment we are in and compute normalised t ∈ [0, 1]
            if theta_mod < np.pi:
                t = theta_mod / np.pi
                pts = swing_pts
            else:
                t = (theta_mod - np.pi) / np.pi
                pts = stance_pts

            # ── Position (always from the current segment) ──
            pos_xz = self._cubic_bezier(t, pts[0], pts[1], pts[2], pts[3])
            foot_positions[foot] = Coordinate(pos_xz[0], 0.0, pos_xz[1])

            # ── Velocity with transition blending ──
            # Raw tangent from current segment
            tangent_xz = self._cubic_bezier_derivative(t, pts[0], pts[1], pts[2], pts[3])
            dt_dtheta = 1.0 / np.pi

            if bw > 0.0:
                # Check proximity to transition boundaries (θ_mod ≈ 0 or ≈ π)
                # Near θ=π (swing→stance): blend swing-end tangent with stance-start tangent
                # Near θ=0/2π (stance→swing): blend stance-end tangent with swing-start tangent
                dist_to_pi = abs(theta_mod - np.pi)
                dist_to_zero = min(theta_mod, 2.0 * np.pi - theta_mod)

                if dist_to_pi < bw:
                    # Near the swing→stance boundary
                    alpha = dist_to_pi / bw  # 0 at boundary, 1 at edge of blend zone
                    swing_end_tangent = self._cubic_bezier_derivative(
                        1.0, swing_pts[0], swing_pts[1], swing_pts[2], swing_pts[3])
                    stance_start_tangent = self._cubic_bezier_derivative(
                        0.0, stance_pts[0], stance_pts[1], stance_pts[2], stance_pts[3])
                    # At the boundary itself (alpha=0), use average of both tangents
                    # At the edge of the blend zone (alpha=1), use the pure segment tangent
                    blended = 0.5 * (swing_end_tangent + stance_start_tangent)
                    tangent_xz = alpha * tangent_xz + (1.0 - alpha) * blended

                elif dist_to_zero < bw:
                    # Near the stance→swing boundary
                    alpha = dist_to_zero / bw
                    stance_end_tangent = self._cubic_bezier_derivative(
                        1.0, stance_pts[0], stance_pts[1], stance_pts[2], stance_pts[3])
                    swing_start_tangent = self._cubic_bezier_derivative(
                        0.0, swing_pts[0], swing_pts[1], swing_pts[2], swing_pts[3])
                    blended = 0.5 * (stance_end_tangent + swing_start_tangent)
                    tangent_xz = alpha * tangent_xz + (1.0 - alpha) * blended

            dx = tangent_xz[0] * dt_dtheta * theta_dot
            dz = tangent_xz[1] * dt_dtheta * theta_dot
            foot_velocities[foot] = Coordinate(dx, 0.0, dz)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)
        return foot_positions, foot_velocities

    # ── Bézier helpers ──────────────────────────────────────────
    @staticmethod
    def _cubic_bezier(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
        u = 1.0 - t
        return u*u*u * P0 + 3.0*u*u*t * P1 + 3.0*u*t*t * P2 + t*t*t * P3

    @staticmethod
    def _cubic_bezier_derivative(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
        u = 1.0 - t
        return 3.0*u*u * (P1 - P0) + 6.0*u*t * (P2 - P1) + 3.0*t*t * (P3 - P2)

    def _get_bezier_control_points(self, foot: Foot) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.bezier_control_points is not None and foot in self.bezier_control_points:
            cp = self.bezier_control_points[foot]
            swing  = [np.asarray(p, dtype=float) for p in cp["swing"]]
            stance = [np.asarray(p, dtype=float) for p in cp["stance"]]
            return swing, stance

        w = self.width
        h = self.height[foot]
        d = self.stance_depth
        lo = self.lift_off_bias    # how far forward P1 shifts from P0 (fraction of w)
        td = self.touchdown_bias   # how high P2 stays above ground (fraction of h)

        is_front = foot in (Foot.FL, Foot.FR)
        if is_front:
            p1_x_frac = lo        # 0.8 → P1 at -0.2w (wide forward arc)
            p2_x_frac = 0.2       # P2 well forward
            p2_z_frac = td        # 0.15 → shallow approach angle
        else:
            p1_x_frac = lo * 0.75  # 0.6 → P1 at -0.4w (tighter lift-off)
            p2_x_frac = 0.1        # P2 moderately forward
            p2_z_frac = td * 1.33  # 0.20 → slightly steeper descent

        swing = [
            np.array([-w,                     0.0]),   # P0: rear, ground
            np.array([-w + p1_x_frac * w,     h  ]),   # P1: steep rise
            np.array([ p2_x_frac * w,  p2_z_frac * h]),# P2: gradual descent
            np.array([ w,                     0.0]),   # P3: front, ground
        ]
        stance = [
            np.array([ w,        0.0]),   # P0: front, ground
            np.array([ w * 0.6, -d  ]),   # P1: settle into ground depth
            np.array([-w * 0.6, -d  ]),   # P2: flat ground contact
            np.array([-w,        0.0]),   # P3: rear, ground (lift-off)
        ]
        return swing, stance

    # ── Bézier trajectory ───────────────────────────────────────

