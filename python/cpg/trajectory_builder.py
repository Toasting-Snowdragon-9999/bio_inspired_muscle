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
    def __init__(
        self,
        robot_interface: RobotInterface,
        width: float = 0.1,
        height: dict[Foot, float] = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05},
        bezier_control_points: dict[Foot, dict] | None = None,
        lift_off_bias: float = 0.3,
        touchdown_bias: float = 0.05,
        transition_blend_width: float = 0.05,
    ) -> None:
        self.robot_interface = robot_interface
        self.width = width
        self.height = height
        self.stance_depth = 0.01
        # Optional per-foot Bézier control point override.
        # When None, control points are derived from width/height/stance_depth.
        # Expected format: {Foot: {"swing": [P0,P1,P2,P3], "stance": [P0,P1,P2,P3]}}
        # where each Pi is an (x, z) tuple or array.
        self.bezier_control_points = bezier_control_points
        # ── Bio-realistic Bézier tuning parameters ──
        # lift_off_bias: controls how far forward P1 shifts from P0 during lift-off.
        #   Higher = more forward travel during initial rise (range: 0.3–1.0).
        #   Expressed as fraction of width: P1.x = -w + lift_off_bias * w
        self.lift_off_bias = lift_off_bias
        # touchdown_bias: controls how close to ground P2 is during swing descent.
        #   Lower = shallower touchdown angle, less vertical velocity at impact.
        #   Expressed as fraction of step height: P2.z = touchdown_bias * h
        self.touchdown_bias = touchdown_bias
        # transition_blend_width: phase window (radians) for velocity blending at
        #   swing↔stance transitions.  Smooths the unavoidable velocity reversal
        #   over a short window to reduce torque spikes in the impedance controller.
        #   Set to 0.0 to disable blending.  Default ≈ 3° (0.05 rad).
        self.transition_blend_width = transition_blend_width
        # Sharpness of the tanh sigmoid used to smoothly blend between swing and
        # stance Z-amplitudes.  Higher values approach a hard switch; 10.0 gives
        # a smooth but fairly quick transition that eliminates the velocity
        # discontinuity at foot landing.
        self.blend_sharpness = 10.0
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
            x = - self.width * np.cos(theta)

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
    
    def build_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Default trajectory method — delegates to the active trajectory builder."""
        return self.build_bezier_trajectory(neuron_output, neuron_phase_velocities)
    
    def build_bezier_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Bézier-curve closed-loop foot trajectory.
        
        The gait cycle is split into two cubic Bézier segments:
          - Swing  (θ mod 2π ∈ [0, π]):  foot lifts and moves forward
          - Stance (θ mod 2π ∈ [π, 2π]): foot pushes backward on ground
        
        Control points default to a bio-realistic asymmetric D-shaped loop
        (see _get_bezier_control_points for details).
        
        Velocities are computed analytically via:
          dpos/dt = B'(t) · (1/π) · dθ/dt
        
        At swing↔stance transitions the foot reverses horizontal direction,
        making true C1 continuity impossible.  Instead, a small blending
        window (self.transition_blend_width) linearly interpolates between
        the outgoing and incoming tangent vectors to reduce the velocity
        discontinuity seen by the impedance controller."""
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

    def build_cycloidal_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        pass

    # ── Bézier helpers ──────────────────────────────────────────

    @staticmethod
    def _cubic_bezier(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
        """Evaluate a cubic Bézier curve at parameter t ∈ [0, 1].
        B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3"""
        u = 1.0 - t
        return u*u*u * P0 + 3.0*u*u*t * P1 + 3.0*u*t*t * P2 + t*t*t * P3

    @staticmethod
    def _cubic_bezier_derivative(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
        """Derivative of a cubic Bézier w.r.t. t (tangent vector).
        B'(t) = 3(1-t)²·(P1-P0) + 6(1-t)t·(P2-P1) + 3t²·(P3-P2)"""
        u = 1.0 - t
        return 3.0*u*u * (P1 - P0) + 6.0*u*t * (P2 - P1) + 3.0*t*t * (P3 - P2)

    def _get_bezier_control_points(self, foot: Foot) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (swing_points, stance_points) for the given foot.

        If self.bezier_control_points is set for this foot, use those directly.
        Otherwise derive bio-realistic asymmetric control points from
        width / height / stance_depth / lift_off_bias / touchdown_bias.

        The default curve produces a D-shaped closed loop modelled after
        measured trotting-dog foot trajectories:
          - Sharp, steep lift-off at the rear of the stride
          - Swing apex shifted rearward (forward-biased swing)
          - Gradual, shallow touchdown at the front of the stride
          - Flat stance with a slight compliance dip

        Front legs use a wider lift-off arc and lower touchdown approach
        (higher clearance, gentler landing).  Rear legs use a tighter,
        more vertical lift-off (stronger push-off).

        Control-point layout (swing, front legs):
        ```
             P1 ●              steep lift-off
                 ╲
                  ╲   apex region (rearward-biased)
                   ╲
                    ╲─────────● P2   gradual descent
                               ╲
        P0 ●                     ● P3   touchdown
        (-w, 0)               (w, 0)
        ```
        """
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

        # ── Swing: rear (-w) → front (+w), airborne ──
        #
        # P0: lift-off at the rear of the stride, ground level.
        # P1: steep rise — mostly vertical with some forward component.
        #     Front legs: P1.x = -w + 0.8w = -0.2w  (wide forward arc)
        #     Rear legs:  P1.x = -w + 0.6w = -0.4w  (tighter, more vertical push)
        #     P1.z = h  (full step height for both)
        # P2: pre-touchdown approach — well forward and close to ground.
        #     Front legs: P2.x = 0.6w, P2.z = 0.15h  (very shallow approach)
        #     Rear legs:  P2.x = 0.5w, P2.z = 0.20h  (slightly steeper)
        # P3: touchdown at the front of the stride, ground level.
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

        # ── Stance: front (+w) → rear (-w), on ground ──
        #
        # P0: just after touchdown, ground level.
        # P1: quick settle into the ground — compliance dip.
        #     P1.x = 0.4w (still forward), P1.z = -d (at ground depth)
        # P2: continued backward travel at ground depth.
        #     P2.x = -0.4w (rearward), P2.z = -d
        # P3: end of stance at rear, back to ground level for lift-off.
        #
        # The long flat region between P1 and P2 at depth -d mimics the
        # foot being planted while the body passes over it.
        stance = [
            np.array([ w,        0.0]),   # P0: front, ground
            np.array([ w * 0.6, -d  ]),   # P1: settle into ground depth
            np.array([-w * 0.6, -d  ]),   # P2: flat ground contact
            np.array([-w,        0.0]),   # P3: rear, ground (lift-off)
        ]
        return swing, stance

    # ── Bézier trajectory ───────────────────────────────────────

