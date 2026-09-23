"""
@brief Foot-trajectory generation from CPG phases for a quadruped.

Maps Kuramoto CPG oscillator phases to Cartesian foot targets and velocities.
Provides the ellipsoid trajectory shape (EllipsoidConfig / TrajectoryBuilder),
the duty-factor stance/swing split (GaitScheduler), and CPG blending toward the
default standing pose.
"""

from enum import Enum, auto
import os, sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *
from shared_module.robot_state import RobotInterface, Hip, Foot, TrajectoryMethod

@dataclass
class Coordinate:
    """
    @brief Cartesian (x, y, z) coordinate or vector.

    @param x: X component (forward / back, metres).
    @param y: Y component (lateral, metres).
    @param z: Z component (height, metres).
    """
    x: float
    y: float
    z: float

class OvalOffset(Enum):
    """
    @brief Named offset keys for tuning the oval foot trajectory.

    Identifies the per-region x and z offsets (fore/rear and top/bottom) used
    to shape the oval trajectory for front and rear legs.
    """
    X_FFORE = auto()
    X_RFORE = auto()
    X_FHIND = auto()
    X_RHIND = auto()

    Z_FTOP = auto()
    Z_FBOTTOM = auto()
    Z_RTOP = auto()
    Z_RBOTTOM = auto()

@dataclass
class EllipsoidConfig:
    """Ellipsoid trajectory shape with independent parameters for front and rear legs.

    The foot traces a smooth, asymmetric ellipse whose x- and z-amplitudes
    blend continuously via tanh between forward/rearward and top/bottom values.
    Rotation and skew are applied afterwards as a post-process.

    Front legs (FL, FR) use the `front_*` fields.
    Rear  legs (RL, RR) use the `rear_*`  fields.

    Attributes:
        front_x_fore   / rear_x_fore   : forward reach from stance (m).
        front_x_hind   / rear_x_hind   : rearward reach from stance (m).
        front_z_top    / rear_z_top    : swing height above stance (m).
        front_z_bottom / rear_z_bottom : stance depth below stance (m) — compliance.
        front_rotation / rear_rotation : ellipse rotation about its centre (radians).
                                         Positive tilts the swing arc forward.
        front_skew     / rear_skew     : constant x-displacement (m) applied after rotation.
                                         Shifts the entire ellipse forward (positive)
                                         or backward (negative) along the x axis.
    """
    # ── Front legs (FL, FR) ───────────────────────────────────────────────────
    front_x_fore:   float = 0.08
    front_x_hind:   float = 0.08
    front_z_top:    float = 0.06
    front_z_bottom: float = 0.01
    front_rotation: float = 0.0
    front_skew:     float = 0.0
    # ── Rear legs (RL, RR) ────────────────────────────────────────────────────
    rear_x_fore:   float = 0.08
    rear_x_hind:   float = 0.08
    rear_z_top:    float = 0.06
    rear_z_bottom: float = 0.01
    rear_rotation: float = 0.0
    rear_skew:     float = 0.0

    def keys(self):
        """
        @brief Keys.
        @return
        """
        return asdict(self).keys()

    def values(self):
        """
        @brief Values.
        @return
        """
        return asdict(self).values()

    def items(self):
        """
        @brief Items.
        @return
        """
        return asdict(self).items()

    def __getitem__(self, key):
        """
        @brief Getitem.
        @param key:
        @return
        """
        return getattr(self, key)

class GaitScheduler:
    """@brief GaitScheduler — gait scheduler."""
    def __init__(self, robot_interface):
        """
        @brief Construct a GaitScheduler instance.
        @param robot_interface:
        """
        self.robot_interface = robot_interface

    def compute(self, phases):
        """
        @brief Compute.
        @param phases:
        @return
        """
        duty = self.robot_interface.duty_factor
        contact = {}
        phase_norm = {}

        for i, foot in enumerate(NEURON_TO_FOOT_DICT.values()):
            phi = (phases[i] % (2*np.pi)) / (2*np.pi)

            phase_norm[foot] = phi
            contact[foot] = 0 if phi < duty else 1

        return phase_norm, contact

class TrajectoryBuilder:
    """@brief TrajectoryBuilder — trajectory builder."""
    def __init__(
        self,
        robot_interface: RobotInterface,
        width:  dict[Foot, float] = {Foot.FL: 0.1, Foot.FR: 0.1, Foot.RL: 0.1, Foot.RR: 0.1},
        height: dict[Foot, float] = {Foot.FL: 0.05, Foot.FR: 0.05, Foot.RL: 0.05, Foot.RR: 0.05},
        duty_factor: float = 0.5,
        oval_offsets: dict[OvalOffset, float] | None = None,
        ellipsoid_config: 'EllipsoidConfig | None' = None
    ) -> None:
        """
        Duty factor is the fraction of stance phase, 0.8 means 80% stance, 20% swing.
        """
        self.robot_interface = robot_interface
        self.robot_interface.current_traj_params = ellipsoid_config
        # Seed `active_traj_params` from the same config. The trajectory
        # builder's read path (new_build_ellipsoid_trajectory and the
        # blended ellipsoid variant) indexes active_traj_params, which is
        # otherwise None until the fuzzy controller completes its first
        # transition — that would crash mjcb_control on the very first
        # step. The fuzzy blender overwrites this during/after a transition.
        self.robot_interface.active_traj_params = ellipsoid_config
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
        # EllipsoidConfig for the ELLIPSOID trajectory method.
        # Front (FL, FR) and rear (RL, RR) legs each have independent parameters
        # via the front_* / rear_* fields of EllipsoidConfig.
        self.ellipsoid_config = ellipsoid_config
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
        """
        @brief Compute target velocities.
        @param neuron_phases:
        @param neuron_phase_velocities:
        @return
        """
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
        """
        @brief Transform world to hip.
        @param target_world_foot_pos:
        @return
        """
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

    def build_ellipsoid_trajectory(self, neuron_output, neuron_phase_velocities) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """
        Asymmetric ellipsoid foot trajectory with independent front/rear parameters,
        rotation, skew, and duty-factor support.

        Pipeline per foot:
          1. Select front_* or rear_* fields from EllipsoidConfig based on leg group.
          2. Apply duty-factor warp to θ  →  θ_w  (stretches stance, compresses swing)
          3. Blend x-amplitude between x_fore / x_hind via tanh on cos(θ_w)
          4. Blend z-amplitude between z_top  / z_bottom via tanh on sin(θ_w)
          5. Build base ellipse:  x_e = -x_amp·cos(θ_w),  z_e = z_amp·sin(θ_w)
          6. Rotate by `rotation` radians about the ellipse centre
          7. Skew along x:  x_s = x_r + skew  (constant x-displacement, m)
          8. Velocities are derived fully analytically (chain rule through all steps)

        Use cfg.front_* fields for FL/FR and cfg.rear_* fields for RL/RR.
        """
        assert self.ellipsoid_config is not None, (
            "ellipsoid_config must be set on TrajectoryBuilder before using ELLIPSOID method."
        )

        FRONT_FEET = {Foot.FL, Foot.FR}
        cfg: EllipsoidConfig = self.robot_interface.active_traj_params
        blend = self.blend_sharpness   # tanh sharpness (shared with egg/oval)
        foot_positions:  dict[Foot, Coordinate] = {}
        foot_velocities: dict[Foot, Coordinate] = {}

        for neuron_idx, foot in NEURON_TO_FOOT_DICT.items():
            theta     = neuron_output[neuron_idx]
            theta_dot = neuron_phase_velocities[neuron_idx]

            # Select front or rear parameters based on leg group
            is_front = foot in FRONT_FEET
            x_fore   = cfg.front_x_fore   if is_front else cfg.rear_x_fore
            x_hind   = cfg.front_x_hind   if is_front else cfg.rear_x_hind
            z_top    = cfg.front_z_top    if is_front else cfg.rear_z_top
            z_bottom = cfg.front_z_bottom if is_front else cfg.rear_z_bottom
            rotation = cfg.front_rotation if is_front else cfg.rear_rotation
            skew     = cfg.front_skew     if is_front else cfg.rear_skew

            # ── 1. Duty-factor warp ───────────────────────────────────────
            # The warp derivative d(θ_w)/d(θ) scales the velocity output so
            # the foot moves at the correct Cartesian speed during stance/swing.
            duty    = self.robot_interface.duty_factor
            phi     = (theta % (2 * np.pi)) / (2 * np.pi)
            theta_w = self.apply_duty_factor(theta)
            # Piecewise-linear warp derivative
            d_tw_dtheta = (0.5 / duty) if phi < duty else (0.5 / (1.0 - duty))

            c = np.cos(theta_w)
            s = np.sin(theta_w)

            # ── 2 & 3. Blended amplitudes ────────────────────────────────
            # x: fore when cos < 0 (foot moving forward), hind when cos > 0
            blend_x = 0.5 * (1.0 + np.tanh(blend * (-c)))
            x_amp   = blend_x * x_fore + (1.0 - blend_x) * x_hind

            # z: top when sin > 0 (swing), bottom when sin < 0 (stance)
            blend_z = 0.5 * (1.0 + np.tanh(blend * s))
            z_amp   = blend_z * z_top + (1.0 - blend_z) * z_bottom

            # ── 4. Base ellipse position ──────────────────────────────────
            x_e = -x_amp * c
            z_e =  z_amp * s

            # ── 5. Rotation ───────────────────────────────────────────────
            ca, sa = np.cos(rotation), np.sin(rotation)
            x_r =  x_e * ca - z_e * sa
            z_r =  x_e * sa + z_e * ca

            # ── 6. Skew (constant x-displacement, m) ──────────────────────
            # Shifts the whole ellipse forward/backward; z is unchanged.
            x_s = x_r + skew
            z_s = z_r

            foot_positions[foot] = Coordinate(x_s, 0.0, z_s)

            # ── 7. Analytic velocities (chain rule) ───────────────────────
            # All partial derivatives are with respect to θ_w, then scaled
            # by d(θ_w)/d(θ) · dθ/dt.
            sech2_x = 1.0 - np.tanh(blend * (-c)) ** 2
            sech2_z = 1.0 - np.tanh(blend * s)    ** 2

            # d(blend_x)/d(θ_w) = 0.5 · blend · sech²(-c) · sin
            dblend_x = 0.5 * blend * sech2_x * s
            # d(blend_z)/d(θ_w) = 0.5 · blend · sech²(s) · cos
            dblend_z = 0.5 * blend * sech2_z * c

            dx_amp = dblend_x * (x_fore - x_hind)   # d(x_amp)/d(θ_w)
            dz_amp = dblend_z * (z_top  - z_bottom) # d(z_amp)/d(θ_w)

            # d(x_e)/d(θ_w) = x_amp·sin + dx_amp·(-cos)
            dx_e = x_amp * s  + dx_amp * (-c)
            # d(z_e)/d(θ_w) = z_amp·cos + dz_amp·sin
            dz_e = z_amp * c  + dz_amp * s

            # Rotate derivatives
            dx_r = dx_e * ca - dz_e * sa
            dz_r = dx_e * sa + dz_e * ca

            # Skew is a constant offset — no contribution to velocity
            dx_s = dx_r
            dz_s = dz_r

            # Final velocity: d(pos)/d(θ_w) · d(θ_w)/d(θ) · dθ/dt
            vx = dx_s * d_tw_dtheta * theta_dot
            vz = dz_s * d_tw_dtheta * theta_dot

            foot_velocities[foot] = Coordinate(vx, 0.0, vz)

        foot_positions = self.transform_relative_world_to_hip(foot_positions)
        return foot_positions, foot_velocities


    def build_trajectory(
        self,
        neuron_phase: dict[Foot, float],
        neuron_phase_velocities: np.ndarray,
    ) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Unified entry point — dispatches based on robot_interface.trajectory_method.
        Switch shape at any time: robot_interface.trajectory_method = TrajectoryMethod.BEZIER

        Inputs are produced by GaitScheduler.compute(phase_outputs):
          - phase_norm[foot]: normalised phase ∈ [0, 1)
          - contact[foot]:    1 if foot is in stance (phi < duty), else 0
        """
        method = self.robot_interface.trajectory_method

        if method is not TrajectoryMethod.ELLIPSOID:
            raise NotImplementedError(
                f"Trajectory method {method.name} is not implemented; only "
                f"TrajectoryMethod.ELLIPSOID is. Construct the RobotInterface with "
                f"trajectory_method=TrajectoryMethod.ELLIPSOID (the default), or set "
                f"robot_interface.trajectory_method = TrajectoryMethod.ELLIPSOID."
            )

        foot_positions, foot_velocities = self.build_ellipsoid_trajectory(
            neuron_phase, neuron_phase_velocities
        )
        return self.apply_cpg_blending(foot_positions, foot_velocities)

    def apply_cpg_blending(
            self,
            foot_positions: dict[Foot, Coordinate],
            foot_velocities: dict[Foot, Coordinate]
        ) -> tuple[dict[Foot, Coordinate], dict[Foot, Coordinate]]:
        """Blend between CPG trajectory and stance position based on robot_interface.cpg_alpha.
           When cpg is disabled the robot will fall back to default standing pose (stance_positions). """
        alpha = self.robot_interface.cpg_alpha

        blended_pos = {}
        blended_vel = {}

        for foot in foot_positions:
            pos = foot_positions[foot]
            vel = foot_velocities[foot]
            stance = self.stance_positions[foot]

            blended_pos[foot] = Coordinate(
                x = alpha * pos.x + (1.0 - alpha) * stance.x,
                y = alpha * pos.y + (1.0 - alpha) * stance.y,
                z = alpha * pos.z + (1.0 - alpha) * stance.z,
            )

            blended_vel[foot] = Coordinate(
                x = alpha * vel.x,
                y = alpha * vel.y,
                z = alpha * vel.z,
            )

            # Optional: hard clamp near zero (prevents jitter)
            if alpha < 1e-3:
                blended_vel[foot] = Coordinate(0.0, 0.0, 0.0)

        return blended_pos, blended_vel