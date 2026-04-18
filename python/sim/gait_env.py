"""
Gymnasium environment for RL-based gait parameter optimization.

Each episode is a single headless simulation run. The agent selects
14 gait parameters (normalized to [-1, 1]), the environment runs
the simulation, and returns a multi-component reward based on
Cost of Transport, forward velocity, and body stability.
"""
import os
import sys
import io
import contextlib
from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Cross-module imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig
from controllers.ik_controller import IKController
from shared_module.robot_state import (
    Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod,
)


# ── Parameter names ──
SHAPE_KEYS: list[str] = [
    "front_x_fore", "front_x_hind", "front_z_top", "front_z_bottom",
    "front_rotation", "front_skew",
    "rear_x_fore", "rear_x_hind", "rear_z_top", "rear_z_bottom",
    "rear_rotation", "rear_skew",
]
ALL_PARAM_KEYS: list[str] = ["freq", "duty_factor"] + SHAPE_KEYS


@dataclass
class RewardWeights:
    """Weights for the multi-component reward function."""
    w_cot: float = 1.0       # Cost of Transport penalty (lower COT = better)
    w_vel: float = 0.5       # Forward velocity bonus
    w_tilt: float = 2.0      # Body tilt penalty (stability)
    w_fall: float = 10.0     # Penalty for falling over


@dataclass
class ParamBounds:
    """Lower and upper bounds for each parameter."""
    low: np.ndarray    # shape (14,)
    high: np.ndarray   # shape (14,)


def default_params(gait: Gait) -> dict[str, float]:
    """
    Default parameter values for the given gait.
    Includes freq and duty_factor (unlike ai_fix_param which excluded duty_factor).
    """
    # Gait-specific defaults — same reference values as ai_fix_param.py
    defaults = {
        Gait.WALK: {
            "freq": 1.40, "duty_factor": 0.24,
            "front_x_fore": 0.14, "front_x_hind": 0.08,
            "front_z_top": 0.10, "front_z_bottom": 0.02,
            "front_rotation": 0.05, "front_skew": 0.00,
            "rear_x_fore": 0.06, "rear_x_hind": 0.14,
            "rear_z_top": 0.10, "rear_z_bottom": 0.02,
            "rear_rotation": -0.05, "rear_skew": 0.00,
        },
        Gait.TROT: {
            "freq": 1.95, "duty_factor": 0.50,
            "front_x_fore": 0.123, "front_x_hind": 0.095,
            "front_z_top": 0.104, "front_z_bottom": 0.025,
            "front_rotation": 0.046, "front_skew": 0.035,
            "rear_x_fore": 0.075, "rear_x_hind": 0.122,
            "rear_z_top": 0.095, "rear_z_bottom": 0.005,
            "rear_rotation": -0.060, "rear_skew": -0.015,
        },
        Gait.AMBLE: {
            "freq": 1.675, "duty_factor": 0.40,
            "front_x_fore": 0.14, "front_x_hind": 0.08,
            "front_z_top": 0.10, "front_z_bottom": 0.02,
            "front_rotation": 0.05, "front_skew": 0.00,
            "rear_x_fore": 0.06, "rear_x_hind": 0.14,
            "rear_z_top": 0.10, "rear_z_bottom": 0.02,
            "rear_rotation": -0.05, "rear_skew": 0.00,
        },
        Gait.CANTER: {
            "freq": 3.50, "duty_factor": 0.31,
            "front_x_fore": 0.14, "front_x_hind": 0.07,
            "front_z_top": 0.12, "front_z_bottom": 0.02,
            "front_rotation": 0.10, "front_skew": 0.02,
            "rear_x_fore": 0.06, "rear_x_hind": 0.15,
            "rear_z_top": 0.10, "rear_z_bottom": 0.03,
            "rear_rotation": -0.05, "rear_skew": -0.03,
        },
        Gait.GALLOP: {
            "freq": 3.50, "duty_factor": 0.65,
            "front_x_fore": 0.14, "front_x_hind": 0.06,
            "front_z_top": 0.14, "front_z_bottom": 0.02,
            "front_rotation": 0.00, "front_skew": 0.00,
            "rear_x_fore": 0.06, "rear_x_hind": 0.16,
            "rear_z_top": 0.14, "rear_z_bottom": 0.02,
            "rear_rotation": 0.00, "rear_skew": 0.00,
        },
    }
    return defaults.get(gait, defaults[Gait.WALK])


def make_bounds(gait: Gait, margin: float = 0.20, min_half_range: float = 0.02) -> ParamBounds:
    """
    Build parameter bounds: ±margin around defaults, with absolute bounds for freq/duty.

    Args:
        gait: Gait type to get default parameter values
        margin: Fractional margin around defaults (0.20 = ±20%)
        min_half_range: Minimum half-range for near-zero parameters

    Returns:
        ParamBounds with low and high arrays of shape (14,)
    """
    ref = default_params(gait)
    low = np.zeros(len(ALL_PARAM_KEYS))
    high = np.zeros(len(ALL_PARAM_KEYS))

    for i, key in enumerate(ALL_PARAM_KEYS):
        val = ref[key]
        if key == "freq":
            # Per-gait frequency bounds to keep values realistic
            freq_bounds = {
                Gait.WALK:   (0.8, 1.8),
                Gait.TROT:   (1.2, 2.5),
                Gait.AMBLE:  (0.8, 2.0),
                Gait.CANTER: (1.6, 4.0),
                Gait.GALLOP: (1.6, 4.0),
            }
            low[i], high[i] = freq_bounds.get(gait, (0.5, 4.0))
        elif key == "duty_factor":
            # Absolute bounds for duty factor
            low[i] = 0.2
            high[i] = 0.8
        else:
            # ±margin% around default, with minimum half-range
            half = max(abs(val) * margin, min_half_range)
            low[i] = val - half
            high[i] = val + half

    return ParamBounds(low=low, high=high)


def _dict_to_cfg(d: dict) -> EllipsoidConfig:
    """Build an EllipsoidConfig from a param dict."""
    return EllipsoidConfig(
        front_x_fore=d["front_x_fore"],
        front_x_hind=d["front_x_hind"],
        front_z_top=d["front_z_top"],
        front_z_bottom=d["front_z_bottom"],
        front_rotation=d["front_rotation"],
        front_skew=d["front_skew"],
        rear_x_fore=d["rear_x_fore"],
        rear_x_hind=d["rear_x_hind"],
        rear_z_top=d["rear_z_top"],
        rear_z_bottom=d["rear_z_bottom"],
        rear_rotation=d["rear_rotation"],
        rear_skew=d["rear_skew"],
    )


class GaitParamEnv(gym.Env):
    """
    Gymnasium environment for gait parameter optimization via RL.

    Each episode is a single headless simulation. The agent's action
    is a 14-dimensional vector in [-1, 1] that gets mapped to physical
    gait parameters. The reward combines COT, forward velocity, and
    body stability.

    Action space:  Box(-1, 1, shape=(14,)) — normalized parameters
    Observation:   Box(..., shape=(5,)) — metrics from the last episode
                   [cot, distance, velocity, avg_tilt, survived]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gait: Gait = Gait.TROT,
        reward_weights: RewardWeights | None = None,
        sim_length: float = 5.0,
        warmup: float = 2.0,
        bounds: ParamBounds | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            gait:           Gait pattern (WALK, TROT, CANTER, GALLOP, etc.)
            reward_weights: Weights for multi-component reward (default: RewardWeights())
            sim_length:     Duration of measurement phase in seconds
            warmup:         Duration of warmup phase in seconds
            bounds:         Parameter bounds (default: auto from gait with ±20%)
            verbose:        If True, print MuJoCo/IK warnings during simulation
        """
        super().__init__()

        self.gait = gait
        self.weights = reward_weights or RewardWeights()
        self.sim_length = sim_length
        self.warmup = warmup
        self.bounds = bounds or make_bounds(gait)
        self.verbose = verbose

        # 14D normalized action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(ALL_PARAM_KEYS),), dtype=np.float32
        )

        # Observation: metrics from the last sim [cot, distance, velocity, avg_tilt, survived]
        # On reset, returns zeros (no previous episode data)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, -5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 50.0, 10.0, np.pi, 1.0], dtype=np.float32),
        )

        # Path to MuJoCo model
        self._xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')

        # Adaptive PD parameters (same as ai_fix_param.py)
        self._pd_params = (0.2, 5.0, 0.05)

        # Last metrics for observation
        self._last_metrics: np.ndarray = np.zeros(5, dtype=np.float32)

        # Episode counter
        self.episode_count = 0

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] to physical parameter ranges."""
        # Linear mapping: param = low + (action + 1) / 2 * (high - low)
        return self.bounds.low + (action + 1.0) / 2.0 * (self.bounds.high - self.bounds.low)

    def _action_to_param_dict(self, action: np.ndarray) -> dict[str, float]:
        """Convert a normalized action vector to a named parameter dict."""
        raw = self._denormalize(action)
        return {key: float(raw[i]) for i, key in enumerate(ALL_PARAM_KEYS)}

    def _run_simulation(self, param_dict: dict[str, float]) -> dict:
        """
        Run one headless simulation with the given parameters.
        Returns the extended metrics dict from headless_sim_extended().
        On any failure, returns a penalty dict.
        """
        freq = param_dict["freq"]
        duty_factor = param_dict["duty_factor"]
        cfg = _dict_to_cfg(param_dict)

        try:
            # Build fresh sim and controller
            robot_interface = RobotInterface(
                starting_state=State(mode=Mode.MOVING, gait=self.gait, frequency=freq),
                trajectory_method=TrajectoryMethod.ELLIPSOID,
                duty_factor=duty_factor,
            )
            sim = MujocoSim(self._xml_path, robot_interface=robot_interface, window_scale=2.0)
            controller = IKController(
                robot_interface=robot_interface,
                stride_length=None, step_height=None,
                params=self._pd_params, use_adaptive_pd=True,
                ellipsoid_config=cfg,
            )

            # Run headless sim with extended metrics
            if self.verbose:
                metrics = sim.headless_sim_extended(
                    controller=controller, sim_length=self.sim_length, warmup=self.warmup
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    metrics = sim.headless_sim_extended(
                        controller=controller, sim_length=self.sim_length, warmup=self.warmup
                    )

            return metrics

        except Exception:
            # Return penalty metrics on any failure
            return {
                "cot": None,
                "distance": 0.0,
                "velocity": 0.0,
                "avg_tilt": np.pi,   # worst-case tilt
                "survived": False,
            }

    def _compute_reward(self, metrics: dict) -> float:
        """
        Multi-component reward function.

        R = -w_cot * COT + w_vel * velocity - w_tilt * avg_tilt - w_fall * (fell)

        If the robot fell or COT is invalid, a large penalty is applied.
        """
        w = self.weights

        # Fall penalty
        if not metrics["survived"]:
            return -w.w_fall

        cot = metrics["cot"]
        if cot is None or cot <= 0:
            # Robot didn't move forward — treat as failure
            return -w.w_fall

        # Multi-component reward
        reward = (
            -w.w_cot * cot
            + w.w_vel * metrics["velocity"]
            - w.w_tilt * metrics["avg_tilt"]
        )

        return float(reward)

    def _metrics_to_obs(self, metrics: dict) -> np.ndarray:
        """Convert metrics dict to observation array."""
        cot = metrics["cot"] if metrics["cot"] is not None else 10.0
        return np.array([
            cot,
            metrics["distance"],
            metrics["velocity"],
            metrics["avg_tilt"],
            1.0 if metrics["survived"] else 0.0,
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Run one simulation episode with the given action (normalized parameters).

        Returns:
            observation: metrics from this episode
            reward: multi-component reward
            terminated: always True (single-step episodes)
            truncated: always False
            info: full metrics dict + parameter dict
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Denormalize to physical parameters
        param_dict = self._action_to_param_dict(action)

        # Run simulation
        metrics = self._run_simulation(param_dict)

        # Compute reward
        reward = self._compute_reward(metrics)

        # Build observation
        obs = self._metrics_to_obs(metrics)
        self._last_metrics = obs

        self.episode_count += 1

        # Info dict with full details for logging
        info = {
            "metrics": metrics,
            "params": param_dict,
            "episode": self.episode_count,
        }

        # Single-step episode: always terminated
        return obs, reward, True, False, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment. Returns zeros as initial observation
        since there's no meaningful state between episodes.
        """
        super().reset(seed=seed)
        obs = self._last_metrics.copy()
        return obs, {}

    def get_default_action(self) -> np.ndarray:
        """
        Return the action corresponding to default parameters (maps to [0, 0, ..., 0]
        approximately, depending on whether defaults are centered in bounds).
        """
        ref = default_params(self.gait)
        raw = np.array([ref[k] for k in ALL_PARAM_KEYS])
        # Inverse of denormalize: action = 2 * (raw - low) / (high - low) - 1
        action = 2.0 * (raw - self.bounds.low) / (self.bounds.high - self.bounds.low) - 1.0
        return action.astype(np.float32)
