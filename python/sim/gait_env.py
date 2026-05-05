"""
Gymnasium environment for RL-based gait parameter optimization.

Each episode is a single headless simulation run. The agent selects
12 ellipsoid shape parameters (normalized to [-1, 1]); frequency and
duty factor are locked to their initial values and excluded from the
search space. The environment runs the simulation and returns a
multi-component reward based on Cost of Transport, forward velocity,
and body stability.
"""
import os
import sys
import io
import contextlib
import traceback
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


# Process-local guard: ``_run_simulation`` swallows every exception so the
# optimiser keeps running, but a *silent* failure on every candidate looks
# identical to a flat fitness landscape and trips CMA-ES's ``tolfun``
# stopping criterion after one generation. We log the first traceback so
# users have something actionable in the LogConsole; subsequent failures
# stay quiet to avoid spamming an overnight run.
_LOGGED_RUN_SIM_FAILURE: bool = False


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
    """Weights for the multi-component reward function.

    Default magnitudes encode the user-requested priority order
    ``gait >> velocity > CoT > stability``. The reward computation in
    :meth:`GaitParamEnv._compute_reward` is *gated by ``curriculum_phase``*:
    Phase 1 uses only ``w_gait``; Phase 2 adds ``w_vel``; Phase 3 enables
    every term below. Tune the weights without changing the priority order
    or the curriculum will lose its meaning.
    """
    # ── Primary: footfall pattern matching ──
    # Dominant term — must be large enough that the optimiser can never
    # buy a better reward by sacrificing the gait pattern for raw speed.
    w_gait: float = 3.0
    # ── Secondary: forward velocity ──
    w_vel: float = 1.0
    # ── Tertiary: efficiency ──
    w_cot: float = 0.2
    # ── Quaternary: stability ──
    w_tilt: float = 0.1      # Body tilt penalty (|roll| + |pitch|)
    w_ang_vel: float = 0.05  # Base angular velocity penalty (rad/s)
    # ── Optional shaping (Phase 3 only) ──
    w_slip: float = 0.1          # Stance-phase foot slipping penalty
    w_clearance: float = 0.5     # Swing-phase ground-clearance penalty
    # ── Hard cliff ──
    w_fall: float = 10.0     # Penalty for falling over / moving backwards / invalid CoT


@dataclass
class ParamBounds:
    """Lower and upper bounds for each optimised shape parameter."""
    low: np.ndarray    # shape (12,)
    high: np.ndarray   # shape (12,)


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
            "freq": 2.30, "duty_factor": 0.31,
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


def make_bounds(
    gait: Gait,
    margin: float = 0.20,
    min_half_range: float = 0.02,
    initial_params: dict[str, float] | None = None,
) -> ParamBounds:
    """
    Build parameter bounds for the 12 ellipsoid shape keys only.

    ``freq`` and ``duty_factor`` are **locked** (not optimised) so they
    are excluded from the bounds array.  The returned arrays have shape
    ``(len(SHAPE_KEYS),)`` = ``(12,)``.

    Args:
        gait: Gait type to get default parameter values
        margin: Fractional margin around defaults (0.20 = ±20%)
        min_half_range: Minimum half-range for near-zero parameters
        initial_params: Optional override for the bound centre. When supplied,
            the per-key ±margin window is built around these values instead of
            the hard-coded ``default_params(gait)`` table — used by the GUI's
            "find_params" flow so the search starts from the user's hand-tuned
            ``.<GAIT>.ini`` values rather than the script defaults.

    Returns:
        ParamBounds with low and high arrays of shape (12,)
    """
    # When the caller passed an initial guess, centre the search around that;
    # otherwise fall back to the gait's hard-coded defaults.
    ref = dict(initial_params) if initial_params is not None else default_params(gait)
    low = np.zeros(len(SHAPE_KEYS))
    high = np.zeros(len(SHAPE_KEYS))

    for i, key in enumerate(SHAPE_KEYS):
        val = ref[key]
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
    is a 12-dimensional vector in [-1, 1] that gets mapped to the
    ellipsoid shape parameters (``SHAPE_KEYS``). Frequency and duty
    factor are **locked** to their initial values (from
    ``initial_params`` or the gait's hard-coded defaults) and are not
    part of the search space — they are injected back into the full
    14-key param dict before every simulation.

    Action space:  Box(-1, 1, shape=(12,)) — normalized shape parameters
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
        initial_params: dict[str, float] | None = None,
        curriculum_phase: int = 3,
    ):
        """
        Args:
            gait:             Gait pattern (WALK, TROT, CANTER, GALLOP, etc.)
            reward_weights:   Weights for multi-component reward (default: RewardWeights())
            sim_length:       Duration of measurement phase in seconds
            warmup:           Duration of warmup phase in seconds
            bounds:           Parameter bounds (default: auto from gait with ±20%)
            verbose:          If True, print MuJoCo/IK warnings during simulation
            initial_params:   Optional starting values for the search (one float
                per key in ALL_PARAM_KEYS). When supplied and ``bounds`` is None,
                the auto-built bounds centre on these values; ``get_default_action``
                also returns the normalised encoding of these values so CMA-ES /
                any other algorithm using ``x0`` starts from the user's tuned
                ``.<GAIT>.ini`` instead of the hard-coded defaults table.
            curriculum_phase: 1, 2, or 3 — gates which reward terms are summed.
                Phase 1 = footfall pattern only (R_gait); Phase 2 adds forward
                velocity (R_gait + R_vel); Phase 3 is the full reward
                (R_gait + R_vel + R_cot + R_stability + R_slip + R_clearance).
                Use Phase 1 to seed a clean gait pattern, then re-run with
                Phase 2 / 3 to optimise speed and efficiency on top.
        """
        super().__init__()

        self.gait = gait
        self.weights = reward_weights or RewardWeights()
        self.sim_length = sim_length
        self.warmup = warmup
        if curriculum_phase not in (1, 2, 3):
            raise ValueError(
                f"curriculum_phase must be 1, 2, or 3 (got {curriculum_phase!r})"
            )
        self.curriculum_phase = curriculum_phase
        # Stored so ``get_default_action`` can prefer it over default_params(gait).
        self.initial_params = dict(initial_params) if initial_params is not None else None

        # ── Locked parameters ────────────────────────────────────────────────
        # freq and duty_factor are held constant at their initial (or default)
        # values — only the 12 ellipsoid shape parameters are optimised.
        ref = self.initial_params if self.initial_params is not None else default_params(gait)
        self._locked_params: dict[str, float] = {
            "freq": float(ref["freq"]),
            "duty_factor": float(ref["duty_factor"]),
        }

        self.bounds = bounds or make_bounds(gait, initial_params=self.initial_params)
        self.verbose = verbose

        # 12D normalized action space — only the ellipsoid shape keys.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(SHAPE_KEYS),), dtype=np.float32
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
        """Convert a normalized 12D action vector to a full 14-key parameter dict.

        The 12D action maps to ``SHAPE_KEYS`` via the bounds; ``freq`` and
        ``duty_factor`` are injected from ``self._locked_params``.
        """
        raw = self._denormalize(action)
        param_dict = {key: float(raw[i]) for i, key in enumerate(SHAPE_KEYS)}
        # Inject locked freq / duty_factor so the full dict has all 14 keys.
        param_dict.update(self._locked_params)
        return param_dict

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
            # Return penalty metrics on any failure so the optimiser keeps
            # exploring instead of crashing. Without diagnostics, however,
            # an "every candidate fails" condition (e.g. constructor
            # mismatch, missing XML) is invisible — and produces an
            # identical penalty across the whole generation, which trips
            # CMA-ES's ``tolfun`` stopping criterion after one iteration.
            # Log once per process so the LogConsole shows what broke.
            global _LOGGED_RUN_SIM_FAILURE
            if not _LOGGED_RUN_SIM_FAILURE:
                _LOGGED_RUN_SIM_FAILURE = True
                print("[gait_env] _run_simulation raised — first occurrence "
                      "shown below; further failures will be suppressed:",
                      file=sys.stderr)
                traceback.print_exc()
            return {
                "cot": None,
                "distance": 0.0,
                "velocity": 0.0,
                "avg_tilt": np.pi,   # worst-case tilt
                "survived": False,
                # Worst-case fillers for the gait-quality terms; matches the
                # semantics of `survived=False` (catastrophic episode) but
                # keeps every key present so `_compute_reward` and
                # `_metrics_to_obs` don't KeyError on the fallback path.
                "gait_error": 4.0,
                "slip": 0.0,
                "swing_clearance": 0.0,
                "ang_vel": 0.0,
            }

    # ── Reward components ──
    # Each helper takes the metrics dict and the active `RewardWeights` and
    # returns a *signed* contribution. `_compute_reward` adds them up under
    # the curriculum gate. Keep them small and pure — easy to unit-test, easy
    # to swap individual terms without touching the orchestrator.
    @staticmethod
    def _r_gait(m: dict, w: RewardWeights) -> float:
        return -w.w_gait * float(m["gait_error"])

    @staticmethod
    def _r_vel(m: dict, w: RewardWeights) -> float:
        return w.w_vel * float(m["velocity"])

    @staticmethod
    def _r_cot(m: dict, w: RewardWeights) -> float:
        return -w.w_cot * float(m["cot"])

    @staticmethod
    def _r_stability(m: dict, w: RewardWeights) -> float:
        return -(w.w_tilt * float(m["avg_tilt"]) + w.w_ang_vel * float(m["ang_vel"]))

    @staticmethod
    def _r_slip(m: dict, w: RewardWeights) -> float:
        return -w.w_slip * float(m["slip"])

    @staticmethod
    def _r_clearance(m: dict, w: RewardWeights) -> float:
        return -w.w_clearance * float(m["swing_clearance"])

    def _compute_reward(self, metrics: dict) -> float:
        """
        Multi-component reward function gated by ``self.curriculum_phase``.

        Priority order is enforced by both the default weights in
        :class:`RewardWeights` and the curriculum:

            Phase 1: R = R_gait                                       (pattern only)
            Phase 2: R = R_gait + R_vel                               (pattern + speed)
            Phase 3: R = R_gait + R_vel + R_cot + R_stability         (full reward)
                       + R_slip + R_clearance

        The hard cliffs below (``-w_fall`` on fall / backward / invalid CoT)
        always apply, regardless of phase, so the optimiser never gets a
        positive score for a catastrophic candidate.
        """
        w = self.weights

        # Hard cliffs — catastrophic failures still score much worse than
        # any viable candidate, but each tier includes a small tiebreaker
        # so CMA-ES receives gradient information even when the entire
        # population is failing.  Without these tiebreakers every failing
        # candidate produces identical fitness, creating a flat landscape
        # that trips CMA-ES's ``tolflatfitness`` stopping criterion after
        # a single generation.
        #
        # Tiebreaker magnitude is capped at 0.1 × w_fall so the penalty
        # structure is preserved: any failing candidate still scores
        # far below any viable one.
        if not metrics["survived"]:
            # Worst tier — robot fell over.  Give a tiny bonus for
            # forward distance traveled before falling so CMA-ES can
            # distinguish "fell immediately" from "walked a bit then fell."
            dist_bonus = 0.01 * max(float(metrics.get("distance", 0.0)), 0.0)
            return -w.w_fall + min(dist_bonus, 0.1 * w.w_fall)
        # Backwards motion is treated as nearly as bad as falling. The
        # optimiser needs a clear, large signal to bias the CMA-ES mean
        # *away* from gaits that produce net-negative forward velocity.
        # The tiebreaker gives a slight gradient: less-negative velocity
        # is preferred (closer to 0 → closer to forward motion).
        if metrics["velocity"] < 0:
            # velocity is negative; clamp so the bonus stays in [0, 0.1*w_fall).
            vel_bonus = 0.01 * max(1.0 + float(metrics["velocity"]), 0.0)
            return -w.w_fall * 0.95 + min(vel_bonus, 0.1 * w.w_fall)
        cot = metrics["cot"]
        if cot is None or cot <= 0:
            # Survived and moved forward but no valid CoT — least-bad
            # failure tier, scored slightly above the other two.
            return -w.w_fall * 0.9

        # Phase-gated sum. Phase 1 trains the policy to *follow the
        # scheduler's footfall pattern* before any other concern. Phase 2
        # adds forward velocity once the pattern is locked in. Phase 3 is
        # the full reward including efficiency, stability, and the slip /
        # clearance shaping terms.
        reward = self._r_gait(metrics, w)
        if self.curriculum_phase >= 2:
            reward += self._r_vel(metrics, w)
        if self.curriculum_phase >= 3:
            reward += self._r_cot(metrics, w)
            reward += self._r_stability(metrics, w)
            reward += self._r_slip(metrics, w)
            reward += self._r_clearance(metrics, w)

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
        Return the 12D action corresponding to default shape parameters.

        Only covers ``SHAPE_KEYS`` — ``freq`` and ``duty_factor`` are locked
        and not part of the action space.

        When ``initial_params`` was supplied at construction, those values are
        used as the seed instead of ``default_params(gait)`` — that's what makes
        CMA-ES (which calls this for its ``x0``) start its search from the
        user's tuned ``.<GAIT>.ini``.
        """
        ref = self.initial_params if self.initial_params is not None else default_params(self.gait)
        raw = np.array([ref[k] for k in SHAPE_KEYS])
        # Inverse of denormalize: action = 2 * (raw - low) / (high - low) - 1
        action = 2.0 * (raw - self.bounds.low) / (self.bounds.high - self.bounds.low) - 1.0
        return action.astype(np.float32)
