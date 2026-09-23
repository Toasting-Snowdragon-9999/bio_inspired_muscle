"""
@brief Gaussian-process parameter sweep for ellipsoid gait trajectories.

Drives the headless MuJoCo simulation to tune the 12 ellipsoid shape
parameters (plus frequency) of a quadruped gait by minimising Cost of
Transport (CoT). Supports a single GUI run for visual inspection
(``elip_traj_test``) and a two-phase Bayesian-optimisation sweep
(``run_sweep``) that alternates between shape-only and frequency-only
search with random pre-sampling to seed the Gaussian-process surrogate.
``duty_factor`` is fixed throughout and never optimised.
"""
import os
import sys
import csv
import argparse
from datetime import datetime

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import RobotInterface, State, Mode, Gait, TrajectoryMethod

# ── Penalty COT returned when a trial fails (robot fell, IK diverged, etc.) ──
PENALTY_COT = 10.0

# ── Search bounds: ±15% of default magnitude, with minimum range for near-zero defaults ──
# Keeping bounds tight increases the feasibility rate (viable sims per random trial).
# The GP only gets to see feasible points, so a higher feasibility rate means
# more useful signal per trial.
MIN_HALF_RANGE = 0.015

# ── 12 shape parameter names (excludes freq and duty_factor) ─────────────────
SHAPE_KEYS: list[str] = [
    "front_x_fore", "front_x_hind", "front_z_top", "front_z_bottom",
    "front_rotation", "front_skew",
    "rear_x_fore", "rear_x_hind", "rear_z_top", "rear_z_bottom",
    "rear_rotation", "rear_skew",
]

# ── All optimised parameter names (freq + shape; duty_factor is always fixed) ─
ALL_KEYS: list[str] = ["freq"] + SHAPE_KEYS


def default_cfg(gait: Gait) -> dict[str, float]:
    """
    @brief Return a dict of hardcoded default parameters for the requested gait.

    Return a dict of hardcoded default parameters for the requested gait.
    duty_factor is included here but is never modified by the sweep — it is
    passed verbatim into every simulation run.

    @param gait: Gait type whose default parameter table to return.
    @return Dict of all 14 parameter values; falls back to the WALK defaults
        for an unknown gait.
    """
    if gait == Gait.WALK:
        return {
            "freq":           1.40,
            "duty_factor":    0.24,
            "front_x_fore":   0.14,
            "front_x_hind":   0.08,
            "front_z_top":    0.10,
            "front_z_bottom": 0.02,
            "front_rotation": 0.05,
            "front_skew":     0.00,
            "rear_x_fore":    0.06,
            "rear_x_hind":    0.14,
            "rear_z_top":     0.10,
            "rear_z_bottom":  0.02,
            "rear_rotation":  -0.05,
            "rear_skew":      0.00,
        }
    elif gait == Gait.TROT:
        return {
            "freq":           1.95,
            "duty_factor":    0.50,
            "front_x_fore":   0.123,
            "front_x_hind":   0.095,
            "front_z_top":    0.104,
            "front_z_bottom": 0.025,
            "front_rotation": 0.046,
            "front_skew":     0.035,
            "rear_x_fore":    0.075,
            "rear_x_hind":    0.122,
            "rear_z_top":     0.095,
            "rear_z_bottom":  0.005,
            "rear_rotation":  -0.060,
            "rear_skew":      -0.015,
        }
    elif gait == Gait.AMBLE:
        return {
            "freq":           1.675,
            "duty_factor":    0.40,
            "front_x_fore":   0.14,
            "front_x_hind":   0.08,
            "front_z_top":    0.10,
            "front_z_bottom": 0.02,
            "front_rotation": 0.05,
            "front_skew":     0.00,
            "rear_x_fore":    0.06,
            "rear_x_hind":    0.14,
            "rear_z_top":     0.10,
            "rear_z_bottom":  0.02,
            "rear_rotation":  -0.05,
            "rear_skew":      0.00,
        }
    elif gait == Gait.CANTER:
        return {
            "freq":           3.50,
            "duty_factor":    0.31,
            "front_x_fore":   0.14,
            "front_x_hind":   0.07,
            "front_z_top":    0.12,
            "front_z_bottom": 0.02,
            "front_rotation": 0.10,
            "front_skew":     0.02,
            "rear_x_fore":    0.06,
            "rear_x_hind":    0.15,
            "rear_z_top":     0.10,
            "rear_z_bottom":  0.03,
            "rear_rotation":  -0.05,
            "rear_skew":      -0.03,
        }
    elif gait == Gait.GALLOP:
        return {
            "freq":           3.50,
            "duty_factor":    0.65,
            "front_x_fore":   0.14,
            "front_x_hind":   0.06,
            "front_z_top":    0.14,
            "front_z_bottom": 0.02,
            "front_rotation": 0.00,
            "front_skew":     0.00,
            "rear_x_fore":    0.06,
            "rear_x_hind":    0.16,
            "rear_z_top":     0.14,
            "rear_z_bottom":  0.02,
            "rear_rotation":  0.00,
            "rear_skew":      0.00,
        }
    else:
        # Fallback: use WALK defaults for unknown gaits
        return default_cfg(Gait.WALK)


def _make_bounds(keys: list[str], reference: dict[str, float]) -> list[tuple[float, float]]:
    """@brief Build (low, high) bounds for the given parameter keys relative to a reference dict.

    Build (low, high) bounds for the given parameter keys relative to a reference dict.

    @param keys: Parameter keys to build bounds for.
    @param reference: Reference dict supplying the centre value of each key.
    @return List of ``(low, high)`` tuples, one per key — ±15% of the
        reference magnitude, widened to ``MIN_HALF_RANGE`` for near-zero values.
    """
    bounds = []
    for k in keys:
        v = reference[k]
        # ±15% — keeps feasibility rate high enough for the GP to learn
        half = max(abs(v) * 0.15, MIN_HALF_RANGE)
        bounds.append((v - half, v + half))
    return bounds


def parse_args() -> argparse.Namespace:
    """@brief Parse command-line arguments for the single-run / sweep entry point.

    Parses ``--gait`` first so per-gait defaults from ``default_cfg`` can be used
    as the defaults for every parameter flag.

    @return Parsed argparse.Namespace with gait, all 14 parameters, and the
        sweep-mode options.
    """
    # ── Pre-parse to find --gait so we can load defaults from default_cfg ────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gait", type=str, default="WALK")
    known, _ = pre.parse_known_args()
    try:
        gait_enum = Gait[known.gait.upper()]
    except KeyError:
        gait_enum = Gait.WALK
    D = default_cfg(gait_enum)  # hardcoded defaults for this gait

    p = argparse.ArgumentParser(
        description="Run the ellipsoid trajectory sim with configurable EllipsoidConfig parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Gait selection ────────────────────────────────────────────────────────
    p.add_argument("--gait", type=str, default="WALK",
                   help="Gait name (WALK, TROT, CANTER, GALLOP). Sets param defaults.")
    # ── Front legs (FL, FR) ──────────────────────────────────────────────────
    p.add_argument("--freq",           type=float, default=D["freq"],           help="Gait frequency (Hz)")
    p.add_argument("--front_x_fore",   type=float, default=D["front_x_fore"],   help="Front: forward reach (m)")
    p.add_argument("--front_x_hind",   type=float, default=D["front_x_hind"],   help="Front: rearward reach (m)")
    p.add_argument("--front_z_top",    type=float, default=D["front_z_top"],    help="Front: swing height (m)")
    p.add_argument("--front_z_bottom", type=float, default=D["front_z_bottom"], help="Front: stance depth (m)")
    p.add_argument("--front_rotation", type=float, default=D["front_rotation"], help="Front: ellipse rotation (rad)")
    p.add_argument("--front_skew",     type=float, default=D["front_skew"],     help="Front: x-displacement (m)")
    # ── Rear legs (RL, RR) ────────────────────────────────────────────────────
    p.add_argument("--rear_x_fore",    type=float, default=D["rear_x_fore"],    help="Rear:  forward reach (m)")
    p.add_argument("--rear_x_hind",    type=float, default=D["rear_x_hind"],    help="Rear:  rearward reach (m)")
    p.add_argument("--rear_z_top",     type=float, default=D["rear_z_top"],     help="Rear:  swing height (m)")
    p.add_argument("--rear_z_bottom",  type=float, default=D["rear_z_bottom"],  help="Rear:  stance depth (m)")
    p.add_argument("--rear_rotation",  type=float, default=D["rear_rotation"],  help="Rear:  ellipse rotation (rad)")
    p.add_argument("--rear_skew",      type=float, default=D["rear_skew"],      help="Rear:  x-displacement (m)")
    # ── Duty factor (fixed, never swept) ─────────────────────────────────────
    p.add_argument("--duty_factor",    type=float, default=D["duty_factor"],
                   help="Fraction of cycle in stance (0–1). Fixed during sweep — not optimised.")
    # ── Sweep mode ────────────────────────────────────────────────────────────
    p.add_argument("--sweep",    action="store_true",    help="Run Bayesian optimisation sweep instead of a single sim")
    p.add_argument("--n_calls",  type=int, default=1000, help="Total optimisation evaluations per phase-1 run")
    p.add_argument("--output",   type=str, default=None, help="CSV output path (default: sweep_results_<timestamp>.csv)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_and_controller(
    freq: float, cfg: EllipsoidConfig, duty_factor: float = 0.5, gait: Gait = Gait.WALK
) -> tuple[MujocoSim, IKController, RobotInterface]:
    """@brief Create a fresh MujocoSim + IKController for the given parameters.

    Create a fresh MujocoSim + IKController for the given parameters.

    @param freq: Gait frequency (Hz).
    @param cfg: Ellipsoid trajectory configuration for the controller.
    @param duty_factor: Fraction of the cycle spent in stance (0-1).
    @param gait: Gait pattern controlling the CPG phase offsets.
    @return Tuple ``(sim, controller, robot_interface)`` ready for a headless run.
    """
    robot_interface = RobotInterface(
        starting_state=State(mode=Mode.MOVING, gait=gait, frequency=freq),
        trajectory_method=TrajectoryMethod.ELLIPSOID,
        duty_factor=duty_factor,
    )
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )
    controller = IKController(
        robot_interface=robot_interface,
        stride_length=None, step_height=None,
        params=params, use_adaptive_pd=True,
        ellipsoid_config=cfg,
    )
    return sim, controller, robot_interface


def _dict_to_cfg(d: dict) -> EllipsoidConfig:
    """@brief Build an EllipsoidConfig from a param dict (keys are SHAPE_KEYS).

    Build an EllipsoidConfig from a param dict (keys are SHAPE_KEYS).

    @param d: Parameter dict containing every SHAPE_KEYS entry.
    @return EllipsoidConfig populated from the 12 shape values in ``d``.
    """
    return EllipsoidConfig(
        front_x_fore   = d["front_x_fore"],
        front_x_hind   = d["front_x_hind"],
        front_z_top    = d["front_z_top"],
        front_z_bottom = d["front_z_bottom"],
        front_rotation = d["front_rotation"],
        front_skew     = d["front_skew"],
        rear_x_fore    = d["rear_x_fore"],
        rear_x_hind    = d["rear_x_hind"],
        rear_z_top     = d["rear_z_top"],
        rear_z_bottom  = d["rear_z_bottom"],
        rear_rotation  = d["rear_rotation"],
        rear_skew      = d["rear_skew"],
    )


def evaluate_params(param_dict: dict, duty_factor: float, gait: Gait = Gait.WALK, verbose: bool = False) -> float:
    """
    @brief Run one headless simulation and return the COT.

    Run one headless simulation and return the COT.
    duty_factor is passed explicitly and is never varied by the optimiser.
    gait controls the CPG phase offsets — must match the intended gait pattern.
    Returns PENALTY_COT on any failure (robot fell, IK error, etc.).
    Suppresses MuJoCo/IK warnings unless verbose=True.

    @param param_dict: Parameter dict containing ``freq`` and the 12 shape keys.
    @param duty_factor: Fixed duty factor passed straight into the simulation.
    @param gait: Gait pattern controlling the CPG phase offsets.
    @param verbose: If True, allow MuJoCo/IK warnings through instead of suppressing them.
    @return Measured Cost of Transport, or PENALTY_COT on failure / invalid CoT.
    """
    import io, contextlib
    freq = param_dict["freq"]
    cfg = _dict_to_cfg(param_dict)
    try:
        sim, controller, _ = _build_sim_and_controller(freq, cfg, duty_factor=duty_factor, gait=gait)
        # Suppress noisy MuJoCo/IK warnings during headless sweep runs
        if verbose:
            cot = sim.headless_sim(controller=controller, sim_length=5.0, warmup=2.0)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cot = sim.headless_sim(controller=controller, sim_length=5.0, warmup=2.0)
        if cot is None or cot <= 0:
            return PENALTY_COT
        return cot
    except Exception:
        return PENALTY_COT


# ─────────────────────────────────────────────────────────────────────────────
# Single-run entry — GUI sim with 2 s warmup before COT measurement
# ─────────────────────────────────────────────────────────────────────────────

def elip_traj_test(gait_enum: Gait = Gait.WALK):
    """@brief Run a single windowed GUI simulation and report its Cost of Transport.

    Builds the sim and controller from the parsed CLI arguments, runs a 10 s
    visualised simulation with a 2 s warmup, then measures CoT.

    @param gait_enum: Gait pattern to simulate.
    @return Measured Cost of Transport for the run.
    """
    args = parse_args()

    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=gait_enum, frequency=args.freq), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=args.duty_factor)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    cfg = EllipsoidConfig(
        front_x_fore   = args.front_x_fore,
        front_x_hind   = args.front_x_hind,
        front_z_top    = args.front_z_top,
        front_z_bottom = args.front_z_bottom,
        front_rotation = args.front_rotation,
        front_skew     = args.front_skew,
        rear_x_fore    = args.rear_x_fore,
        rear_x_hind    = args.rear_x_hind,
        rear_z_top     = args.rear_z_top,
        rear_z_bottom  = args.rear_z_bottom,
        rear_rotation  = args.rear_rotation,
        rear_skew      = args.rear_skew,
    )
    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )
    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
    sim.sim(controller=controller, sim_length=10, slow_factor=2.0, warmup=2.0)

    cot = sim.compute_CoT()
    return cot


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian optimisation sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(n_calls: int, output_path: str, initial_params: dict[str, float], gait: Gait = Gait.WALK) -> None:
    """
    @brief Two-phase Bayesian optimisation of ellipsoid trajectory parameters.

    Two-phase Bayesian optimisation of ellipsoid trajectory parameters.

    duty_factor is NEVER varied — it is fixed from initial_params throughout.

    Outer loop (restart loop):
        Phase 1 — Shape parameters only, freq fixed:
            Uses the full n_calls budget to explore the 12-D shape space.
        Phase 2 — Frequency only, shape fixed at phase 1 best:
            Uses a smaller budget (20% of n_calls) to search for a better freq.
        If phase 2 finds a strictly better freq (lower COT), the current best
        params are updated with the new freq and phase 1 restarts.
        The loop exits when phase 2 fails to improve.

    Early stopping: if best COT improves < 1% over 50 consecutive trials
    the current phase terminates early.

    All trials are logged to CSV.

    @param n_calls: Total optimisation evaluations granted to phase 1 per restart.
    @param output_path: CSV path that every trial row is appended to.
    @param initial_params: Starting values for ``freq``, the 12 shape keys, and
        the fixed ``duty_factor``; the search is centred on these.
    @param gait: Gait pattern controlling the CPG phase offsets.
    """
    from skopt import gp_minimize
    from skopt.callbacks import EarlyStopper

    # ── Fixed duty_factor — never changes ─────────────────────────────────────
    duty_factor = initial_params["duty_factor"]

    # ── Budget split ──────────────────────────────────────────────────────────
    phase2_budget = max(10, int(n_calls * 0.2))
    phase1_budget = max(10, n_calls)  # phase 1 always gets full n_calls budget

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_file = open(output_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["restart", "phase", "trial", "cot"] + ALL_KEYS)
    trial_counter = [0]  # global trial counter across all restarts

    print(f"═══ {gait.name} Gait COT Bayesian Optimisation ═══")
    print(f"  duty_factor  : {duty_factor} (fixed)")
    print(f"  Phase 1 budget per restart : {phase1_budget}")
    print(f"  Phase 2 budget per restart : {phase2_budget}")
    print(f"  Output       : {output_path}")
    print()

    # ── Early stopping callback ───────────────────────────────────────────────
    class ConvergenceStopper(EarlyStopper):
        """@brief Stops if best COT improves < tol fraction over the last `window` evals.

        Stops if best COT improves < tol fraction over the last `window` evals.
        Requires at least `min_successes` non-penalty evaluations before it can
        fire — prevents premature termination when the failure rate is very high
        and all recent func_vals are PENALTY_COT.
        Window=200 and tol=0.005 are intentionally lenient — we want the GP to
        exhaust a large neighbourhood before declaring convergence.
        """
        def __init__(self, window: int = 200, tol: float = 0.005, min_successes: int = 30):
            """@brief Configure the convergence-based early stopper.

            @param window: Number of most-recent evaluations to measure improvement over.
            @param tol: Minimum fractional improvement required to keep going.
            @param min_successes: Minimum non-penalty evaluations before the
                stopper is allowed to fire.
            """
            super().__init__()
            self.window = window
            self.tol = tol
            self.min_successes = min_successes

        def _criterion(self, result):
            """@brief Decide whether the optimisation has converged.

            @param result: skopt result object exposing ``func_vals`` so far.
            @return True if improvement over the window has fallen below ``tol``
                (and enough successful evaluations exist), else False.
            """
            import numpy as _np
            vals = _np.array(result.func_vals)
            # Don't stop until we have enough successful (non-penalty) evaluations
            n_success = _np.sum(vals < PENALTY_COT)
            if n_success < self.min_successes:
                return False
            if len(vals) < self.window:
                return False
            recent = vals[-self.window:]
            best_recent = _np.min(recent)
            best_before = _np.min(vals[:-self.window]) if len(vals) > self.window else best_recent
            if best_before <= 0:
                return False
            improvement = (best_before - best_recent) / abs(best_before)
            return improvement < self.tol

    # ── Outer restart loop ────────────────────────────────────────────────────
    current_params = {k: initial_params[k] for k in ALL_KEYS}  # freq + shape (no duty_factor)
    global_best_cot = PENALTY_COT
    restart_idx = 0

    while True:
        restart_idx += 1
        fail_count = [0]
        success_count = [0]

        # ── Phase 1: optimise shape params, freq fixed ────────────────────────
        fixed_freq = current_params["freq"]
        print(f"── Restart {restart_idx} | Phase 1: shape optimisation (freq = {fixed_freq:.4f} Hz, duty = {duty_factor}) ──")
        shape_bounds = _make_bounds(SHAPE_KEYS, {**current_params, "duty_factor": duty_factor})
        best_cot_phase1 = [PENALTY_COT]

        def phase1_objective(x: list[float]) -> float:
            """@brief Phase 1 objective: evaluate a shape-only candidate at fixed freq.

            @param x: Candidate values for the 12 SHAPE_KEYS (in order).
            @return Measured CoT (or PENALTY_COT on failure); also logged to CSV.
            """
            trial_counter[0] += 1
            d = {k: v for k, v in zip(SHAPE_KEYS, x)}
            d["freq"] = fixed_freq
            cot = evaluate_params(d, duty_factor=duty_factor, gait=gait)
            if cot >= PENALTY_COT:
                fail_count[0] += 1
            else:
                success_count[0] += 1
            if cot < best_cot_phase1[0]:
                best_cot_phase1[0] = cot
            writer.writerow([restart_idx, 1, trial_counter[0], f"{cot:.6f}"] + [f"{d.get(k, fixed_freq):.6f}" for k in ALL_KEYS])
            csv_file.flush()
            if trial_counter[0] % 50 == 0:
                print(f"  [R{restart_idx} P1] trial {trial_counter[0]:>4d}  |  best COT = {best_cot_phase1[0]:.4f}  |  ok={success_count[0]} fail={fail_count[0]}")
            return cot

        # ── Random pre-sampling: find feasible points before invoking the GP ──
        # The GP surrogate scales as O(n³) per step. If almost all evaluations
        # return PENALTY_COT the GP has nothing useful to learn from and each
        # step becomes slower as n grows. We instead run pure random search
        # until we accumulate MIN_FEASIBLE successful trials, then seed the GP
        # with those points so it immediately has structure to exploit.
        MIN_FEASIBLE = 20        # target feasible points before starting GP
        MAX_RANDOM   = 600       # hard cap on random-phase trials per restart
        import numpy as _rng_np
        rng = _rng_np.random.default_rng(restart_idx * 17)

        pre_x: list[list[float]] = []   # feasible x vectors
        pre_y: list[float]       = []   # corresponding COT values
        all_pre_x: list[list[float]] = [[current_params[k] for k in SHAPE_KEYS]]  # include default
        all_pre_y: list[float]       = []

        # Always evaluate the current best point first
        _d0 = {k: current_params[k] for k in SHAPE_KEYS}
        _d0["freq"] = fixed_freq
        _cot0 = phase1_objective([current_params[k] for k in SHAPE_KEYS])
        all_pre_y.append(_cot0)
        if _cot0 < PENALTY_COT:
            pre_x.append([current_params[k] for k in SHAPE_KEYS])
            pre_y.append(_cot0)

        random_trials_used = 1
        while len(pre_x) < MIN_FEASIBLE and random_trials_used < MAX_RANDOM and trial_counter[0] < phase1_budget:
            # Sample uniformly within bounds
            x_rand = [rng.uniform(lo, hi) for lo, hi in shape_bounds]
            cot_rand = phase1_objective(x_rand)
            all_pre_x.append(x_rand)
            all_pre_y.append(cot_rand)
            if cot_rand < PENALTY_COT:
                pre_x.append(x_rand)
                pre_y.append(cot_rand)
            random_trials_used += 1

        print(f"  Pre-sampling done: {random_trials_used} random trials, {len(pre_x)} feasible found.")

        # Remaining budget for the GP phase
        gp_budget = phase1_budget - trial_counter[0]

        if gp_budget < 10:
            # Budget exhausted in random phase — use whatever we found
            print("  Budget exhausted in random pre-sampling phase.")
            if pre_y:
                best_idx = int(_rng_np.argmin(pre_y))
                phase1_best_shape = {k: v for k, v in zip(SHAPE_KEYS, pre_x[best_idx])}
                phase1_best_cot = pre_y[best_idx]
            else:
                phase1_best_shape = {k: current_params[k] for k in SHAPE_KEYS}
                phase1_best_cot = PENALTY_COT
        else:
            # Seed the GP with ONLY feasible points — passing hundreds of identical
            # PENALTY_COT values gives the GP nothing to learn from and makes each
            # kernel fit O(n³) slow.  With only feasible points the GP is small,
            # fast, and has real structure to exploit.
            if not pre_x:
                # No feasible points found at all — fall back to defaults
                print("  WARNING: no feasible points found. Using default params.")
                phase1_best_shape = {k: current_params[k] for k in SHAPE_KEYS}
                phase1_best_cot = PENALTY_COT
            else:
                result1 = gp_minimize(
                    phase1_objective,
                    shape_bounds,
                    n_calls=gp_budget,
                    n_initial_points=0,      # 0 = skip GP's own random phase; we supply x0/y0
                    x0=pre_x,               # feasible points only — keeps GP small and fast
                    y0=pre_y,
                    acq_func="EI",
                    random_state=restart_idx * 17,
                    callback=[ConvergenceStopper(window=200, tol=0.005, min_successes=30)],
                    verbose=False,
                )
                phase1_best_shape = {k: v for k, v in zip(SHAPE_KEYS, result1.x)}
                phase1_best_cot = result1.fun

        if phase1_best_cot < global_best_cot:
            global_best_cot = phase1_best_cot

        print(f"\n  Phase 1 complete — best COT = {phase1_best_cot:.4f}  (ok={success_count[0]}, fail={fail_count[0]})")
        print(f"  Trials used: {trial_counter[0]} / {phase1_budget}")
        print()

        # Reset counters for phase 2
        fail_count[0] = 0
        success_count[0] = 0

        # ── Phase 2: freq search only, shape fixed at phase 1 best ───────────
        print(f"── Restart {restart_idx} | Phase 2: frequency search (shape fixed) ──")
        # Frequency bounds: ±15% around current freq, clamped to [0.5, 4.0]
        freq_lo = max(0.5, fixed_freq * 0.85)
        freq_hi = min(4.0, fixed_freq * 1.15)
        freq_bounds = [(freq_lo, freq_hi)]
        best_cot_phase2 = [phase1_best_cot]  # phase 2 must beat phase 1 to trigger a restart

        def phase2_objective(x: list[float]) -> float:
            """@brief Phase 2 objective: evaluate a frequency candidate at fixed shape.

            @param x: Single-element list holding the candidate frequency.
            @return Measured CoT (or PENALTY_COT on failure); also logged to CSV.
            """
            trial_counter[0] += 1
            trial_freq = x[0]
            d = {**phase1_best_shape, "freq": trial_freq}
            cot = evaluate_params(d, duty_factor=duty_factor, gait=gait)
            if cot >= PENALTY_COT:
                fail_count[0] += 1
            else:
                success_count[0] += 1
            if cot < best_cot_phase2[0]:
                best_cot_phase2[0] = cot
            writer.writerow([restart_idx, 2, trial_counter[0], f"{cot:.6f}"] + [f"{d[k]:.6f}" for k in ALL_KEYS])
            csv_file.flush()
            if trial_counter[0] % 10 == 0:
                print(f"  [R{restart_idx} P2] trial {trial_counter[0]:>4d}  |  best COT = {best_cot_phase2[0]:.4f}  |  ok={success_count[0]} fail={fail_count[0]}")
            return cot

        # Phase 2 is 1-D so fewer initial points needed, but still vary the seed.
        n_initial2 = max(10, min(int(phase2_budget * 0.40), phase2_budget - 1))
        result2 = gp_minimize(
            phase2_objective,
            freq_bounds,
            n_calls=phase2_budget,
            n_initial_points=n_initial2,
            x0=[fixed_freq],
            acq_func="EI",
            random_state=restart_idx * 31,  # vary per restart
            callback=[ConvergenceStopper(window=60, tol=0.005, min_successes=10)],
            verbose=False,
        )

        phase2_best_freq = result2.x[0]
        phase2_best_cot  = result2.fun

        print(f"\n  Phase 2 complete — best COT = {phase2_best_cot:.4f}  new freq = {phase2_best_freq:.4f} Hz  (ok={success_count[0]}, fail={fail_count[0]})")
        print(f"  Trials used (phase 2): {len(result2.func_vals)} / {phase2_budget}")
        print()

        # ── Decide whether to restart ─────────────────────────────────────────
        if phase2_best_cot < phase1_best_cot:
            # Phase 2 found a strictly better frequency — restart phase 1 with it
            print(f"  ★ New best freq = {phase2_best_freq:.4f} Hz (COT {phase2_best_cot:.4f} < {phase1_best_cot:.4f}). Restarting phase 1.")
            current_params = {**phase1_best_shape, "freq": phase2_best_freq}
            global_best_cot = phase2_best_cot
        else:
            # No improvement — converged
            print(f"  Phase 2 found no better frequency. Converged after {restart_idx} restart(s).")
            # Overall best is phase 1 best (phase 2 did not beat it)
            current_params = {**phase1_best_shape, "freq": fixed_freq}
            global_best_cot = phase1_best_cot
            break

    csv_file.close()

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("═══════════════════════════════════════════════════════════════")
    print(f"  BEST COT = {global_best_cot:.6f}")
    print(f"  duty_factor = {duty_factor}  (unchanged)")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("Optimal parameters:")
    for k in ALL_KEYS:
        print(f"  --{k:<20s} {current_params[k]:.6f}")
    print(f"  --{'duty_factor':<20s} {duty_factor:.6f}")
    print()
    # Copy-pasteable CLI command
    cli_args = " ".join(f"--{k} {current_params[k]:.6f}" for k in ALL_KEYS)
    cli_args += f" --duty_factor {duty_factor:.6f}"
    print(f"Re-run with:\n  python ai_fix_param.py {cli_args}")
    print()
    print(f"Full results saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """@brief CLI entry point: dispatch to a single GUI run or the sweep.

    Parses arguments, resolves the gait enum, and either launches the Bayesian
    optimisation sweep (``--sweep``) or a single visualised run, printing the
    resulting Cost of Transport.
    """
    args = parse_args()

    # Resolve gait enum — fall back to WALK if unrecognised
    try:
        gait_enum = Gait[args.gait.upper()]
    except KeyError:
        print(f"Warning: unknown gait '{args.gait}', falling back to WALK.")
        gait_enum = Gait.WALK

    if args.sweep:
        # Build initial params dict from CLI args (which may be gait defaults if no args given)
        initial_params: dict[str, float] = {k: getattr(args, k) for k in ALL_KEYS}
        initial_params["duty_factor"] = args.duty_factor  # carry duty_factor for sim, not for sweep

        output = args.output or f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        run_sweep(n_calls=args.n_calls, output_path=output, initial_params=initial_params, gait=gait_enum)
    else:
        cot = elip_traj_test(gait_enum)
        print("Cost of Transport:", cot)


if __name__ == "__main__":
    main()

