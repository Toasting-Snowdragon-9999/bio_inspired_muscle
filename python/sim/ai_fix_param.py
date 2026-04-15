import os
import sys
import csv
import argparse
from datetime import datetime

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod

# ── Penalty COT returned when a trial fails (robot fell, IK diverged, etc.) ──
PENALTY_COT = 10.0

# ── Default parameter values for the WALK gait ──────────────────────────────
DEFAULTS = {
    "freq":           1.40,
    "duty_factor":    0.27,
    "front_x_fore":   0.14,
    "front_x_hind":   0.08,
    "front_z_top":    0.10,
    "front_z_bottom": 0.02,
    "front_rotation": 0.00,
    "front_skew":     0.00,
    "rear_x_fore":    0.06,
    "rear_x_hind":    0.12,
    "rear_z_top":     0.10,
    "rear_z_bottom":  0.02,
    "rear_rotation": -0.00,
    "rear_skew":      0.00,
}

# ── Search bounds: ±15% of default magnitude, with minimum range for near-zero defaults ──
# The IK solver and sim are very sensitive to parameter changes — even ±25%
# pushes most random samples into infeasible territory.  ±15% keeps a higher
# fraction of trials feasible so the Gaussian Process gets useful data.
MIN_HALF_RANGE = 0.015

def _make_bounds(keys: list[str]) -> list[tuple[float, float]]:
    """Build (low, high) bounds for the given parameter keys."""
    bounds = []
    for k in keys:
        v = DEFAULTS[k]
        half = max(abs(v) * 0.15, MIN_HALF_RANGE)
        bounds.append((v - half, v + half))
    return bounds


# ── 12 shape parameter names (excludes freq and duty_factor) ─────────────────
SHAPE_KEYS: list[str] = [
    "front_x_fore", "front_x_hind", "front_z_top", "front_z_bottom",
    "front_rotation", "front_skew",
    "rear_x_fore", "rear_x_hind", "rear_z_top", "rear_z_bottom",
    "rear_rotation", "rear_skew",
]

# ── All parameter names (freq + duty_factor + shape) ─────────────────────────
ALL_KEYS: list[str] = ["freq", "duty_factor"] + SHAPE_KEYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ellipsoid trajectory sim with configurable EllipsoidConfig parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Front legs (FL, FR) ──────────────────────────────────────────────────
    p.add_argument("--freq",   type=float, default=1.00,  help="Gait frequency (Hz)")
    p.add_argument("--front_x_fore",   type=float, default=0.14,  help="Front: forward reach (m)")
    p.add_argument("--front_x_hind",   type=float, default=0.08,  help="Front: rearward reach (m)")
    p.add_argument("--front_z_top",    type=float, default=0.10,  help="Front: swing height (m)")
    p.add_argument("--front_z_bottom", type=float, default=0.02,  help="Front: stance depth (m)")
    p.add_argument("--front_rotation", type=float, default=0.00,  help="Front: ellipse rotation (rad)")
    p.add_argument("--front_skew",     type=float, default=0.00,  help="Front: x-displacement (m)")
    # ── Rear legs (RL, RR) ────────────────────────────────────────────────────
    p.add_argument("--rear_x_fore",    type=float, default=0.06,  help="Rear:  forward reach (m)")
    p.add_argument("--rear_x_hind",    type=float, default=0.12,  help="Rear:  rearward reach (m)")
    p.add_argument("--rear_z_top",     type=float, default=0.10,  help="Rear:  swing height (m)")
    p.add_argument("--rear_z_bottom",  type=float, default=0.02,  help="Rear:  stance depth (m)")
    p.add_argument("--rear_rotation",  type=float, default=-0.00, help="Rear:  ellipse rotation (rad)")
    p.add_argument("--rear_skew",      type=float, default=0.00,  help="Rear:  x-displacement (m)")
    # ── Duty factor ───────────────────────────────────────────────────────────
    p.add_argument("--duty_factor",    type=float, default=0.27,  help="Fraction of cycle in stance (0–1)")
    # ── Sweep mode ────────────────────────────────────────────────────────────
    p.add_argument("--sweep",    action="store_true",       help="Run Bayesian optimisation sweep instead of a single sim")
    p.add_argument("--n_calls",  type=int, default=1000,    help="Total optimisation evaluations (phase 1 + phase 2)")
    p.add_argument("--output",   type=str, default=None,    help="CSV output path (default: sweep_results_<timestamp>.csv)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_and_controller(
    freq: float, cfg: EllipsoidConfig, duty_factor: float = 0.5
) -> tuple[MujocoSim, IKController, RobotInterface]:
    """Create a fresh MujocoSim + IKController for the given parameters."""
    robot_interface = RobotInterface(
        starting_state=State(mode=Mode.MOVING, gait=Gait.WALK, frequency=freq),
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
    """Build an EllipsoidConfig from a param dict (keys are SHAPE_KEYS)."""
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


def evaluate_params(param_dict: dict, verbose: bool = False) -> float:
    """
    Run one headless simulation and return the COT.
    Returns PENALTY_COT on any failure (robot fell, IK error, etc.).
    Suppresses MuJoCo/IK warnings unless verbose=True.
    """
    import io, contextlib
    freq = param_dict.get("freq", DEFAULTS["freq"])
    duty_factor = param_dict.get("duty_factor", DEFAULTS["duty_factor"])
    cfg = _dict_to_cfg(param_dict)
    try:
        sim, controller, _ = _build_sim_and_controller(freq, cfg, duty_factor=duty_factor)
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

def elip_traj_test():
    args = parse_args()

    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.WALK, frequency=args.freq), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=args.duty_factor)

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

def run_sweep(n_calls: int, output_path: str) -> None:
    """
    Two-phase Bayesian optimisation of WALK gait ellipsoid trajectory parameters.

    Phase 1 — Shape parameters only (freq and duty_factor fixed at defaults):
        Uses ~80% of the evaluation budget to explore the 12-dimensional shape space.
    Phase 2 — All 14 parameters (freq + duty_factor + shape):
        Uses ~20% of the budget, seeded with the phase 1 best, to also refine
        freq and duty_factor.  duty_factor is only adjusted here as a last resort.

    Early stopping: if the best COT improves < 1% over 50 consecutive trials
    in either phase, that phase terminates early.

    All trials are logged to a CSV file.
    """
    from skopt import gp_minimize
    from skopt.callbacks import EarlyStopper

    # ── Budget split ──────────────────────────────────────────────────────────
    # Phase 2 needs at least 10 calls to fit a GP; give phase 1 the rest.
    phase2_budget = max(10, int(n_calls * 0.2))
    phase1_budget = max(10, n_calls - phase2_budget)

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_file = open(output_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["phase", "trial", "cot"] + ALL_KEYS)
    trial_counter = [0]  # mutable counter shared across phases

    print(f"═══ WALK Gait COT Bayesian Optimisation ═══")
    print(f"  Total budget : {n_calls}  (phase 1: {phase1_budget}, phase 2: {phase2_budget})")
    print(f"  Output       : {output_path}")
    print()

    # ── Early stopping callback ───────────────────────────────────────────────
    # Stops if the best COT improves < 1% over the last 50 evaluations.
    class ConvergenceStopper(EarlyStopper):
        def __init__(self, window: int = 50, tol: float = 0.01):
            super().__init__()
            self.window = window
            self.tol = tol

        def _criterion(self, result):
            if len(result.func_vals) < self.window:
                return False
            recent = result.func_vals[-self.window:]
            import numpy as _np
            best_recent = _np.min(recent)
            best_before = _np.min(result.func_vals[:-self.window]) if len(result.func_vals) > self.window else best_recent
            # Stop if improvement over the window is < tol fraction
            if best_before <= 0:
                return False
            improvement = (best_before - best_recent) / abs(best_before)
            return improvement < self.tol

    stopper = ConvergenceStopper(window=50, tol=0.01)

    # ── Phase 1: shape params only, freq and duty_factor fixed at defaults ─────
    print(f"── Phase 1: Optimising 12 shape parameters (freq = {DEFAULTS['freq']} Hz, duty = {DEFAULTS['duty_factor']}) ──")
    shape_bounds = _make_bounds(SHAPE_KEYS)
    best_cot_so_far = [PENALTY_COT]
    fail_count = [0]
    success_count = [0]

    def phase1_objective(x: list[float]) -> float:
        trial_counter[0] += 1
        d = {k: v for k, v in zip(SHAPE_KEYS, x)}
        d["freq"] = DEFAULTS["freq"]              # fixed freq
        d["duty_factor"] = DEFAULTS["duty_factor"] # fixed duty_factor
        cot = evaluate_params(d)
        if cot >= PENALTY_COT:
            fail_count[0] += 1
        else:
            success_count[0] += 1
        if cot < best_cot_so_far[0]:
            best_cot_so_far[0] = cot
        # Log to CSV
        writer.writerow([1, trial_counter[0], f"{cot:.6f}"] + [f"{d[k]:.6f}" for k in ALL_KEYS])
        csv_file.flush()
        # Progress report every 50 trials
        if trial_counter[0] % 50 == 0:
            print(f"  [Phase 1] trial {trial_counter[0]:>4d}/{phase1_budget}  |  best COT = {best_cot_so_far[0]:.4f}  |  ok={success_count[0]} fail={fail_count[0]}")
        return cot

    # n_initial_points must be strictly less than n_calls for gp_minimize
    # Seed with default params so the GP starts from a known-good point,
    # then 15 random samples to explore, rest is GP-guided.
    n_initial = min(15, phase1_budget - 2)
    x0_phase1 = [DEFAULTS[k] for k in SHAPE_KEYS]
    result1 = gp_minimize(
        phase1_objective,
        shape_bounds,
        n_calls=phase1_budget,
        n_initial_points=n_initial,
        x0=x0_phase1,
        random_state=42,
        callback=[stopper],
        verbose=False,
    )

    # Extract phase 1 best as a dict
    phase1_best = {k: v for k, v in zip(SHAPE_KEYS, result1.x)}
    phase1_best["freq"] = DEFAULTS["freq"]
    phase1_best["duty_factor"] = DEFAULTS["duty_factor"]
    phase1_best_cot = result1.fun

    print(f"\n  Phase 1 complete — best COT = {phase1_best_cot:.4f}  (ok={success_count[0]}, fail={fail_count[0]})")
    print(f"  Trials used: {len(result1.func_vals)} / {phase1_budget}")
    print()

    # Reset counters for phase 2
    fail_count[0] = 0
    success_count[0] = 0

    # ── Phase 2: all 14 params (freq + duty_factor + shape), seeded around phase 1 best ──
    print("── Phase 2: Optimising 14 parameters (freq + duty_factor + shape) ──")
    # Freq bounds: ±15% around default, clamped to [0.5, 3.0]
    freq_bounds = [(max(0.5, DEFAULTS["freq"] * 0.85), min(3.0, DEFAULTS["freq"] * 1.15))]
    # Duty factor bounds: ±15% around default, clamped to [0.1, 0.8]
    duty_lo = max(0.1, DEFAULTS["duty_factor"] * 0.85)
    duty_hi = min(0.8, DEFAULTS["duty_factor"] * 1.15)
    duty_bounds = [(duty_lo, duty_hi)]
    all_bounds = freq_bounds + duty_bounds + _make_bounds(SHAPE_KEYS)

    # Starting point from phase 1 best
    x0 = [phase1_best[k] for k in ALL_KEYS]

    def phase2_objective(x: list[float]) -> float:
        trial_counter[0] += 1
        d = {k: v for k, v in zip(ALL_KEYS, x)}
        cot = evaluate_params(d)
        if cot >= PENALTY_COT:
            fail_count[0] += 1
        else:
            success_count[0] += 1
        if cot < best_cot_so_far[0]:
            best_cot_so_far[0] = cot
        writer.writerow([2, trial_counter[0], f"{cot:.6f}"] + [f"{d[k]:.6f}" for k in ALL_KEYS])
        csv_file.flush()
        if trial_counter[0] % 50 == 0:
            print(f"  [Phase 2] trial {trial_counter[0]:>4d}  |  best COT = {best_cot_so_far[0]:.4f}  |  ok={success_count[0]} fail={fail_count[0]}")
        return cot

    # n_initial_points must be strictly less than n_calls for gp_minimize
    n_initial2 = min(10, phase2_budget - 1)
    result2 = gp_minimize(
        phase2_objective,
        all_bounds,
        n_calls=phase2_budget,
        n_initial_points=n_initial2,
        x0=x0,
        random_state=42,
        callback=[stopper],
        verbose=False,
    )

    phase2_best = {k: v for k, v in zip(ALL_KEYS, result2.x)}
    phase2_best_cot = result2.fun

    print(f"\n  Phase 2 complete — best COT = {phase2_best_cot:.4f}  (ok={success_count[0]}, fail={fail_count[0]})")
    print(f"  Trials used: {len(result2.func_vals)} / {phase2_budget}")

    csv_file.close()

    # ── Pick overall best ─────────────────────────────────────────────────────
    if phase2_best_cot <= phase1_best_cot:
        best = phase2_best
        best_cot = phase2_best_cot
        best_phase = 2
    else:
        best = phase1_best
        best_cot = phase1_best_cot
        best_phase = 1

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("═══════════════════════════════════════════════════════════════")
    print(f"  BEST COT = {best_cot:.6f}  (from phase {best_phase})")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("Optimal parameters:")
    for k in ALL_KEYS:
        print(f"  --{k:<20s} {best[k]:.6f}")
    print()
    # Copy-pasteable CLI command
    cli_args = " ".join(f"--{k} {best[k]:.6f}" for k in ALL_KEYS)
    print(f"Re-run with:\n  python ai_fix_param.py {cli_args}")
    print()
    print(f"Full results saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.sweep:
        output = args.output or f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        run_sweep(n_calls=args.n_calls, output_path=output)
    else:
        cot = elip_traj_test()
        print("Cost of Transport:", cot)
    
    

if __name__ == "__main__":
    main()

