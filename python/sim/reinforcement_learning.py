"""
Reinforcement learning gait parameter optimizer.

Supports two algorithms:
  - PPO (Proximal Policy Optimization) via stable-baselines3
  - CMA-ES (Covariance Matrix Adaptation Evolution Strategy) via cma

Usage:
    # PPO training (500 episodes)
    python reinforcement_learning.py --algo ppo --gait TROT --total-timesteps 500

    # CMA-ES optimization (200 generations)
    python reinforcement_learning.py --algo cma --gait TROT --n-generations 200

    # Custom reward weights
    python reinforcement_learning.py --algo ppo --w-cot 1.5 --w-vel 0.3 --w-tilt 3.0

    # Resume PPO training from checkpoint
    python reinforcement_learning.py --algo ppo --resume checkpoint_dir/best_model.zip
"""

import os
import sys
import csv
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# ── Cross-module imports using the sys.path.insert pattern ────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Gait
from gait_env import GaitParamEnv, RewardWeights, ALL_PARAM_KEYS, SHAPE_KEYS, default_params


# ─────────────────────────────────────────────────────────────────────────────
# CSV Trial Logger
# ─────────────────────────────────────────────────────────────────────────────

class TrialLogger:
    """Logs every evaluation to a CSV file for later analysis.

    CSV columns:
        trial, reward, cot, distance, velocity, avg_tilt, survived,
        freq, duty_factor, front_x_fore, ...  (all 14 param keys)
    """

    # Column order: meta → metrics → gait parameters
    META_COLS: list[str] = ["trial", "reward"]
    METRIC_COLS: list[str] = ["cot", "distance", "velocity", "avg_tilt", "survived"]

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        # Write header row
        header = self.META_COLS + self.METRIC_COLS + list(ALL_PARAM_KEYS)
        self._writer.writerow(header)
        self._file.flush()

    def log(
        self,
        trial_num: int,
        reward: float,
        metrics: dict[str, float],
        params: dict[str, float],
    ) -> None:
        """Append one evaluation row."""
        row: list[Any] = [trial_num, f"{reward:.6f}"]
        # Metrics in fixed order — handle None values (e.g. cot when robot fell)
        for col in self.METRIC_COLS:
            val = metrics.get(col, 0.0)
            row.append(f"{val:.6f}" if val is not None else "")
        # Parameters in ALL_PARAM_KEYS order
        for key in ALL_PARAM_KEYS:
            row.append(f"{params.get(key, 0.0):.6f}")
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# SB3 Logging Callback (PPO)
# ─────────────────────────────────────────────────────────────────────────────

def _make_logging_callback(
    logger: TrialLogger,
    checkpoint_dir: str,
    print_every: int = 10,
):
    """Factory that returns a LoggingCallback class.

    We import stable-baselines3 lazily so the rest of the file can be used
    without SB3 installed (e.g. when only running CMA-ES).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class LoggingCallback(BaseCallback):
        """Logs each episode to CSV, saves best checkpoint, prints progress."""

        def __init__(self, verbose: int = 0) -> None:
            super().__init__(verbose)
            self.trial_count: int = 0
            self.best_reward: float = -np.inf
            self.best_metrics: dict[str, float] = {}
            self.best_params: dict[str, float] = {}
            # Rolling stats for progress printing
            self._recent_rewards: list[float] = []
            self._feasible_count: int = 0

        def _on_step(self) -> bool:
            """Called after every env.step(). Since episodes are single-step,
            each call corresponds to one completed episode."""
            infos = self.locals.get("infos", [])
            if not infos:
                return True

            info = infos[0]
            metrics = info.get("metrics", {})
            params = info.get("params", {})
            # The reward returned by the environment for this step
            reward = self.locals.get("rewards", [0.0])[0]

            self.trial_count += 1
            logger.log(self.trial_count, float(reward), metrics, params)
            self._recent_rewards.append(float(reward))
            if metrics.get("survived", 0.0) > 0.5:
                self._feasible_count += 1

            # Track best
            if float(reward) > self.best_reward:
                self.best_reward = float(reward)
                self.best_metrics = dict(metrics)
                self.best_params = dict(params)
                # Save best model checkpoint
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.model.save(os.path.join(checkpoint_dir, "best_model"))

            # Print progress every N episodes
            if self.trial_count % print_every == 0:
                avg_r = np.mean(self._recent_rewards[-print_every:])
                feas_rate = self._feasible_count / self.trial_count
                best_cot = self.best_metrics.get("cot") or float("nan")
                print(
                    f"  [PPO] trial {self.trial_count:>5d} | "
                    f"avg_R(last {print_every})={avg_r:+.3f} | "
                    f"best_R={self.best_reward:+.3f} | "
                    f"best_COT={best_cot:.3f} | "
                    f"feasible={feas_rate:.0%}"
                )

            return True

    return LoggingCallback


# ─────────────────────────────────────────────────────────────────────────────
# PPO Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    gait: Gait,
    total_timesteps: int,
    reward_weights: RewardWeights,
    output_csv: str,
    checkpoint_dir: str,
    resume_path: str | None = None,
    sim_length: float = 5.0,
    warmup: float = 2.0,
    verbose: bool = False,
) -> dict[str, float]:
    """Train a PPO agent to optimise gait parameters.

    Each episode is a single env.step() that runs a full MuJoCo simulation
    and returns (obs, reward, terminated=True, ...).  PPO explores the 14-D
    normalised parameter space and learns to maximise the reward signal.

    Returns:
        Dictionary of the best gait parameters found during training.
    """
    from stable_baselines3 import PPO

    # ── Environment ──────────────────────────────────────────────────────────
    env = GaitParamEnv(
        gait=gait,
        reward_weights=reward_weights,
        sim_length=sim_length,
        warmup=warmup,
        verbose=verbose,
    )

    # ── Logger & callback ────────────────────────────────────────────────────
    trial_logger = TrialLogger(output_csv)
    CallbackCls = _make_logging_callback(trial_logger, checkpoint_dir)
    callback = CallbackCls(verbose=1 if verbose else 0)

    # ── Model ────────────────────────────────────────────────────────────────
    if resume_path is not None:
        print(f"[PPO] Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        # Each episode is a single step (env returns terminated=True immediately).
        # SB3 requires batch_size > 1 and n_steps >= batch_size, so we buffer
        # 8 episodes before each PPO update — a reasonable mini-batch for this
        # low-dimensional parameter search.
        # Use CPU — PPO with MlpPolicy has poor GPU utilisation.
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,        # encourage exploration in parameter space
            device="cpu",
            verbose=1,
        )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"[PPO] Starting training — {total_timesteps} timesteps, gait={gait}")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # ── Save final model ─────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"[PPO] Final model saved to {final_path}.zip")

    trial_logger.close()

    # ── Report ───────────────────────────────────────────────────────────────
    print_results(callback.best_params, callback.best_reward,
                  callback.best_metrics, algo="PPO", gait_name=gait.name)
    return callback.best_params


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES Training
# ─────────────────────────────────────────────────────────────────────────────

def train_cma(
    gait: Gait,
    n_generations: int,
    reward_weights: RewardWeights,
    output_csv: str,
    population_size: int | None = None,
    sigma0: float = 0.3,
    sim_length: float = 5.0,
    warmup: float = 2.0,
    verbose: bool = False,
) -> dict[str, float]:
    """Optimise gait parameters with CMA-ES.

    CMA-ES operates in the normalised [-1, 1] action space of the Gymnasium
    environment.  It minimises a cost function, so we negate the reward.

    Returns:
        Dictionary of the best gait parameters found during optimisation.
    """
    import cma

    # ── Environment ──────────────────────────────────────────────────────────
    env = GaitParamEnv(
        gait=gait,
        reward_weights=reward_weights,
        sim_length=sim_length,
        warmup=warmup,
        verbose=verbose,
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    trial_logger = TrialLogger(output_csv)

    # ── CMA-ES setup ─────────────────────────────────────────────────────────
    # Start search at the default parameter set (centre of normalised space)
    x0 = env.get_default_action().tolist()
    opts: dict[str, Any] = {"bounds": [-1, 1]}
    if population_size is not None:
        opts["popsize"] = population_size
    # Suppress CMA-ES internal printing unless verbose
    if not verbose:
        opts["verbose"] = -9

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # ── Tracking ─────────────────────────────────────────────────────────────
    best_reward: float = -np.inf
    best_metrics: dict[str, float] = {}
    best_params: dict[str, float] = {}
    trial_count: int = 0
    feasible_count: int = 0

    # ── Objective: CMA-ES minimises, so we negate the reward ─────────────────
    def evaluate(action: np.ndarray) -> float:
        """Run one simulation and return negative reward (for minimisation)."""
        nonlocal trial_count, best_reward, best_metrics, best_params, feasible_count
        obs, reward, terminated, truncated, info = env.step(action)
        # Reset the env for the next evaluation
        env.reset()

        metrics = info.get("metrics", {})
        params = info.get("params", {})
        trial_count += 1
        trial_logger.log(trial_count, reward, metrics, params)

        if metrics.get("survived", 0.0) > 0.5:
            feasible_count += 1

        if reward > best_reward:
            best_reward = reward
            best_metrics = dict(metrics)
            best_params = dict(params)

        return -reward  # CMA-ES minimises

    # ── Optimisation loop ────────────────────────────────────────────────────
    print(f"[CMA-ES] Starting optimisation — {n_generations} generations, "
          f"sigma0={sigma0}, gait={gait}")

    generation = 0
    while not es.stop() and generation < n_generations:
        solutions = es.ask()
        fitnesses = [evaluate(np.asarray(x, dtype=np.float32)) for x in solutions]
        es.tell(solutions, fitnesses)
        generation += 1

        # Progress print every generation
        gen_best = -min(fitnesses)
        feas_rate = feasible_count / max(trial_count, 1)
        best_cot = best_metrics.get("cot") or float("nan")
        if generation % 10 == 0 or generation == 1:
            print(
                f"  [CMA-ES] gen {generation:>4d} | "
                f"gen_best_R={gen_best:+.3f} | "
                f"overall_best_R={best_reward:+.3f} | "
                f"best_COT={best_cot:.3f} | "
                f"evals={trial_count} | "
                f"feasible={feas_rate:.0%}"
            )

    trial_logger.close()

    # ── Report ───────────────────────────────────────────────────────────────
    print_results(best_params, best_reward, best_metrics,
                  algo="CMA-ES", gait_name=gait.name)
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# Results Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    best_params: dict[str, float],
    best_reward: float,
    best_metrics: dict[str, float],
    algo: str,
    gait_name: str = "TROT",
) -> None:
    """Print a summary of the best result and a copy-pasteable re-run command."""
    print("\n" + "=" * 72)
    print(f"  {algo} — Best Result")
    print("=" * 72)
    def _fmt(val, fmt=".4f"):
        """Format a metric value, treating None as NaN."""
        return f"{val:{fmt}}" if val is not None else "nan"

    print(f"  Reward   : {best_reward:+.4f}")
    print(f"  COT      : {_fmt(best_metrics.get('cot'))}")
    print(f"  Distance : {_fmt(best_metrics.get('distance', 0.0))} m")
    print(f"  Velocity : {_fmt(best_metrics.get('velocity', 0.0))} m/s")
    print(f"  Avg Tilt : {_fmt(best_metrics.get('avg_tilt', 0.0))} rad")
    print(f"  Survived : {_fmt(best_metrics.get('survived', 0.0), '.0f')}")
    print("-" * 72)
    print("  Parameters:")
    for key in ALL_PARAM_KEYS:
        val = best_params.get(key, 0.0)
        print(f"    {key:<20s} = {val:.6f}")

    # ── Copy-pasteable EllipsoidConfig for code insertion ────────────────────
    print("-" * 72)
    print("  Copy-pasteable EllipsoidConfig:\n")
    print("EllipsoidConfig(")
    front_keys = [k for k in SHAPE_KEYS if k.startswith("front_")]
    rear_keys = [k for k in SHAPE_KEYS if k.startswith("rear_")]
    for key in front_keys:
        val = best_params.get(key, 0.0)
        print(f"    {key:<14s} = {val:.2f},")
    print()
    for key in rear_keys:
        val = best_params.get(key, 0.0)
        print(f"    {key:<14s} = {val:.2f},")
    print(")")
    print(f"\nfreq = {best_params.get('freq', 0.0):.2f}")
    print(f"duty_factor = {best_params.get('duty_factor', 0.0):.2f}")

    # ── Copy-pasteable command for ai_fix_param.py ───────────────────────────
    param_args = " ".join(
        f"--{key} {best_params.get(key, 0.0):.6f}" for key in ALL_PARAM_KEYS
    )
    print("-" * 72)
    print("  Re-run best gait with:")
    print(f"    python sim/ai_fix_param.py --gait {gait_name} {param_args}")
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the RL trainer."""
    p = argparse.ArgumentParser(
        description="Reinforcement learning gait parameter optimizer (PPO / CMA-ES).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Algorithm selection ───────────────────────────────────────────────────
    p.add_argument("--algo", type=str, choices=["ppo", "cma"], required=True,
                   help="Optimisation algorithm to use.")
    p.add_argument("--gait", type=str, default="TROT",
                   help="Gait name (WALK, TROT, AMBLE, CANTER, GALLOP).")

    # ── PPO-specific ─────────────────────────────────────────────────────────
    p.add_argument("--total-timesteps", type=int, default=500,
                   help="Number of PPO training timesteps (= episodes for single-step env).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a saved PPO model .zip to resume training from.")

    # ── CMA-ES-specific ──────────────────────────────────────────────────────
    p.add_argument("--n-generations", type=int, default=200,
                   help="Number of CMA-ES generations.")
    p.add_argument("--population-size", type=int, default=None,
                   help="CMA-ES population size (None = auto-select).")
    p.add_argument("--sigma0", type=float, default=0.3,
                   help="CMA-ES initial step size in normalised space.")

    # ── Simulation ───────────────────────────────────────────────────────────
    p.add_argument("--sim-length", type=float, default=5.0,
                   help="Simulation length per episode (seconds).")
    p.add_argument("--warmup", type=float, default=2.0,
                   help="Warmup time before metrics collection (seconds).")

    # ── Reward weights ───────────────────────────────────────────────────────
    p.add_argument("--w-cot", type=float, default=1.0,
                   help="Reward weight for cost of transport.")
    p.add_argument("--w-vel", type=float, default=0.5,
                   help="Reward weight for forward velocity.")
    p.add_argument("--w-tilt", type=float, default=2.0,
                   help="Reward weight for body tilt penalty.")
    p.add_argument("--w-fall", type=float, default=10.0,
                   help="Reward weight for fall penalty.")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--output", type=str, default=None,
                   help="CSV output path (auto-generated if omitted).")
    p.add_argument("--checkpoint-dir", type=str, default="rl_checkpoints",
                   help="Directory for PPO model checkpoints.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose simulation output.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve gait enum ────────────────────────────────────────────────────
    try:
        gait = Gait[args.gait.upper()]
    except KeyError:
        print(f"ERROR: Unknown gait '{args.gait}'. "
              f"Available: {list(Gait._registry.keys())}")
        sys.exit(1)

    # ── Build reward weights from CLI args ───────────────────────────────────
    reward_weights = RewardWeights(
        w_cot=args.w_cot,
        w_vel=args.w_vel,
        w_tilt=args.w_tilt,
        w_fall=args.w_fall,
    )

    # ── Auto-generate output CSV path if not provided ────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_csv = f"rl_results_{args.algo}_{args.gait.upper()}_{timestamp}.csv"
    else:
        output_csv = args.output

    print(f"{'─' * 72}")
    print(f"  Gait Parameter Optimizer — {args.algo.upper()}")
    print(f"  Gait    : {gait.name}")
    print(f"  Weights : COT={args.w_cot}, vel={args.w_vel}, "
          f"tilt={args.w_tilt}, fall={args.w_fall}")
    print(f"  Output  : {output_csv}")
    print(f"{'─' * 72}\n")

    # ── Dispatch to the chosen algorithm ─────────────────────────────────────
    if args.algo == "ppo":
        best_params = train_ppo(
            gait=gait,
            total_timesteps=args.total_timesteps,
            reward_weights=reward_weights,
            output_csv=output_csv,
            checkpoint_dir=args.checkpoint_dir,
            resume_path=args.resume,
            sim_length=args.sim_length,
            warmup=args.warmup,
            verbose=args.verbose,
        )
    elif args.algo == "cma":
        best_params = train_cma(
            gait=gait,
            n_generations=args.n_generations,
            reward_weights=reward_weights,
            output_csv=output_csv,
            population_size=args.population_size,
            sigma0=args.sigma0,
            sim_length=args.sim_length,
            warmup=args.warmup,
            verbose=args.verbose,
        )
    else:
        # Should not reach here due to argparse choices, but just in case
        print(f"ERROR: Unknown algorithm '{args.algo}'.")
        sys.exit(1)

    # Store gait name for any future post-processing
    best_params["_gait"] = gait.name


if __name__ == "__main__":
    main()
