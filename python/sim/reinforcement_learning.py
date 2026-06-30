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
import json
import pickle
import argparse
import traceback
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
    """@brief Logs every evaluation to a CSV file for later analysis.

    Logs every evaluation to a CSV file for later analysis.

    CSV columns:
        trial, reward, cot, distance, velocity, avg_tilt, survived,
        freq, duty_factor, front_x_fore, ...  (all 14 param keys)
    """

    # Column order: meta → metrics → gait parameters
    META_COLS: list[str] = ["trial", "reward"]
    METRIC_COLS: list[str] = ["cot", "distance", "velocity", "avg_tilt", "survived"]

    def __init__(self, path: str) -> None:
        """@brief Open the CSV file and write the header row.

        @param path: Filesystem path for the CSV output file (overwritten if it exists).
        """
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
        """@brief Append one evaluation row.

        Append one evaluation row.

        @param trial_num: 1-based index of this evaluation.
        @param reward: Scalar reward returned by the environment for the trial.
        @param metrics: Metrics dict (cot, distance, velocity, avg_tilt, survived).
        @param params: Full parameter dict for the trial (keyed by ALL_PARAM_KEYS).
        """
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
        """@brief Flush and close the underlying file.

        Flush and close the underlying file.
        """
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
    """@brief Factory that returns a LoggingCallback class.

    Factory that returns a LoggingCallback class.

    We import stable-baselines3 lazily so the rest of the file can be used
    without SB3 installed (e.g. when only running CMA-ES).

    @param logger: TrialLogger that each episode is written to.
    @param checkpoint_dir: Directory where the best PPO model is saved.
    @param print_every: Print a progress line every this many episodes.
    @return A ``LoggingCallback`` subclass of SB3's ``BaseCallback``.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class LoggingCallback(BaseCallback):
        """@brief Logs each episode to CSV, saves best checkpoint, prints progress.

        Logs each episode to CSV, saves best checkpoint, prints progress.
        """

        def __init__(self, verbose: int = 0) -> None:
            """@brief Initialise per-run tracking state for the callback.

            @param verbose: SB3 verbosity level forwarded to ``BaseCallback``.
            """
            super().__init__(verbose)
            self.trial_count: int = 0
            self.best_reward: float = -np.inf
            self.best_metrics: dict[str, float] = {}
            self.best_params: dict[str, float] = {}
            # Rolling stats for progress printing
            self._recent_rewards: list[float] = []
            self._feasible_count: int = 0

        def _on_step(self) -> bool:
            """@brief Called after every env.step(). Since episodes are single-step,
            each call corresponds to one completed episode.

            Called after every env.step(). Since episodes are single-step,
            each call corresponds to one completed episode.

            @return True so SB3 continues training.
            """
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
    initial_params: dict[str, float] | None = None,
    curriculum_phase: int = 3,
) -> dict[str, float]:
    """@brief Train a PPO agent to optimise gait parameters.

    Train a PPO agent to optimise gait parameters.

    Each episode is a single env.step() that runs a full MuJoCo simulation
    and returns (obs, reward, terminated=True, ...).  PPO explores the 14-D
    normalised parameter space and learns to maximise the reward signal.

    Args:
        initial_params: Optional starting parameter values (one float per key
            in ``ALL_PARAM_KEYS``). When supplied, the env's bound window is
            centred on these values so PPO's exploration is anchored to the
            user's tuned ``.<GAIT>.ini`` instead of the hard-coded defaults.

    Returns:
        Dictionary of the best gait parameters found during training.

    @param gait: Gait pattern to optimise.
    @param total_timesteps: Number of PPO training timesteps (= episodes here).
    @param reward_weights: Weights for the multi-component reward.
    @param output_csv: CSV path that every episode is logged to.
    @param checkpoint_dir: Directory for the best/final PPO model checkpoints.
    @param resume_path: Optional path to a saved PPO model .zip to resume from.
    @param sim_length: Duration of the measurement phase per episode (seconds).
    @param warmup: Duration of the warmup phase per episode (seconds).
    @param verbose: If True, print MuJoCo/IK warnings during simulation.
    @param curriculum_phase: 1, 2, or 3 — gates which reward terms are summed.
    @return Dictionary of the best gait parameters found during training.
    """
    from stable_baselines3 import PPO

    # ── Environment ──────────────────────────────────────────────────────────
    env = GaitParamEnv(
        gait=gait,
        reward_weights=reward_weights,
        sim_length=sim_length,
        warmup=warmup,
        verbose=verbose,
        initial_params=initial_params,
        curriculum_phase=curriculum_phase,
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
    initial_params: dict[str, float] | None = None,
    cma_state_path: str | None = None,
    curriculum_phase: int = 3,
) -> dict[str, float]:
    """@brief Optimise gait parameters with CMA-ES.

    Optimise gait parameters with CMA-ES.

    CMA-ES operates in the normalised [-1, 1] action space of the Gymnasium
    environment.  It minimises a cost function, so we negate the reward.

    Args:
        initial_params: Optional starting parameter values (one float per key
            in ``ALL_PARAM_KEYS``). When supplied, ``env.get_default_action()``
            (used as CMA-ES's ``x0``) returns the normalised encoding of these
            values, so the search begins at the user's tuned ``.<GAIT>.ini``.
        cma_state_path: Optional path to a pickle file holding prior CMA-ES
            state ``{"initial_params": dict, "es": CMAEvolutionStrategy,
            "generations_run": int}``. When the file exists the search
            resumes from that state (covariance matrix, mean, sigma all
            preserved); otherwise a fresh strategy is built. The updated
            state is written back to this same path on successful completion.
            Bounds are anchored on the *original* ``initial_params`` from the
            saved state — not the one passed in this call — so the [-1, 1]
            normalised action space stays consistent across resumes.

    Returns:
        Dictionary of the best gait parameters found during optimisation.

    @param gait: Gait pattern to optimise.
    @param n_generations: Number of CMA-ES generations to run this call.
    @param reward_weights: Weights for the multi-component reward.
    @param output_csv: CSV path that every evaluation is logged to.
    @param population_size: CMA-ES population size (None = auto-select).
    @param sigma0: Initial CMA-ES step size in normalised space.
    @param sim_length: Duration of the measurement phase per episode (seconds).
    @param warmup: Duration of the warmup phase per episode (seconds).
    @param verbose: If True, print MuJoCo/IK warnings and CMA-ES internals.
    @param curriculum_phase: 1, 2, or 3 — gates which reward terms are summed.
    @return Dictionary of the best gait parameters found during optimisation.
    """
    import cma

    # ── Resume-from-checkpoint handling ──────────────────────────────────────
    # If a prior pickle exists, the bounds anchor (initial_params) is locked
    # to whatever the *first* run used so the [-1, 1] action space stays
    # mappable to the same physical parameter window across resumes. The
    # caller's ``initial_params`` is only used on a fresh run.
    saved: dict[str, Any] | None = None
    prior_generations = 0
    if cma_state_path and os.path.isfile(cma_state_path):
        with open(cma_state_path, "rb") as fp:
            saved = pickle.load(fp)
        prior_generations = int(saved.get("generations_run", 0))
        seed_params: dict[str, float] | None = saved.get("initial_params")
        print(f"[CMA-ES] resuming from {cma_state_path} "
              f"(prior generations: {prior_generations})")
    else:
        seed_params = initial_params

    # ── Environment ──────────────────────────────────────────────────────────
    env = GaitParamEnv(
        gait=gait,
        reward_weights=reward_weights,
        sim_length=sim_length,
        warmup=warmup,
        verbose=verbose,
        initial_params=seed_params,
        curriculum_phase=curriculum_phase,
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    trial_logger = TrialLogger(output_csv)

    # ── CMA-ES setup ─────────────────────────────────────────────────────────
    if saved is not None:
        # Re-use the exact strategy object that was running last time —
        # preserves mean (best estimate so far), covariance matrix, sigma,
        # and the internal generation counter.
        es: cma.CMAEvolutionStrategy = saved["es"]

        # Resume-recovery: a previously-saved strategy may already report a
        # non-empty ``stop()`` (e.g. ``tolfun`` triggered because every
        # candidate in the last generation hit ``_run_simulation``'s
        # exception path and returned identical penalty metrics → flat
        # fitness landscape). Without intervention the main loop's
        # ``while not es.stop() and …`` would short-circuit and the user's
        # newly-chosen ``--n-generations`` budget would be ignored, leaving
        # the pickle permanently bricked.
        #
        # Two recovery levels:
        #   1. Always raise ``maxiter`` to ``prior + n_generations`` and
        #      relax the tolerance-based criteria so the new budget is
        #      honoured.
        #   2. If the previous stop reason is *degenerate* (the strategy
        #      believes it has converged on a flat fitness surface),
        #      rebuild a fresh strategy anchored at the learned mean with
        #      the user-chosen ``sigma0`` — this preserves what was
        #      learned but escapes the trap.
        prev_stop = dict(es.stop())
        if prev_stop:
            print(f"[CMA-ES] previous state stopped on {prev_stop}; "
                  f"resetting termination criteria")
            # Criteria we always relax on resume so the user's budget wins.
            es.opts.set({
                "maxiter": prior_generations + n_generations,
                "tolfun": 0.0,
                "tolfunhist": 0.0,
                "tolx": 0.0,
                "tolstagnation": 10**9,
                "tolflatfitness": 10**9,
            })
            # ``cma.CMAEvolutionStrategy.stop()`` caches triggered conditions
            # in ``_stopdict``; once a condition fires it stays in the dict
            # forever, even after the threshold that produced it is relaxed
            # via ``opts.set``. Clearing the cache (private attribute, so
            # gated on hasattr for forward-compat with future cma releases)
            # is the only way to actually unstick the loop.
            if hasattr(es, "_stopdict"):
                es._stopdict.clear()
            degenerate = {
                "tolfun", "tolfunhist", "tolflatfitness",
                "noeffectaxis", "noeffectcoord", "tolx",
            }
            if degenerate & set(prev_stop.keys()):
                # Rebuild from the learned mean with fresh sigma0 to escape
                # the flat-fitness trap. Bounds and verbosity are
                # re-applied to match the fresh-start branch below.
                x_resume = np.asarray(es.mean, dtype=float).tolist()
                opts: dict[str, Any] = {"bounds": [-1, 1]}
                if population_size is not None:
                    opts["popsize"] = population_size
                if not verbose:
                    opts["verbose"] = -9
                es = cma.CMAEvolutionStrategy(x_resume, sigma0, opts)
                print(f"[CMA-ES] rebuilt strategy from learned mean with "
                      f"sigma0={sigma0} to escape degenerate stop")
    else:
        # Fresh start at the seed point in normalised space.
        x0 = env.get_default_action().tolist()
        opts: dict[str, Any] = {
            "bounds": [-1, 1],
            # Prevent early termination on flat fitness — same criteria
            # used in the resume-recovery path above.  The user's
            # ``--n-generations`` budget must always be honoured;
            # ``tolflatfitness`` / ``tolfun`` would otherwise trigger
            # after a single generation if every candidate hits the
            # penalty cliff (identical reward → flat fitness landscape).
            "tolfun": 0,
            "tolfunhist": 0,
            "tolx": 0,
            "tolflatfitness": 10**9,
            "tolstagnation": 10**9,
        }
        if population_size is not None:
            opts["popsize"] = population_size
        # Suppress CMA-ES internal printing unless verbose
        if not verbose:
            opts["verbose"] = -9

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # ── Tracking ─────────────────────────────────────────────────────────────
    best_reward: float = -np.inf
    best_metrics: dict[str, float] = {}
    # Pre-seed best_params with the initial guess so a run that finds no
    # improvement (e.g. CMA-ES already converged on resume) still returns a
    # usable parameter dict instead of an empty one. The first improving
    # candidate inside ``evaluate`` overwrites this.
    best_params: dict[str, float] = dict(seed_params) if seed_params else {}
    trial_count: int = 0
    feasible_count: int = 0

    # ── Objective: CMA-ES minimises, so we negate the reward ─────────────────
    def evaluate(action: np.ndarray) -> float:
        """@brief Run one simulation and return negative reward (for minimisation).

        Run one simulation and return negative reward (for minimisation).

        @param action: Candidate action vector in the normalised [-1, 1] space.
        @return Negated reward (CMA-ES minimises, so lower is better).
        """
        nonlocal trial_count, best_reward, best_metrics, best_params, feasible_count
        # Diagnostic wrapper: surfaces any exception that escapes
        # ``_run_simulation``'s own try/except (e.g. raised inside
        # ``env.reset()`` or in observation/reward post-processing).
        # Without this, MuJoCo's C-side callback wrapper prints
        # "ERROR: Python exception raised" but the actual Python traceback
        # is lost, making the search appear to die silently.
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            # Reset the env for the next evaluation
            env.reset()
        except Exception:
            traceback.print_exc()
            raise

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

    # ── Persist CMA-ES state for resume on the next click ────────────────────
    # The strategy object carries the full search state — mean, covariance
    # matrix, step size, internal counter — so the next invocation that
    # passes the same ``cma_state_path`` continues evolving instead of
    # restarting cold from x0.
    #
    # IMPORTANT: only overwrite the on-disk pickle if at least one
    # generation actually ran this call. A 0-iteration run (e.g. the
    # caller asked for ``--n-generations 0`` or the loop short-circuited
    # before the resume-recovery logic was added) must never replace a
    # healthy checkpoint with one whose strategy has already terminated —
    # that is the bug that originally bricked users' AMBLE checkpoint.
    if cma_state_path and generation > 0:
        os.makedirs(os.path.dirname(cma_state_path) or ".", exist_ok=True)
        new_total = prior_generations + generation
        with open(cma_state_path, "wb") as fp:
            pickle.dump(
                {
                    "initial_params": seed_params,
                    "es": es,
                    "generations_run": new_total,
                },
                fp,
            )
        print(f"[CMA-ES] saved state to {cma_state_path} "
              f"(total generations: {new_total})")
    elif cma_state_path:
        print(f"[CMA-ES] no generations ran this call; "
              f"leaving {cma_state_path} untouched")

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
    """@brief Print a summary of the best result and a copy-pasteable re-run command.

    Print a summary of the best result and a copy-pasteable re-run command.

    @param best_params: Best gait parameter dict found by the optimiser.
    @param best_reward: Reward of the best candidate.
    @param best_metrics: Metrics dict for the best candidate.
    @param algo: Algorithm label used in the report header ("PPO" / "CMA-ES").
    @param gait_name: Gait name used in the copy-pasteable re-run command.
    """
    print("\n" + "=" * 72)
    print(f"  {algo} — Best Result")
    print("=" * 72)
    def _fmt(val, fmt=".4f"):
        """@brief Format a metric value, treating None as NaN.

        Format a metric value, treating None as NaN.

        @param val: Value to format (may be None).
        @param fmt: Python format spec applied when ``val`` is not None.
        @return Formatted string, or "nan" when ``val`` is None.
        """
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
    """@brief Parse command-line arguments for the RL trainer.

    Parse command-line arguments for the RL trainer.

    @return Parsed argparse.Namespace covering algorithm selection, gait,
        PPO/CMA-ES hyperparameters, reward weights, curriculum phase, and I/O paths.
    """
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
    # Defaults mirror ``RewardWeights`` in gait_env.py and encode the
    # priority order gait >> velocity > CoT > stability requested by the
    # user. Override per run from the CLI when retuning.
    p.add_argument("--w-gait", type=float, default=3.0,
                   help="Reward weight for footfall-pattern matching (PRIMARY).")
    p.add_argument("--w-vel", type=float, default=1.0,
                   help="Reward weight for forward velocity.")
    p.add_argument("--w-cot", type=float, default=0.2,
                   help="Reward weight for cost of transport.")
    p.add_argument("--w-tilt", type=float, default=0.1,
                   help="Reward weight for body tilt (|roll| + |pitch|).")
    p.add_argument("--w-ang-vel", type=float, default=0.05,
                   help="Reward weight for base angular-velocity penalty.")
    p.add_argument("--w-slip", type=float, default=0.1,
                   help="Reward weight for stance-phase foot-slip penalty (Phase 3 only).")
    p.add_argument("--w-clearance", type=float, default=0.5,
                   help="Reward weight for swing-phase ground-clearance penalty (Phase 3 only).")
    p.add_argument("--w-fall", type=float, default=10.0,
                   help="Reward weight for the hard fall / backward-motion / invalid-CoT cliff.")

    # ── Curriculum ──────────────────────────────────────────────────────────
    # Gates which reward terms are summed. Phase 1 trains footfall pattern
    # only, Phase 2 adds forward velocity, Phase 3 enables the full reward.
    p.add_argument("--curriculum-phase", type=int, default=3, choices=[1, 2, 3],
                   help="1=gait only, 2=gait+velocity, 3=full reward (default).")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--output", type=str, default=None,
                   help="CSV output path (auto-generated if omitted).")
    p.add_argument("--checkpoint-dir", type=str, default="rl_checkpoints",
                   help="Directory for PPO model checkpoints.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose simulation output.")

    # ── Initial-guess / structured I/O (used by the GUI subprocess flow) ─────
    p.add_argument("--initial-params-json", type=str, default=None,
                   help="Path to a JSON file containing a {key: float} dict "
                        "of starting parameter values (keys = ALL_PARAM_KEYS, "
                        "i.e. 'freq', 'duty_factor', and the 12 ellipsoid "
                        "fields). Search bounds are centred on these values.")
    p.add_argument("--output-json", type=str, default=None,
                   help="Path to write the best-params dict to as JSON on "
                        "successful completion. Used by the GUI to read back "
                        "the optimisation result.")
    p.add_argument("--cma-state-path", type=str, default=None,
                   help="Path to the pickled CMA-ES checkpoint. If the file "
                        "exists at start, the strategy resumes from it; "
                        "otherwise a fresh strategy is built. The updated "
                        "state is written back here on successful "
                        "completion. CMA-ES only.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """@brief CLI entry point: build the run config and dispatch to PPO or CMA-ES.

    Parses arguments, resolves the gait enum, assembles the reward weights and
    output paths, loads any initial-guess JSON, runs the selected optimiser, and
    optionally writes the best-params dict back out as JSON for the GUI flow.
    """
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
        w_gait=args.w_gait,
        w_vel=args.w_vel,
        w_cot=args.w_cot,
        w_tilt=args.w_tilt,
        w_ang_vel=args.w_ang_vel,
        w_slip=args.w_slip,
        w_clearance=args.w_clearance,
        w_fall=args.w_fall,
    )

    # ── Auto-generate output CSV path if not provided ────────────────────────
    # Default destination is ``<repo>/bio_inspired_muscle/data/`` — a
    # dedicated folder one level above the python package, sibling of
    # ``python/``. Keeps the python source tree clean and groups every
    # run's CSV in one predictable place. Bare-filename ``--output``
    # values (no directory separator) get the same treatment so the
    # GUI flow lands here too; absolute or explicitly-relative paths
    # are passed through untouched.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ``__file__`` -> .../bio_inspired_muscle/python/sim/reinforcement_learning.py
    # parents[2]  -> .../bio_inspired_muscle
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )
    if args.output is None:
        os.makedirs(data_dir, exist_ok=True)
        output_csv = os.path.join(
            data_dir,
            f"rl_results_{args.algo}_{args.gait.upper()}_{timestamp}.csv",
        )
    elif os.path.dirname(args.output) == "":
        os.makedirs(data_dir, exist_ok=True)
        output_csv = os.path.join(data_dir, args.output)
    else:
        output_csv = args.output

    # ── Optional initial-guess JSON (used by the GUI subprocess flow) ────────
    # We don't validate the keys here; ``GaitParamEnv`` raises a clear KeyError
    # if a required ALL_PARAM_KEYS entry is missing when bounds are computed.
    initial_params: dict[str, float] | None = None
    if args.initial_params_json:
        with open(args.initial_params_json, "r") as fp:
            raw = json.load(fp)
        initial_params = {str(k): float(v) for k, v in raw.items()}
        print(f"  Seed    : {args.initial_params_json} (initial guess)")

    print(f"{'─' * 72}")
    print(f"  Gait Parameter Optimizer — {args.algo.upper()}")
    print(f"  Gait    : {gait.name}")
    print(f"  Phase   : {args.curriculum_phase}  "
          f"(1=gait only, 2=gait+vel, 3=full)")
    print(f"  Weights : gait={args.w_gait}, vel={args.w_vel}, "
          f"cot={args.w_cot}, tilt={args.w_tilt}, ang_vel={args.w_ang_vel}, "
          f"slip={args.w_slip}, clearance={args.w_clearance}, fall={args.w_fall}")
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
            initial_params=initial_params,
            curriculum_phase=args.curriculum_phase,
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
            initial_params=initial_params,
            cma_state_path=args.cma_state_path,
            curriculum_phase=args.curriculum_phase,
        )
    else:
        # Should not reach here due to argparse choices, but just in case
        print(f"ERROR: Unknown algorithm '{args.algo}'.")
        sys.exit(1)

    # Store gait name for any future post-processing
    best_params["_gait"] = gait.name

    # ── Structured result for the GUI subprocess flow ────────────────────────
    # Dump the best-params dict as JSON so the parent process (the Qt
    # ReinforcementWorker) can read it without having to parse stdout.
    if args.output_json:
        with open(args.output_json, "w") as fp:
            json.dump(best_params, fp, indent=2)
        print(f"[RL] Wrote best-params JSON to {args.output_json}")


if __name__ == "__main__":
    main()
