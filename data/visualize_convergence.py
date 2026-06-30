#!/usr/bin/env python3
"""Visualize CMA-ES parameter convergence from `rl_results_*.csv` sweep files.

Each CSV row is one trial. The non-metric columns are the parameters CMA-ES is
tuning; the script plots every parameter on a grid of subplots so you can see
the full sample cloud per trial *and* the best-so-far value the optimizer is
converging on.

Usage
-----
    # Plot every CSV next to this script, one figure per file:
    python visualize_convergence.py

    # Overlay all runs of each gait (one figure per gait label):
    python visualize_convergence.py --group

    # Specific files, save to disk without opening windows:
    python visualize_convergence.py rl_results_cma_WALK_*.csv \\
        --save plots --no-show
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Outcome columns — everything else is treated as a tunable parameter.
METRIC_COLS = {"trial", "reward", "cot", "distance", "velocity", "avg_tilt", "survived"}


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    """@brief Load a CSV into {column_name: float64 array}. Empty cells become NaN.

    Load a CSV into {column_name: float64 array}. Empty cells become NaN.

    @param path: Path to the CSV file to load.
    @return A dict mapping each column name to its float64 numpy array.
    """
    # genfromtxt handles missing values cleanly (the CSVs leave `cot` blank
    # when the trial died and divisions blew up). `ndmin=1` keeps single-row
    # files from collapsing into 0-d records.
    arr = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64))
    return {name: arr[name] for name in arr.dtype.names}

# Pull "AMBLE" / "WALK" / etc. out of "rl_results_cma_WALK_20260507_073815.csv".
_GAIT_RE = re.compile(r"rl_results_cma_([A-Z]+)_")


def _gait_label(path: Path) -> str:
    """@brief Extract the gait name (e.g. "WALK") from a result CSV filename.

    @param path: Path to a result CSV file.
    @return The matched gait label, or the file stem if no gait is found.
    """
    m = _GAIT_RE.search(path.name)
    return m.group(1) if m else path.stem


def _best_so_far_index(reward: np.ndarray) -> np.ndarray:
    """@brief Return, for each row, the index of the highest finite reward seen so far.

    Return, for each row, the index of the highest finite reward seen so far.

    @param reward: Array of per-trial reward values (may contain NaN/inf).
    @return An integer array where entry i is the index of the best reward
        observed in rows 0..i (used to draw the best-so-far convergence line).
    """
    out = np.zeros(len(reward), dtype=int)
    best_val = -np.inf
    best_idx = 0
    for i, r in enumerate(reward):
        if np.isfinite(r) and r > best_val:
            best_val = r
            best_idx = i
        out[i] = best_idx
    return out


def _grid(n: int) -> tuple[int, int]:
    """@brief Choose a (rows, cols) subplot grid that fits n panels.

    @param n: Number of subplots to lay out.
    @return A (rows, cols) tuple for the subplot grid.
    """
    cols = 4 if n > 6 else min(n, 3)
    rows = (n + cols - 1) // cols
    return rows, cols


def plot_single(csv_path: Path, save_dir: Path | None, show: bool) -> None:
    """@brief One figure for one CSV: scatter of trials + best-so-far convergence line.

    One figure for one CSV: scatter of trials + best-so-far convergence line.

    @param csv_path: Path to the result CSV to plot.
    @param save_dir: Directory to save the PNG to, or None to skip saving.
    @param show: Whether to open an interactive window.
    """
    df = _load_csv(csv_path)
    param_cols = [c for c in df if c not in METRIC_COLS]
    if not param_cols or "trial" not in df or "reward" not in df:
        print(f"Skipping {csv_path.name}: missing trial/reward or no parameters.")
        return

    trial = df["trial"]
    best_idx = _best_so_far_index(df["reward"])

    rows, cols = _grid(len(param_cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, param_cols):
        values = df[col]
        ax.scatter(trial, values, s=8, alpha=0.30, color="C0", label="sample")
        ax.plot(trial, values[best_idx], color="C3", lw=1.6, label="best-so-far")
        final = values[best_idx[-1]]
        ax.axhline(final, ls="--", color="C3", alpha=0.5)
        ax.set_title(f"{col}  →  {final:.4f}", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(param_cols):]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{csv_path.name}   ({len(trial)} trials, gait={_gait_label(csv_path)})", fontsize=11)
    fig.supxlabel("trial")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        out = save_dir / f"{csv_path.stem}.png"
        fig.savefig(out, dpi=120)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_group(gait: str, paths: list[Path], save_dir: Path | None, show: bool) -> None:
    """@brief One figure per gait, overlaying best-so-far curves from every run.

    One figure per gait, overlaying best-so-far curves from every run.

    @param gait: The gait label these runs belong to (used in the title).
    @param paths: Result CSV files for this gait to overlay.
    @param save_dir: Directory to save the PNG to, or None to skip saving.
    @param show: Whether to open an interactive window.
    """
    frames = [(p, _load_csv(p)) for p in paths]
    # Use the first frame's parameter columns as the canonical set.
    param_cols = [c for c in frames[0][1] if c not in METRIC_COLS]
    if not param_cols:
        return

    rows, cols = _grid(len(param_cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    cmap = plt.get_cmap("tab10")
    for ax, col in zip(axes, param_cols):
        finals: list[float] = []
        for i, (path, df) in enumerate(frames):
            if col not in df:
                continue
            trial = df["trial"]
            values = df[col]
            best_idx = _best_so_far_index(df["reward"])
            color = cmap(i % cmap.N)
            ax.plot(trial, values[best_idx], color=color, lw=1.4,
                    label=path.stem.replace("rl_results_cma_", ""))
            finals.append(values[best_idx[-1]])
        if finals:
            ax.set_title(f"{col}  →  {np.mean(finals):.4f} ± {np.std(finals):.4f}", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(param_cols):]:
        ax.axis("off")

    # Single shared legend, placed in the unused corner if there is one.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_ax = axes[len(param_cols)] if len(param_cols) < len(axes) else axes[-1]
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, fontsize=7, loc="center")

    fig.suptitle(f"gait={gait}   ({len(paths)} runs)", fontsize=11)
    fig.supxlabel("trial")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        out = save_dir / f"convergence_{gait}.png"
        fig.savefig(out, dpi=120)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """@brief Parse CLI arguments and render convergence plots per file or per gait."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path,
                        help="CSV files (default: all rl_results_*.csv next to this script).")
    parser.add_argument("--group", action="store_true",
                        help="Overlay all runs of the same gait on one figure per gait.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Directory to save PNGs to (created if missing).")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive windows (useful with --save).")
    args = parser.parse_args()

    if args.files:
        files = [p for p in args.files if p.is_file()]
    else:
        here = Path(__file__).resolve().parent
        files = sorted(here.glob("rl_results_*.csv"))

    if not files:
        print("No CSV files found.")
        return

    save_dir = args.save
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    if args.group:
        groups: dict[str, list[Path]] = {}
        for p in files:
            groups.setdefault(_gait_label(p), []).append(p)
        for gait, paths in sorted(groups.items()):
            plot_group(gait, paths, save_dir=save_dir, show=show)
    else:
        for p in files:
            plot_single(p, save_dir=save_dir, show=show)


if __name__ == "__main__":
    main()
