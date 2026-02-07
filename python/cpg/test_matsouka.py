from matsuoka_oscillator import MatsuokaCPG
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """Test the Matsuoka CPG oscillator (arbitrary number of neurons)."""

    # Create oscillator
    cpg = MatsuokaCPG(dt=0.001)
    n = cpg.neurons_cnt

    # Check single-neuron stability condition
    single_oscillates = cpg.check_single_neuron_behavior()
    print(f"Single neuron oscillation check (should be False): {single_oscillates}")

    # Reset to initial conditions
    cpg.reset()

    # Run simulation
    duration = 20.0  # seconds
    print(f"Running Matsuoka CPG simulation for {duration} seconds...")
    results = cpg.run(duration)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    time = results['time']

    # ---- Neural outputs ----
    for i in range(n):
        axes[0].plot(
            time,
            results[f'neuron{i+1}_output'],
            label=f'Neuron {i+1}',
            linewidth=0.8
        )

    axes[0].set_ylabel('Output (y)')
    axes[0].set_title('Matsuoka CPG – Neural Outputs')
    axes[0].legend(ncol=min(n, 4))
    axes[0].grid(True, alpha=0.3)

    # ---- Internal states ----
    for i in range(n):
        axes[1].plot(
            time,
            results[f'neuron{i+1}_internal'],
            label=f'Neuron {i+1}',
            linewidth=0.8
        )

    axes[1].set_ylabel('Internal State (x)')
    axes[1].legend(ncol=min(n, 4))
    axes[1].grid(True, alpha=0.3)

    # ---- Adaptation states ----
    for i in range(n):
        axes[2].plot(
            time,
            results[f'neuron{i+1}_fatigue'],
            label=f'Neuron {i+1}',
            linewidth=0.8
        )

    axes[2].set_ylabel('Adaptation (x_adapt)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(ncol=min(n, 4))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent / 'output_files'
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'matsuoka_cpg_python.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # ---- Summary ----
    print("\nSimulation complete!")
    for i in range(n):
        print(
            f"Final output Neuron {i+1}: "
            f"{results[f'neuron{i+1}_output'][-1]:.4f}"
        )

    # ---- Analysis ----
    cpg.print_analysis(results=results, skip_initial_seconds=2.5)


if __name__ == '__main__':
    main()