from python.cpg.brian_sim.stein_oscillator import SteinCPGsim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from plotting.time_series_plotter import TimeSeriesPlotter


"""
Neuron 0 → Front Left  (FL)
Neuron 1 → Front Right (FR)
Neuron 2 → Rear  Right (RR)
Neuron 3 → Rear  Left  (RL)
"""


def map_cpg_output_to_activation(result, front: bool):
    FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)  # radians
    BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)  # radians
    max_val = np.max(result)
    if max_val == 0:
        max_val = 1.0
    normalized = result / max_val
    if front:
        theta_target = (
            FRONT_HIP_POS_RANGE[0]
            + normalized * (FRONT_HIP_POS_RANGE[1] - FRONT_HIP_POS_RANGE[0])
        )
        return theta_target
    else:
        theta_target = (
            BACK_HIP_POS_RANGE[0]
            + normalized * (BACK_HIP_POS_RANGE[1] - BACK_HIP_POS_RANGE[0])
        )
        return theta_target


def main():
    """Test the Stein phase-oscillator CPG."""

    # Create oscillator (default: trot gait at 1 Hz)
    cpg = SteinCPGsim(dt=0.001)
    n = cpg.neurons_cnt

    # Optionally change gait / frequency:
    # cpg.set_gait('walk')
    # cpg.set_frequency(2.0)

    # Reset to initial conditions
    cpg.reset()

    # Run simulation
    duration = 20.0  # seconds
    print(f"Running Stein CPG simulation for {duration} seconds...")
    results = cpg.run(duration)

    # --- Plotting ---
    time = results['time']

    # Create dictionary of neuron outputs
    neuron_data = {
        f'Neuron {i+1}': map_cpg_output_to_activation(
            results[f'neuron{i+1}_output'], front=(i < 2)
        )
        for i in range(n)
    }

    # Plot using the TimeSeriesPlotter
    plotter = TimeSeriesPlotter(time, figsize=(12, 3), subplot_height=3.0)
    plotter.plot(
        neuron_data,
        ylabel='Output (y)',
        xlabel='Time (seconds)',
        title_prefix='Stein ',
        linewidth=0.8
    )

    # Show plot
    plotter.show()

    # Save plot
    output_dir = Path(__file__).parent / 'output_files'
    plot_path = output_dir / 'stein_cpg_python.png'
    plotter.save(plot_path, dpi=150)

    # ---- Summary ----
    print("\nSimulation complete!")
    for i in range(n):
        print(
            f"Final output Neuron {i+1}: "
            f"{results[f'neuron{i+1}_output'][-1]:.4f}"
        )

    # ---- Analysis ----
    cpg.print_analysis(results=results, skip_initial_seconds=2.0)


if __name__ == '__main__':
    main()
