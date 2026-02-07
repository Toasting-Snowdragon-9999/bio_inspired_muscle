"""
Matsuoka CPG implementation using Brian2 neural simulator.
This ensures mathematically correct oscillations based on validated neural dynamics.
Matches the C++ implementation in matsuokaneuron.cpp
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq


class MatsuokaCPG:
    """
    Matsuoka Central Pattern Generator using Brian2 for validated neural dynamics.
    Parameters match the C++ implementation for direct comparison.
    """

    def __init__(self, tau_r=0.1, tau_f=1.2, beta=2.5, w_mutual=2.5, w_self_1=2.0,
                 w_self_2=1.5, s0=1.0, dt=0.01):
        """
        Initialize Matsuoka oscillator with two mutually inhibiting neurons.

        Args:
            tau_r: Time constant for internal state recovery (seconds)
            tau_f: Time constant for fatigue (seconds)
            beta: Fatigue feedback strength
            w_mutual: Mutual inhibition weight between neurons
            w_self_1: Self-excitation weight for neuron 1
            w_self_2: Self-excitation weight for neuron 2
            s0: Tonic excitation input
            dt: Integration time step (seconds)
        """
        self.dt = dt * second

        # Define the Matsuoka neuron equations
        eqs = '''
        du/dt = (-u + synaptic_input + w_self * s0 - beta * v + external_input) / tau_r : 1
        dv/dt = (-v + y) / tau_f : 1
        y = clip(u, 0, inf) : 1
        synaptic_input : 1
        external_input : 1
        w_self : 1
        tau_r : second
        tau_f : second
        beta : 1
        s0 : 1
        '''

        # Create two neurons
        self.neurons = NeuronGroup(2, eqs, method='euler', dt=self.dt)

        # Set parameters for each neuron (matching C++ implementation)
        self.neurons.tau_r = tau_r * second
        self.neurons.tau_f = tau_f * second
        self.neurons.beta = beta
        self.neurons.s0 = s0
        self.neurons.w_self = [w_self_1, w_self_2]
        self.neurons.external_input = 0.0

        # Store mutual inhibition weight
        self.w_mutual = w_mutual

        # State monitors to record outputs
        self.mon = StateMonitor(self.neurons, ['u', 'v', 'y'], record=True)

        # Network
        self.net = Network(self.neurons, self.mon)
        self.net.store('initial')

    def reset(self):
        """Reset the oscillator to initial conditions."""
        self.net.restore('initial')

    def run(self, duration):
        """
        Run the oscillator simulation.

        Args:
            duration: Simulation duration in seconds

        Returns:
            dict with 'time', 'neuron1_output', 'neuron2_output'
        """
        # Update synaptic input based on mutual inhibition at each timestep
        @network_operation(dt=self.dt)
        def update_synaptic_input():
            self.neurons.synaptic_input[0] = -self.w_mutual * self.neurons.y[1]
            self.neurons.synaptic_input[1] = -self.w_mutual * self.neurons.y[0]

        # Add operation to network and run
        self.net.add(update_synaptic_input)
        self.net.run(duration * second)
        self.net.remove(update_synaptic_input)

        # Return results
        return {
            'time': self.mon.t / second,
            'neuron1_output': self.mon.y[0],
            'neuron2_output': self.mon.y[1],
            'neuron1_internal': self.mon.u[0],
            'neuron2_internal': self.mon.u[1],
            'neuron1_fatigue': self.mon.v[0],
            'neuron2_fatigue': self.mon.v[1],
        }

    def set_external_input(self, value):
        """Set external input current for both neurons."""
        self.neurons.external_input = value

    def analyze_oscillations(self, results, neuron_idx=0, skip_initial_seconds=10.0):
        """
        Analyze oscillation characteristics of a neuron's output.

        Args:
            results: Results dictionary from run() method
            neuron_idx: Which neuron to analyze (0 or 1)
            skip_initial_seconds: Skip initial transient period (seconds)

        Returns:
            dict with analysis metrics:
                - mean_period: Mean oscillation period (seconds)
                - cv_period: Coefficient of variation of period (std/mean)
                - mean_peak_amp: Mean amplitude of peaks
                - dominant_freq_hz: Dominant frequency from FFT (Hz)
                - periods: List of individual periods
                - peak_amplitudes: List of peak amplitudes
        """
        time = results['time']
        output = results[f'neuron{neuron_idx + 1}_output']

        # Skip initial transient
        skip_idx = np.searchsorted(time, skip_initial_seconds)
        time_analysis = time[skip_idx:]
        output_analysis = output[skip_idx:]

        if len(output_analysis) < 100:
            raise ValueError("Not enough data points after skipping initial period")

        # Find peaks in the signal
        # Use a minimum distance to avoid detecting noise
        dt = time[1] - time[0]
        min_distance = int(0.5 / dt)  # Minimum 0.5 second between peaks

        peaks, properties = signal.find_peaks(output_analysis,
                                              height=np.mean(output_analysis),
                                              distance=min_distance)

        if len(peaks) < 2:
            return {
                'mean_period': np.nan,
                'cv_period': np.nan,
                'mean_peak_amp': np.nan,
                'dominant_freq_hz': np.nan,
                'periods': [],
                'peak_amplitudes': [],
                'num_peaks': 0
            }

        # Calculate periods (time between consecutive peaks)
        peak_times = time_analysis[peaks]
        periods = np.diff(peak_times)
        mean_period = np.mean(periods)
        std_period = np.std(periods)
        cv_period = std_period / mean_period if mean_period > 0 else np.nan

        # Get peak amplitudes
        peak_amplitudes = output_analysis[peaks]
        mean_peak_amp = np.mean(peak_amplitudes)

        # Calculate dominant frequency using FFT
        N = len(output_analysis)
        yf = fft(output_analysis)
        xf = fftfreq(N, dt)

        # Only look at positive frequencies
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = np.abs(yf[pos_mask])

        # Find dominant frequency (peak in power spectrum)
        dominant_idx = np.argmax(yf_pos)
        dominant_freq_hz = xf_pos[dominant_idx]

        return {
            'mean_period': mean_period,
            'cv_period': cv_period,
            'mean_peak_amp': mean_peak_amp,
            'dominant_freq_hz': dominant_freq_hz,
            'periods': periods,
            'peak_amplitudes': peak_amplitudes,
            'num_peaks': len(peaks),
            'peak_times': peak_times,
            'peak_indices': peaks
        }

    def print_analysis(self, results, skip_initial_seconds=10.0):
        """
        Print analysis for both neurons.

        Args:
            results: Results dictionary from run() method
            skip_initial_seconds: Skip initial transient period (seconds)
        """
        print("\n" + "="*60)
        print("CPG OSCILLATION ANALYSIS")
        print("="*60)

        for neuron_idx in [0, 1]:
            analysis = self.analyze_oscillations(results, neuron_idx, skip_initial_seconds)

            print(f"\nNeuron {neuron_idx + 1}:")
            print(f"  Mean Period:        {analysis['mean_period']:.4f} s")
            print(f"  CV of Period:       {analysis['cv_period']:.4f}")
            print(f"  Mean Peak Amp:      {analysis['mean_peak_amp']:.4f}")
            print(f"  Dominant Freq:      {analysis['dominant_freq_hz']:.4f} Hz")
            print(f"  Number of Peaks:    {analysis['num_peaks']}")

            if len(analysis['periods']) > 0:
                print(f"  Period Range:       [{np.min(analysis['periods']):.4f}, {np.max(analysis['periods']):.4f}] s")
                print(f"  Peak Amp Range:     [{np.min(analysis['peak_amplitudes']):.4f}, {np.max(analysis['peak_amplitudes']):.4f}]")

        print("="*60 + "\n")

    def save_to_csv(self, filename, results):
        """Save results to CSV file matching C++ output format."""
        output_dir = Path(__file__).parent / 'output_files'
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / f"{filename}.matsuoka"

        with open(filepath, 'w') as f:
            f.write("time,o1,o2\n")
            for t, o1, o2 in zip(results['time'], results['neuron1_output'], results['neuron2_output']):
                f.write(f"{t:.6f},{o1:.6f},{o2:.6f}\n")

        print(f"Saved output to {filepath}")
        return True



