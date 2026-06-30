"""@brief Brian2 simulation of a Stein/Kuramoto phase-oscillator CPG for quadruped gaits, with analysis utilities."""
import brian2 as b2
from brian2 import (
    second, ms, Hz,
    NeuronGroup, StateMonitor, Network, network_operation,
    np
)
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


class SteinCPGsim:
    """
    @brief Brian2 Stein/Kuramoto phase-oscillator CPG with gait presets and phase-locking analysis.
    Stein (phase) oscillator CPG for quadruped locomotion.

    Each oscillator is described by a single phase variable θ_i:

        dθ_i/dt = ω_i + Σ_j  w_ij · sin(θ_j - θ_i - φ_ij)

    where:
        ω_i   - intrinsic angular frequency  (rad/s)
        w_ij  - coupling strength from oscillator j → i  (rad/s)
        φ_ij  - desired phase offset of j relative to i  (rad)

    Output per oscillator (half-wave rectified cosine):
        y_i = A_i · max(0, cos(θ_i))

    Gait is directly encoded by the phase-offset matrix φ.
    Preset gaits: trot, walk, bound, pace, gallop.
    """

    # ── Preset gait phase vectors ───────────────────────────────────
    # Neuron mapping:  0 = FL,  1 = FR,  2 = RR,  3 = RL
    GAITS = {
        'trot':   np.array([0.0,       np.pi,     0.0,       np.pi]),
        'walk':   np.array([0.0,       np.pi/2,   np.pi,     3*np.pi/2]),

        'bound':  np.array([0.0,       0.0,       np.pi,     np.pi]),
        'pace':   np.array([0.0,       np.pi,     np.pi,     0.0]),
        'gallop': np.array([0.0,       0.0,       np.pi*0.8, np.pi*0.8]),
    }

    def __init__(self, dt=0.001):
        """@brief Build the Brian2 phase-oscillator group, coupling weights, default gait, and monitors.
        @param dt: integration timestep in seconds (converted to a Brian2 quantity).
        """
        self.dt = dt * second

        # ── Brian2 equations ────────────────────────────────────────
        eqs = '''
        dtheta/dt = omega + coupling_input : 1
        y = amplitude * clip(cos(theta), 0, inf) : 1
        omega : Hz
        coupling_input : Hz
        amplitude : 1
        '''

        self.neurons_cnt = 4
        self.neurons = NeuronGroup(
            self.neurons_cnt,
            eqs,
            method='rk4',
            dt=self.dt
        )

        # ── Default parameters ──────────────────────────────────────
        freq_hz = 1.0  # intrinsic cycle frequency (Hz)
        self.neurons.omega = freq_hz * 2 * np.pi * Hz
        step_height = 0.02  # m
        multiplier = 1.0  # scale up to increase step height
        self.neurons.amplitude = step_height * multiplier
        self.neurons.coupling_input = 0 * Hz

        # ── Coupling weights (rad/s) ───────────────────────────────
        # Uniform coupling; increase for faster phase-locking
        self.coupling_weights = np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]) * 2 * np.pi  # scale to ~ one full-cycle correction per second

        # ── Gait (phase offsets) ────────────────────────────────────
        self.set_gait('gallop')                       # sets phase_offsets matrix + initial θ

        print("Initialized Stein phase-oscillator CPG with parameters:")
        print(f"  Intrinsic frequency : {freq_hz:.2f} Hz")
        print(f"  Gait                : trot")
        print(f"  Neurons             : {self.neurons_cnt}")
        print(f"  dt                  : {float(self.dt/second)*1000:.1f} ms")

        # ── Network operation: update coupling at each step ─────────
        @network_operation(dt=self.dt)
        def update_coupling():
            """@brief Brian2 network operation: recompute each oscillator's phase-coupling input each timestep."""
            thetas = np.array(self.neurons.theta[:])
            coupling = np.zeros(self.neurons_cnt)
            for i in range(self.neurons_cnt):
                for j in range(self.neurons_cnt):
                    if i != j:
                        coupling[i] += (
                            self.coupling_weights[i, j]
                            * np.sin(thetas[j] - thetas[i] - self.phase_offsets[i, j])
                        )
            self.neurons.coupling_input = coupling * Hz

        # ── Monitors & Network ──────────────────────────────────────
        self.mon = StateMonitor(
            self.neurons,
            ['theta', 'y'],
            record=True
        )

        self.net = Network(self.neurons, update_coupling, self.mon)
        self.net.store('initial')

    # ── Gait helpers ────────────────────────────────────────────────

    def set_gait(self, gait_name: str):
        """
        @brief Select a preset gait by name, rebuilding the phase-offset matrix and seeding initial phases.
        @param gait_name: one of the preset gaits (trot, walk, bound, pace, gallop).
        Set gait by name.  Also resets initial phases to match the
        desired pattern so the network locks in quickly.

        Available gaits: trot, walk, bound, pace, gallop
        """
        if gait_name not in self.GAITS:
            raise ValueError(
                f"Unknown gait '{gait_name}'. "
                f"Choose from: {list(self.GAITS.keys())}"
            )
        desired_phases = self.GAITS[gait_name]
        self._build_phase_offset_matrix(desired_phases)
        self.neurons.theta = desired_phases           # start at target pattern
        self._current_gait = gait_name

    def set_custom_gait(self, desired_phases: np.ndarray):
        """
        @brief Set an arbitrary gait from a desired per-oscillator phase vector.
        @param desired_phases: phase vector of length neurons_cnt (radians), one entry per oscillator.
        Set an arbitrary gait by providing the desired phase vector
        (one entry per oscillator, in radians).
        """
        desired_phases = np.asarray(desired_phases, dtype=float)
        if desired_phases.shape != (self.neurons_cnt,):
            raise ValueError(
                f"Expected phase vector of length {self.neurons_cnt}, "
                f"got {desired_phases.shape}"
            )
        self._build_phase_offset_matrix(desired_phases)
        self.neurons.theta = desired_phases
        self._current_gait = 'custom'

    def _build_phase_offset_matrix(self, desired_phases: np.ndarray):
        """@brief Build the φ_ij matrix from a desired-phase vector.
        @param desired_phases: per-oscillator target phases (radians) defining the gait pattern.
        Build the φ_ij matrix from a desired-phase vector."""
        n = self.neurons_cnt
        self.phase_offsets = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.phase_offsets[i, j] = desired_phases[j] - desired_phases[i]

    # ── Frequency control ───────────────────────────────────────────

    def set_frequency(self, freq_hz: float):
        """@brief Set intrinsic oscillation frequency for all oscillators (Hz).
        @param freq_hz: intrinsic cycle frequency applied to every oscillator (Hz).
        Set intrinsic oscillation frequency for all oscillators (Hz)."""
        self.neurons.omega = freq_hz * 2 * np.pi * Hz

    def set_frequency_per_neuron(self, freqs_hz):
        """@brief Set per-oscillator intrinsic frequencies (Hz).
        @param freqs_hz: array-like of per-oscillator intrinsic frequencies (Hz).
        Set per-oscillator intrinsic frequencies (Hz)."""
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        self.neurons.omega = freqs_hz * 2 * np.pi * Hz

    # ── Simulation lifecycle ────────────────────────────────────────

    def reset(self):
        """@brief Restore the network to its stored initial state."""
        self.net.restore('initial')

    def run(self, duration):
        """@brief Run the CPG for *duration* seconds. Returns a results dict.
        @param duration: simulation length in seconds.
        @return dict with 'time' plus per-neuron phase and output arrays.
        Run the CPG for *duration* seconds. Returns a results dict."""
        self.net.run(duration * second)

        return_dict = {'time': self.mon.t / second}
        for i in range(self.neurons_cnt):
            return_dict[f'neuron{i+1}_phase']  = self.mon.theta[i]
            return_dict[f'neuron{i+1}_output'] = self.mon.y[i]

        return return_dict

    # ── Analysis ────────────────────────────────────────────────────

    def analyze_oscillations(self, results, neuron_idx=0, skip_initial_seconds=2.0):
        """
        @brief Analyze a neuron's output oscillation: peak-based cycle frequency and FFT spectral frequency.
        @param results: results dict returned by run().
        @param neuron_idx: zero-based index of the neuron to analyze.
        @param skip_initial_seconds: leading transient duration (seconds) to discard before analysis.
        @return dict of oscillation metrics (cycle/spectral frequency, period stats, peaks); NaN-filled if too few peaks.
        Analyze oscillation characteristics of a neuron's output.

        Returns both:
        - cycle frequency (from peak timing)  ← locomotion-relevant
        - dominant spectral frequency (FFT)   ← diagnostic only
        """
        time = results['time']
        output = results[f'neuron{neuron_idx + 1}_output']

        # --- Remove initial transient ---
        skip_idx = np.searchsorted(time, skip_initial_seconds)
        time_analysis = time[skip_idx:]
        output_analysis = output[skip_idx:]

        if len(output_analysis) < 100:
            raise ValueError("Not enough data after transient removal")

        dt = time_analysis[1] - time_analysis[0]

        # --- Peak-based cycle analysis (PRIMARY) ---
        min_distance = int(0.3 / dt)  # minimum 0.3 s between peaks
        peaks, properties = find_peaks(
            output_analysis,
            height=np.mean(output_analysis) * 0.5,
            distance=min_distance
        )

        if len(peaks) < 2:
            return {
                'cycle_freq_hz': np.nan,
                'mean_period': np.nan,
                'cv_period': np.nan,
                'dominant_spectral_freq_hz': np.nan,
                'num_peaks': len(peaks)
            }

        peak_times = time_analysis[peaks]
        periods = np.diff(peak_times)

        mean_period = np.mean(periods)
        std_period = np.std(periods)
        cv_period = std_period / mean_period if mean_period > 0 else np.nan

        cycle_freq_hz = 1.0 / mean_period

        peak_amplitudes = output_analysis[peaks]
        mean_peak_amp = np.mean(peak_amplitudes)

        # --- FFT-based spectral analysis (SECONDARY / DIAGNOSTIC) ---
        signal_demeaned = output_analysis - np.mean(output_analysis)

        N = len(signal_demeaned)
        yf = np.abs(fft(signal_demeaned))
        xf = fftfreq(N, dt)

        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]

        dominant_spectral_freq_hz = xf_pos[np.argmax(yf_pos)]

        return {
            'cycle_freq_hz': cycle_freq_hz,
            'mean_period': mean_period,
            'cv_period': cv_period,
            'mean_peak_amp': mean_peak_amp,
            'dominant_spectral_freq_hz': dominant_spectral_freq_hz,
            'periods': periods,
            'peak_amplitudes': peak_amplitudes,
            'num_peaks': len(peaks),
            'peak_times': peak_times
        }

    def analyze_phase_locking(self, results, skip_initial_seconds=2.0):
        """
        @brief Measure how closely each oscillator pair locked to its desired phase offset.
        @param results: results dict returned by run().
        @param skip_initial_seconds: leading transient duration (seconds) to discard before analysis.
        @return dict keyed by (i, j) pair of mean/std circular phase error and the desired offset (radians).
        Measure how well oscillators have locked to the desired phase
        offsets.  Returns mean and std of phase errors for each pair.
        """
        time = results['time']
        skip_idx = np.searchsorted(time, skip_initial_seconds)

        n = self.neurons_cnt
        phase_errors = {}

        for i in range(n):
            for j in range(i + 1, n):
                theta_i = results[f'neuron{i+1}_phase'][skip_idx:]
                theta_j = results[f'neuron{j+1}_phase'][skip_idx:]
                desired = self.phase_offsets[i, j]

                # Circular error: wrap to [-π, π]
                diff = (theta_j - theta_i - desired + np.pi) % (2 * np.pi) - np.pi
                phase_errors[(i, j)] = {
                    'mean_error_rad': np.mean(np.abs(diff)),
                    'std_error_rad': np.std(diff),
                    'desired_offset_rad': desired,
                }

        return phase_errors

    def print_analysis(self, results, skip_initial_seconds=2.0):
        """
        @brief Print per-neuron oscillation analysis plus pairwise phase-locking quality.
        @param results: results dict returned by run().
        @param skip_initial_seconds: leading transient duration (seconds) to discard before analysis.
        Print oscillation analysis for all neurons, plus phase-locking quality.
        """
        print("\n" + "=" * 60)
        print("STEIN PHASE-OSCILLATOR CPG ANALYSIS")
        print("=" * 60)

        for neuron_idx in range(self.neurons_cnt):
            analysis = self.analyze_oscillations(
                results=results,
                neuron_idx=neuron_idx,
                skip_initial_seconds=skip_initial_seconds
            )

            print(f"\nNeuron {neuron_idx + 1}:")

            if np.isnan(analysis['cycle_freq_hz']):
                print("  No stable oscillation detected.")
                continue

            print(f"  Mean Period:              {analysis['mean_period']:.4f} s")
            print(f"  Cycle Frequency (1/T):    {analysis['cycle_freq_hz']:.4f} Hz")
            print(f"  Period CV:                {analysis['cv_period']:.6f}")
            print(f"  Mean Peak Amplitude:      {analysis['mean_peak_amp']:.4f}")
            print(f"  Dominant Spectral Freq:   {analysis['dominant_spectral_freq_hz']:.4f} Hz")
            print(f"  Number of Peaks:          {analysis['num_peaks']}")

            if len(analysis['periods']) > 0:
                print(
                    f"  Period Range:             "
                    f"[{np.min(analysis['periods']):.4f}, "
                    f"{np.max(analysis['periods']):.4f}] s"
                )
                print(
                    f"  Peak Amplitude Range:     "
                    f"[{np.min(analysis['peak_amplitudes']):.4f}, "
                    f"{np.max(analysis['peak_amplitudes']):.4f}]"
                )

        # ── Phase-locking quality ───────────────────────────────────
        print("\n" + "-" * 60)
        print("PHASE-LOCKING QUALITY")
        print("-" * 60)
        leg_names = ['FL', 'FR', 'RR', 'RL']
        phase_errors = self.analyze_phase_locking(results, skip_initial_seconds)
        for (i, j), err in phase_errors.items():
            desired_deg = np.degrees(err['desired_offset_rad'])
            mean_err_deg = np.degrees(err['mean_error_rad'])
            print(
                f"  {leg_names[i]}-{leg_names[j]}:  "
                f"desired {desired_deg:+7.1f}°  |  "
                f"mean error {mean_err_deg:5.2f}°  |  "
                f"std {np.degrees(err['std_error_rad']):5.2f}°"
            )

        print("=" * 60 + "\n")
