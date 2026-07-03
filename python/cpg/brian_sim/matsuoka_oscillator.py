"""@brief Brian2 spiking-style simulation of a four-neuron Matsuoka CPG, with oscillation analysis utilities."""
import brian2 as b2
from brian2 import (
    second, ms, 
    NeuronGroup, Synapses, StateMonitor, Network, network_operation,
    np
)
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class MatsuokaCPGsim:
    """@brief Brian2-backed four-neuron Matsuoka CPG simulation with mutual inhibition and state monitoring."""
    def __init__(self, dt=0.001):
        """@brief Build the Brian2 neuron group, coupling, network operation, and monitors for the Matsuoka CPG.
        @param dt: integration timestep in seconds (converted to a Brian2 quantity).
        """
        self.dt = dt * second

        eqs = '''
        dx/dt = (-x - I_inh + s - b*x_adapt + feed) / tau : 1
        dx_adapt/dt = (-x_adapt + y) / T : 1
        y = clip(x, 0, inf) : 1
        I_inh : 1
        s : 1
        tau: second
        T: second
        b: 1
        feed : 1
        '''

        self.neurons_cnt = 4
        self.neurons = NeuronGroup(
            self.neurons_cnt,
            eqs,
            method='rk4', # Runge-Kutta 4th order, better for keeping freq
            dt=self.dt
        )

        self.neurons.tau = 0.05 * second            # This can change the freq
        self.neurons.T   = 0.5 * second             # 
        self.neurons.b   = 2.5                      # adaptation strength
        self.neurons.s = [1.0, 1.0, 1.0, 1.0]       # tonic drive, like bias current, constant and always present

        self.neurons.I_inh = [0, 0, 0, 0]           # Weighted connection to other neurons, should be negative for inhibition
        self.neurons.feed = [0.0, 0.0, 0.02, 0.0]   # External input 

        print("Initialized Matsuoka CPG with parameters:")

        for neuron in self.neurons:
            neuron.x = np.random.rand()  # Small random initial state
            print(neuron.x)

        self.neurons.x = [0.1, 0.0, 0.0, 0.0]       # Internal state
        self.neurons.x_adapt = [0.0, 0.0, 0.0, 0.0] # Adaption state (fatigue)

        # The a_ij weights for mutual inhibition 
        # All diagonal elements are 0 (no self-inhibition)
        # Changing this will change the gait pattern
        # Should be negative for inhibition
        
        self.inhibitory_connection = np.array([
            [0.0, 1.5, 2.0, 1.0],
            [1.5, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0]
        ])

        @network_operation(dt=self.dt) 
        def update_inhibition(): 
            """@brief Brian2 network operation: recompute each neuron's inhibitory input each timestep."""
            # Matrix multiply: I_inh[i] = Σ_j a[i,j] * y[j]
            # self.neurons.I_inh = (
            #     np.dot(self.inhibitory_connection, self.neurons.y)
            #     / np.sum(self.inhibitory_connection, axis=1)
            # )
            self.neurons.I_inh = np.dot(self.inhibitory_connection, self.neurons.y)

        self.mon = StateMonitor(
            self.neurons,
            ['x', 'x_adapt', 'y'],
            record=True
        )

        self.net = Network(self.neurons, update_inhibition, self.mon)
        self.net.store('initial')

    def reset(self):
        """@brief Restore the network to its stored initial state."""
        self.net.restore('initial')

    def check_single_neuron_behavior(self) -> bool:
        """@brief Check the stability condition guaranteeing a single neuron does not self-oscillate.
        @return True if (T + tau)^2 >= 4·T·tau·b (single neuron stable / non-oscillating), else False.
        """
        T = float(self.neurons.T[0] / second)  # Convert to dimensionless
        tau = float(self.neurons.tau[0])
        b = float(self.neurons.b[0])
        return bool((T + tau)**2 >= (4 * T * tau * b))

    def run(self, duration):
        """@brief Run the Matsuoka CPG simulation and return monitored time series for each neuron.
        @param duration: simulation length in seconds.
        @return dict with 'time' plus per-neuron internal state, fatigue, and output arrays.
        """

        single_neuron_check = self.check_single_neuron_behavior()
        if single_neuron_check:
            print("Correct parameters for a single neuron to not oscillate.")
        else:
            print("Warning: Parameters will make single neuron oscillate.")

        self.net.run(duration * second)

        return_dict = {'time': self.mon.t / second}
        for i in range(self.neurons_cnt):
            return_dict[f'neuron{i+1}_internal'] = self.mon.x[i] 
            return_dict[f'neuron{i+1}_fatigue'] = self.mon.x_adapt[i] 
            return_dict[f'neuron{i+1}_output'] = self.mon.y[i] 

        return return_dict

    def analyze_oscillations(self, results, neuron_idx=0, skip_initial_seconds=10.0):
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
        min_distance = int(0.5 / dt)  # minimum 0.5 s between peaks
        peaks, properties = find_peaks(
            output_analysis,
            height=np.mean(output_analysis),
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

        # --- Clear, explicit reporting ---
        # print(f"CPG cycle frequency (1 / mean period): {cycle_freq_hz:.3f} Hz")
        # print(f"Dominant spectral frequency (FFT):     {dominant_spectral_freq_hz:.3f} Hz")
        # print(f"Mean period: {mean_period:.3f} s | CV: {cv_period:.3f}")

        return {
            'cycle_freq_hz': cycle_freq_hz,                     # ← use this for gait
            'mean_period': mean_period,
            'cv_period': cv_period,
            'mean_peak_amp': mean_peak_amp,
            'dominant_spectral_freq_hz': dominant_spectral_freq_hz,  # ← diagnostic
            'periods': periods,
            'peak_amplitudes': peak_amplitudes,
            'num_peaks': len(peaks),
            'peak_times': peak_times
        }

    def print_analysis(self, results, skip_initial_seconds=10.0):
        """
        @brief Print the oscillation analysis (cycle and spectral frequencies) for every neuron.
        @param results: results dict returned by run().
        @param skip_initial_seconds: leading transient duration (seconds) to discard before analysis.
        Print oscillation analysis for both neurons.

        Clearly distinguishes:
        - cycle frequency (time-domain, gait-relevant)
        - dominant spectral frequency (FFT, diagnostic)
        """

        print("\n" + "=" * 60)
        print("CPG OSCILLATION ANALYSIS")
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
            print(f"  Period CV:                {analysis['cv_period']:.4f}")
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

        print("=" * 60 + "\n")
