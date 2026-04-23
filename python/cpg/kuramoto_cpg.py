import os, sys
import numpy as np
from dataclasses import dataclass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import NEURON_CNT
from shared_module.robot_state import RobotInterface


class KuramotoCpg:
    """
    Kuramoto (phase) oscillator CPG for a quadruped.

    4 coupled oscillators, one per leg (FL, FR, RR, RL).
    Each has a single phase variable θ_i:
        dθ_i/dt = ω_i + Σ_j  w_ij · sin(θ_j - θ_i - φ_ij)

    Output: Cartesian foot position per leg in the hip-local frame:
        x = x_home - stride_length * cos(θ)
        z = z_home + step_height * max(0, sin(θ))   (lift only, no dig)

    Usage:  init → set_foot_home (per leg) → run each timestep → get_targets
    Gait and frequency are read from the RobotInterface at construction time.
    """

    @dataclass
    class Neuron:
        phase: float        # θ
        frequency: float    # ω
        amplitude: float    # A

    def __init__(
        self,
        robot_interface: RobotInterface
    ) -> None:
        self.neurons_cnt = NEURON_CNT  # one oscillator per leg (FL, FR, RR, RL)
        self.robot_interface = robot_interface
        self.time_passed = 0.0

        # Read gait phase offsets and frequency from RobotInterface
        gait = self.robot_interface.current_gait    
        if gait is None:
            # Remove this when transition is implemented since then the gait can dynamically be set and changed
            raise ValueError("RobotInterface must have a valid gait at CPG initialization.")
        
        gait_phases = np.array(gait.value)  # 4-tuple (FL, FR, RR, RL)
        frequency = self.robot_interface.frequency

        # Coupling: all-to-all with uniform strength
        self.coupling_weights = np.zeros((self.neurons_cnt, self.neurons_cnt))
        coupling_strength = 1.5 * np.pi
        for i in range(self.neurons_cnt):
            for j in range(self.neurons_cnt):
                if i != j:
                    self.coupling_weights[i, j] = coupling_strength

        # Used by oscillator graph overlay (knee lags leg phase by π)
        self.knee_phase_offset = -np.pi

        # Build pairwise phase-offset matrix from desired gait phases
        self.phase_offsets = np.zeros((self.neurons_cnt, self.neurons_cnt))
        self._build_phase_offset_matrix(gait_phases)

        self.neurons = [
            self.Neuron(
                phase=gait_phases[i],
                frequency=frequency,
                amplitude=1.0,
            )
            for i in range(self.neurons_cnt)
        ]

        self.d_theta = np.zeros(self.neurons_cnt)

    def reset(self) -> None:
        """Reset oscillator phases to the gait's initial values and clear accumulated state.
        Call before re-running the simulation with a new parameter set."""
        gait_phases = np.array(self.robot_interface.current_gait.value)
        frequency = self.robot_interface.frequency
        for i, n in enumerate(self.neurons):
            n.phase = gait_phases[i]
            n.frequency = frequency
        self.d_theta = np.zeros(self.neurons_cnt)
        self.time_passed = 0.0

    def set_frequency(self, frequency: float, index: int = None) -> None:
        if index is None:
            for n in self.neurons:
                n.frequency = frequency
        else:
            self.neurons[index].frequency = frequency

    # ── Internals ───────────────────────────────────────────────

    def _build_phase_offset_matrix(self, desired_phases: np.ndarray) -> None:
        n = self.neurons_cnt
        self.phase_offsets = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.phase_offsets[i, j] = desired_phases[j] - desired_phases[i]

    def derivatives(self, thetas: np.ndarray) -> np.ndarray:
        omegas = np.array([n.frequency * 2 * np.pi for n in self.neurons])
        coupling = np.zeros(self.neurons_cnt)
        for i in range(self.neurons_cnt):
            for j in range(self.neurons_cnt):
                if i != j:
                    coupling[i] += (
                        self.coupling_weights[i, j]
                        * np.sin(thetas[j] - thetas[i] - self.phase_offsets[i, j])
                    )
        return omegas + coupling

    def rk4_integration(self, dt: float) -> None:
        """Advance all oscillator phases by one timestep dt using 4th-order Runge-Kutta."""


        thetas = np.array([n.phase for n in self.neurons])
        k1 = self.derivatives(thetas)
        k2 = self.derivatives(thetas + 0.5 * dt * k1)
        k3 = self.derivatives(thetas + 0.5 * dt * k2)
        k4 = self.derivatives(thetas + dt * k3)
        thetas += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        # Use the RK4-averaged derivative for consistency with the actual phase integration
        self.d_theta = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        for i, n in enumerate(self.neurons):
            n.phase = thetas[i]

    def run(self) -> None:
        """
        Advance the CPG one timestep and update foot position targets.
        Reads dt from robot_interface.
        After calling, retrieve targets via get_targets().
        """
        dt = self.robot_interface.dt
        self.time_passed += dt

        self.rk4_integration(dt)

    def get_phase_outputs(self) -> np.ndarray:
        """Return current phase of each oscillator, for graph overlay."""
        return np.array([n.phase for n in self.neurons])

    def get_phase_velocities(self) -> np.ndarray:
        return self.d_theta

    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalised output signal per leg for oscillator graph overlay.
        Returns:
            leg_outputs:  shape (4,)  — -sin(θ_i)
            knee_outputs: shape (4,)  — clamped sin(θ_i + knee_offset)
        """
        outputs = []
        for i in range(self.neurons_cnt):
            phase = self.neurons[i].phase
            outputs.append(max(np.cos(phase), 0.0)) # Clamp to [0, 1]

        return outputs
