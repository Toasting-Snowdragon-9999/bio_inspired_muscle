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
        robot_interface: RobotInterface,
        stride_length: float = 0.06,
        step_height: float = 0.04,
        warmup_seconds: float = 2.0,
    ) -> None:
        self.neurons_cnt = NEURON_CNT  # one oscillator per leg (FL, FR, RR, RL)
        self.robot_interface = robot_interface
        self.warmup_seconds = warmup_seconds
        self.sim_time = 0.0
        self.stride_length = stride_length
        self.step_height = step_height

        # Read gait phase offsets and frequency from RobotInterface
        gait = self.robot_interface.current_gait
        foot_phases = np.array(gait.value)  # 4-tuple (FL, FR, RR, RL)
        frequency = self.robot_interface.current_state.frequency

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
        self._build_phase_offset_matrix(foot_phases)

        self.neurons = [
            self.Neuron(
                phase=foot_phases[i],
                frequency=frequency,
                amplitude=1.0,
            )
            for i in range(self.neurons_cnt)
        ]

        self.d_theta = np.zeros(self.neurons_cnt)

        # Home foot positions in hip-local frame, per oscillator index.
        # Must be populated via set_foot_home() before run() produces output.
        self._foot_home: dict[int, np.ndarray] = {}

        # Current output: oscillator index → foot position [x, y, z]
        self._targets: dict[int, np.ndarray] = {}

    # ── Configuration ───────────────────────────────────────────

    def set_foot_home(self, osc_idx: int, position: np.ndarray) -> None:
        """Set the rest/home foot position for oscillator osc_idx (hip-local frame)."""
        self._foot_home[osc_idx] = position.copy()

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

    def rk4_integration(self, dt: float) -> None:
        """Advance all oscillator phases by one timestep dt using 4th-order Runge-Kutta."""
        def derivatives(thetas: np.ndarray) -> np.ndarray:
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

        thetas = np.array([n.phase for n in self.neurons])
        k1 = derivatives(thetas)
        k2 = derivatives(thetas + 0.5 * dt * k1)
        k3 = derivatives(thetas + 0.5 * dt * k2)
        k4 = derivatives(thetas + dt * k3)
        thetas += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        for i, n in enumerate(self.neurons):
            n.phase = thetas[i]

    # ── Main step ───────────────────────────────────────────────

    def run(self) -> None:
        """
        Advance the CPG one timestep and update foot position targets.
        Reads dt from robot_interface.
        After calling, retrieve targets via get_targets().
        """
        dt = self.robot_interface.dt
        self.sim_time += dt

        if self.sim_time < self.warmup_seconds:
            # Hold feet at home position during warmup
            self._targets = {
                i: self._foot_home[i].copy()
                for i in range(self.neurons_cnt)
                if i in self._foot_home
            }
            return

        blend_duration = 1.0  # seconds to ramp from home to full trajectory
        blend = min(1.0, (self.sim_time - self.warmup_seconds) / blend_duration)

        self.rk4_integration(dt)

        for osc_idx in range(self.neurons_cnt):
            phase = self.neurons[osc_idx].phase
            home = self._foot_home.get(osc_idx, np.zeros(3))

            dx = -self.stride_length * np.cos(phase)
            dz = self.step_height * max(0.0, np.sin(phase))

            self._targets[osc_idx] = np.array([
                home[0] + blend * dx,
                home[1],                    # lateral stays at home
                home[2] + blend * dz,
            ])

    # ── Outputs ─────────────────────────────────────────────────

    def get_targets(self) -> dict[int, np.ndarray]:
        """Return current foot position targets: oscillator index → [x, y, z] in hip-local frame."""
        return dict(self._targets)

    def get_oscillator_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalised output signal per leg for oscillator graph overlay.
        Returns:
            leg_outputs:  shape (4,)  — -sin(θ_i)
            knee_outputs: shape (4,)  — clamped sin(θ_i + knee_offset)
        """
        leg_out = np.zeros(self.neurons_cnt)
        knee_out = np.zeros(self.neurons_cnt)
        for i in range(self.neurons_cnt):
            phase = self.neurons[i].phase
            leg_out[i] = -np.sin(phase)
            knee_out[i] = min(0.0, np.sin(phase + self.knee_phase_offset))
        return leg_out, knee_out
