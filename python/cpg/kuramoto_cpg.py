"""
@brief Kuramoto phase-oscillator Central Pattern Generator (CPG) for a quadruped.

Defines KuramotoCpg, a network of four coupled phase oscillators (one per leg)
that produces the rhythmic phase signals driving foot-trajectory generation.
"""

import os, sys
import numpy as np
from dataclasses import dataclass
from scipy.optimize import root
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
        """
        @brief State of a single Kuramoto oscillator (one leg).

        @param phase: Oscillator phase θ (radians).
        @param frequency: Intrinsic oscillation frequency ω (Hz).
        @param amplitude: Oscillator amplitude A.
        """
        phase: float        # θ
        frequency: float    # ω
        amplitude: float    # A

    def __init__(
        self,
        robot_interface: RobotInterface
    ) -> None:
        """
        @brief Construct the CPG and initialise all oscillators from the active gait.

        Reads the active gait's phase offsets and the oscillation frequency from
        the RobotInterface, builds the all-to-all coupling and pairwise
        phase-offset matrices, and seeds one Neuron per leg.

        @param robot_interface: Shared robot state; supplies the active gait,
            frequency, timestep and CPG enable flag.
        @return None
        """
        self.neurons_cnt = NEURON_CNT  # one oscillator per leg (FL, FR, RR, RL)
        self.robot_interface = robot_interface
        self.time_passed = 0.0

        # Read gait phase offsets and frequency from RobotInterface
        self.gait = self.robot_interface.active_gait    
        if self.gait is None:
            # Remove this when transition is implemented since then the gait can dynamically be set and changed
            raise ValueError("RobotInterface must have a valid gait at CPG initialization.")
        
        gait_phases = np.array(self.gait.value)  # 4-tuple (FL, FR, RR, RL)
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
        self.old_phases = np.array([n.phase for n in self.neurons])  # Store current phases to restore later
        self.last_toggle = robot_interface.enable_cpg


    def reset(self) -> None:
        """@brief Reset oscillator phases and clear accumulated state.

        Reset oscillator phases to the gait's initial values and clear accumulated state.
        Call before re-running the simulation with a new parameter set.

        @return None"""
        gait_phases = np.array(self.robot_interface.active_gait.value)
        frequency = self.robot_interface.frequency
        for i, n in enumerate(self.neurons):
            n.phase = gait_phases[i]
            n.frequency = frequency
        self.d_theta = np.zeros(self.neurons_cnt)
        self.time_passed = 0.0


    def set_frequency(self, frequency: float, index: int = None) -> None:
        """
        @brief Set the oscillation frequency of one or all oscillators.

        @param frequency: New intrinsic frequency ω (Hz).
        @param index: Oscillator index to update; if None, applies to all oscillators.
        @return None
        """
        if index is None:
            for n in self.neurons:
                n.frequency = frequency
        else:
            self.neurons[index].frequency = frequency
    
    def set_gait(self, gait) -> None:
        """
        @brief Switch to a new gait, snapping oscillator phases and rebuilding offsets.

        No-op if the requested gait is already active. Otherwise each neuron's
        phase is snapped to the new gait's initial value and the pairwise
        phase-offset matrix is rebuilt.

        @param gait: Target gait; its `.value` is the 4-tuple of per-leg phases.
        @return None
        """
        if self.gait == gait:
            return
        self.gait = gait
        gait_phases = np.array(gait.value)
        for i, n in enumerate(self.neurons):
            n.phase = gait_phases[i]
        self._build_phase_offset_matrix(gait_phases)

    # ── Internals ───────────────────────────────────────────────

    def _build_phase_offset_matrix(self, desired_phases: np.ndarray) -> None:
        """
        @brief Build the pairwise phase-offset matrix from desired per-leg phases.

        Entry (i, j) holds the desired relative phase desired_phases[j] - desired_phases[i],
        used as the target offset φ_ij in the Kuramoto coupling term.

        @param desired_phases: Per-oscillator desired absolute phases (radians), one per leg.
        @return None
        """
        n = self.neurons_cnt
        self.phase_offsets = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.phase_offsets[i, j] = desired_phases[j] - desired_phases[i]

    def derivatives(self, thetas: np.ndarray) -> np.ndarray:
        """
        @brief Evaluate the Kuramoto phase derivatives dθ_i/dt at the given phases.

        Computes the natural angular frequency plus the all-to-all coupling term
        Σ_j w_ij · sin(θ_j - θ_i - φ_ij) for each oscillator.

        @param thetas: Current phases θ of all oscillators (radians).
        @return Phase velocities dθ/dt for all oscillators (radians/second).
        """
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
        """@brief Advance oscillator phases one timestep with 4th-order Runge-Kutta.

        Advance all oscillator phases by one timestep dt using 4th-order Runge-Kutta.

        @param dt: Integration timestep (seconds).
        @return None"""


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

    def trapezoidal_integration(self, dt: float) -> None:
        """
        @brief Advance oscillator phases one timestep with the implicit trapezoidal method.

        Advance oscillator phases using the implicit trapezoidal method.

        @param dt: Integration timestep (seconds).
        @return None
        """

        thetas = np.array([n.phase for n in self.neurons])

        # f(t_n, y_n)
        f_n = self.derivatives(thetas)

        # Explicit Euler initial guess
        theta_guess = thetas + dt * f_n

        # Residual of trapezoidal equation
        def phi(theta_new):
            """
            @brief Phi.
            @param theta_new:
            @return
            """
            return (
                theta_new
                - thetas
                - 0.5 * dt * (
                    f_n + self.derivatives(theta_new)
                )
            )

        sol = root(phi, theta_guess)

        if not sol.success:
            raise RuntimeError("Trapezoidal Newton solve failed")

        theta_new = sol.x

        # Approximate derivative used for plotting/logging
        self.d_theta = (
            theta_new - thetas
        ) / dt

        # Update neuron phases
        for i, n in enumerate(self.neurons):
            n.phase = theta_new[i]

    def run(self) -> None:
        """
        @brief Advance the CPG one timestep using the configured timestep.

        Advance the CPG one timestep and update foot position targets.
        Reads dt from robot_interface.
        After calling, retrieve targets via get_targets().

        @return None
        """
        dt = self.robot_interface.dt
        self.time_passed += dt

        self.trapezoidal_integration(dt)

    def get_phase_outputs(self) -> np.ndarray:
        """@brief Return the current phase of each oscillator.

        Return current phase of each oscillator, for graph overlay.

        @return Array of current phases θ (radians), one per oscillator."""
        return np.array([n.phase for n in self.neurons])

    def get_phase_velocities(self) -> np.ndarray:
        """
        @brief Return the most recent phase velocities of all oscillators.

        @return Array of phase velocities dθ/dt (radians/second), one per oscillator.
        """
        return self.d_theta

    def get_oscillator_outputs(self) -> list[float]:
        """
        @brief Return the per-leg normalised oscillator output signal.

        Produces one value per leg, used for the oscillator graph overlay and as
        the per-foot CPG signal consumed by IKController.run. The knee phase
        offset is deliberately not applied here.

        @return Length-4 list of cos(theta_i), one per oscillator, in leg order
                (see NEURON_TO_FOOT_DICT). Each value lies in [-1, 1].
        """
        outputs = []
        for i in range(self.neurons_cnt):
            phase = self.neurons[i].phase
            # outputs.append(max(np.cos(phase), 0.0)) # Clamp to [0, 1]
            outputs.append(np.cos(phase))
        return outputs
