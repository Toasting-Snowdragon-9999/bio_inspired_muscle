"""
Muscle-like PD controller for quadruped knee joints.

Physics model
─────────────
Hip/thigh joints use stiff position-tracking PD control (high gains) so the
CPG can prescribe their trajectories directly.

Knee (calf) joints are modelled as passive spring-damper systems:

    τ = k · (θ_ref − θ) − d · θ̇

where
    k   - torsional spring stiffness  [N·m / rad]
    d   - viscous damping coefficient  [N·m·s / rad]
    θ   - current joint angle
    θ̇   - current joint angular velocity
    θ_ref - reference (rest) angle supplied by the CPG

This is NOT a high-gain trajectory tracker.  The knee *lags* the CPG
reference by an amount that depends on the damping ratio

    ζ = d / (2 · √(k · I))

where I is the joint's rotational inertia (estimated from MuJoCo).
"""

import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco as mj

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.global_constants import *


# ── Output directory for test artefacts ─────────────────────────
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_images')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


class MusclePdController:
    """Dual-mode joint controller: stiff PD for hips, spring-damper for knees."""

    # ─── Exceptions ─────────────────────────────────────────────
    class InvalidDampingError(Exception):
        """Raised when the damping coefficient is non-positive."""

    # ─── Data container returned by test routines ───────────────
    @dataclass
    class TestRecord:
        """Time-series data produced by step_response / frequency_response."""
        time: list[float] = field(default_factory=list)
        theta: list[float] = field(default_factory=list)
        theta_dot: list[float] = field(default_factory=list)
        theta_ref: list[float] = field(default_factory=list)
        torque: list[float] = field(default_factory=list)

    # ─── Construction ───────────────────────────────────────────
    def __init__(
        self,
        k: float = 40.0,
        d: float = 4.0,
        passive_kp: float = 40.0,
        passive_kd: float = 4.0,
    ) -> None:
        """
        Parameters
        ----------
        k : float
            Knee spring stiffness  [N·m / rad].
        d : float
            Knee viscous damping   [N·m·s / rad].
        passive_kp : float
            Hip position gain (stiff tracker).
        passive_kd : float
            Hip derivative gain (stiff tracker).
        """
        # ── Knee: passive spring-damper parameters ──────────────
        if d < 0.0:
            raise self.InvalidDampingError(
                f"Damping coefficient d must be ≥ 0, got {d}"
            )
        self.k: float = k   # spring stiffness
        self.d: float = d   # viscous damping

        # ── Hip / thigh: stiff PD tracker ───────────────────────
        self.passive_kp: float = passive_kp
        self.passive_kd: float = passive_kd

        # ── Simulation handles (set in init_controller) ─────────
        self.model: Optional[mj.MjModel] = None
        self.data: Optional[mj.MjData] = None
        self.dt: Optional[float] = None

    # ─── MuJoCo bootstrap ───────────────────────────────────────
    def init_controller(self, model: mj.MjModel, data: mj.MjData) -> None:
        """Called once by MujocoSim before the simulation loop."""
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

        # Load 'home' keyframe for a stable starting pose
        key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mj.mj_resetDataKeyframe(model, data, key_id)
            mj.mj_forward(model, data)
            print(f"Loaded 'home' keyframe (id={key_id})")
        else:
            print("Warning: 'home' keyframe not found, starting from default pose")

        # Print damping-ratio diagnostics now that the model is available
        self._print_damping_diagnostics()

    # ─── Damping-ratio diagnostics ──────────────────────────────
    def _estimate_joint_inertia(self, joint_name: str) -> float:
        """
        Estimate the rotational inertia I of the body attached to *joint_name*
        using MuJoCo's full composite inertia matrix M(q).

        Returns the diagonal element M[dof, dof] for the joint's single DoF,
        which captures the *effective* inertia seen at the joint including all
        child bodies.
        """
        # Compute the full mass matrix in the current configuration
        n_dof = self.model.nv
        M = np.zeros((n_dof, n_dof))
        mj.mj_fullM(self.model, M, self.data.qM)

        jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id < 0:
            print(f"Warning: joint '{joint_name}' not found; using fallback I=0.01")
            return 0.01
        dof_idx = self.model.jnt_dofadr[jnt_id]
        inertia = M[dof_idx, dof_idx]
        return max(inertia, 1e-6)  # guard against zero

    def compute_damping_ratio(self, joint_name: str) -> float:
        """
        ζ = d / (2 · √(k · I))

        Must be called after init_controller (needs the MuJoCo model).
        """
        I = self._estimate_joint_inertia(joint_name)
        if self.k <= 0:
            return float('inf')
        zeta = self.d / (2.0 * np.sqrt(self.k * I))
        return zeta

    def _print_damping_diagnostics(self) -> None:
        """Print damping ratio ζ for every knee (calf) joint."""
        calf_joints = [name for name in ACTUATOR_DICT if 'calf' in name]
        for name in calf_joints:
            zeta = self.compute_damping_ratio(name)
            I = self._estimate_joint_inertia(name)
            if zeta < 1.0:
                category = "underdamped"
            elif np.isclose(zeta, 1.0, atol=1e-3):
                category = "critically damped"
            else:
                category = "overdamped"
            print(
                f"  {name}: ζ = {zeta:.4f} ({category}), "
                f"I = {I:.5f} kg·m², k = {self.k}, d = {self.d}"
            )

    # ─── Core torque law ────────────────────────────────────────

    def apply_pd(self, targets: dict[str, float]) -> None:
        """
        Apply control to all actuators listed in *targets*.

        * Hip / thigh joints → stiff PD position tracker
        * Calf (knee) joints → passive spring-damper:  τ = k(θ_ref - θ) - d·θ̇
        """
        for joint_name, actuator_idx in ACTUATOR_DICT.items():
            if joint_name not in targets:
                continue

            theta = self.data.sensordata[SENSOR_POS_DICT[joint_name]]
            theta_dot = self.data.sensordata[SENSOR_VEL_DICT[joint_name]]
            theta_ref = targets[joint_name]
            
            if 'calf' in joint_name:
                # ── Knee: passive spring-damper (muscle-like compliance) ──
                #    τ = k · (θ_ref − θ) − d · θ̇
                #  No velocity reference — the damper resists all motion.
                torque = self.k * (theta_ref - theta) - self.d * theta_dot
                print(f"Calf: {joint_name}, velocity={theta_dot}")
            else:
                # ── Hip / thigh: stiff PD tracker ────────────────────────
                #    τ = kp · (θ_ref − θ) + kd · (0 − θ̇)
                torque = (
                    self.passive_kp * (theta_ref - theta)
                    + self.passive_kd * (0.0 - theta_dot)
                )

            self.data.ctrl[actuator_idx] = torque

    # ─── Simulation-state helpers ───────────────────────────────

    def _save_sim_state(self) -> tuple:
        """Snapshot qpos, qvel, ctrl, time so we can restore later."""
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
            self.data.time,
        )

    def _restore_sim_state(self, state: tuple) -> None:
        """Restore a snapshot produced by _save_sim_state."""
        qpos, qvel, ctrl, t = state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self.data.time = t
        mj.mj_forward(self.model, self.data)

    def _disable_other_joints(self, keep_joint: str) -> None:
        """Zero the control signal for every actuator except *keep_joint*."""
        keep_idx = ACTUATOR_DICT.get(keep_joint, -1)
        for idx in range(self.model.nu):
            if idx != keep_idx:
                self.data.ctrl[idx] = 0.0

    def _optionally_disable_contacts(self, disable: bool) -> Optional[int]:
        """
        If *disable* is True, turn off all contacts.
        Returns the original flag value so it can be restored.
        """
        if not disable:
            return None
        original = self.model.opt.disableflags
        self.model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_CONTACT
        return original

    def _restore_contacts(self, original_flags: Optional[int]) -> None:
        if original_flags is not None:
            self.model.opt.disableflags = original_flags

    # ─── Step-response test ─────────────────────────────────────

    def step_response(
        self,
        joint_name: str,
        step_size: float = 0.5,
        duration: float = 2.0,
        disable_other_joints: bool = True,
        disable_contacts: bool = True,
    ) -> 'MusclePdController.TestRecord':
        """
        Apply a constant step in θ_ref and record the transient response.

        The reference is held constant for the entire *duration* — this is a
        true step input, not a one-timestep impulse.

        Parameters
        ----------
        joint_name : str
            Must be a key in ACTUATOR_DICT (e.g. 'front_left_calf').
        step_size : float
            Magnitude of the step in radians (added to the initial position).
        duration : float
            How long to record the response (seconds).
        disable_other_joints : bool
            If True, zero all other actuators during the test.
        disable_contacts : bool
            If True, disable ground contacts so the leg swings freely.

        Returns
        -------
        TestRecord
            Contains time, θ, θ̇, θ_ref, and τ arrays.
        """
        if self.model is None:
            raise RuntimeError("Call init_controller() before running tests.")
        if joint_name not in ACTUATOR_DICT:
            raise ValueError(f"Unknown joint name: '{joint_name}'")

        actuator_idx = ACTUATOR_DICT[joint_name]
        pos_idx = SENSOR_POS_DICT[joint_name]
        vel_idx = SENSOR_VEL_DICT[joint_name]

        # Save / prepare simulation state
        saved_state = self._save_sim_state()
        orig_flags = self._optionally_disable_contacts(disable_contacts)

        # Compute the constant step reference from the current position
        theta_0 = self.data.sensordata[pos_idx]
        theta_ref = theta_0 + step_size

        record = self.TestRecord()
        start_time = self.data.time

        while self.data.time - start_time < duration:
            theta = self.data.sensordata[pos_idx]
            theta_dot = self.data.sensordata[vel_idx]

            # Spring-damper torque:  τ = k(θ_ref − θ) − d·θ̇
            torque = self.k * (theta_ref - theta) - self.d * theta_dot

            if disable_other_joints:
                self._disable_other_joints(joint_name)
            self.data.ctrl[actuator_idx] = torque

            # Log BEFORE stepping so t=0 captures initial conditions
            record.time.append(self.data.time - start_time)
            record.theta.append(theta)
            record.theta_dot.append(theta_dot)
            record.theta_ref.append(theta_ref)
            record.torque.append(torque)

            mj.mj_step(self.model, self.data)

        # Restore original simulation state
        self._restore_contacts(orig_flags)
        self._restore_sim_state(saved_state)

        return record

    # ─── Frequency-response test ────────────────────────────────

    def frequency_response(
        self,
        joint_name: str,
        drive_freq_hz: float,
        amplitude: float = 0.3,
        n_cycles: int = 10,
        settle_cycles: int = 3,
        disable_other_joints: bool = True,
        disable_contacts: bool = True,
    ) -> tuple['MusclePdController.TestRecord', dict]:
        """
        Drive θ_ref with a sinusoid at the CPG frequency and measure the
        steady-state amplitude ratio and phase lag.

        Parameters
        ----------
        joint_name : str
            Knee joint to test (must contain 'calf').
        drive_freq_hz : float
            Sinusoidal drive frequency in Hz (typically the CPG frequency).
        amplitude : float
            Peak amplitude of the sinusoidal reference [rad].
        n_cycles : int
            Total number of drive cycles to simulate.
        settle_cycles : int
            Initial cycles to discard (transient die-out).
        disable_other_joints / disable_contacts : bool
            Same as in step_response.

        Returns
        -------
        record : TestRecord
            Full time-series (including transient).
        metrics : dict
            'amplitude_ratio': |θ| / |θ_ref|
            'phase_lag_rad':   phase lag in radians (positive = lag)
            'phase_lag_deg':   phase lag in degrees
            'drive_freq_hz':   the drive frequency used
        """
        if self.model is None:
            raise RuntimeError("Call init_controller() before running tests.")
        if joint_name not in ACTUATOR_DICT:
            raise ValueError(f"Unknown joint name: '{joint_name}'")

        actuator_idx = ACTUATOR_DICT[joint_name]
        pos_idx = SENSOR_POS_DICT[joint_name]
        vel_idx = SENSOR_VEL_DICT[joint_name]

        saved_state = self._save_sim_state()
        orig_flags = self._optionally_disable_contacts(disable_contacts)

        omega = 2.0 * np.pi * drive_freq_hz  # angular frequency [rad/s]
        duration = n_cycles / drive_freq_hz
        theta_0 = self.data.sensordata[pos_idx]  # DC offset = current pos

        record = self.TestRecord()
        start_time = self.data.time

        while self.data.time - start_time < duration:
            t_rel = self.data.time - start_time
            theta_ref = theta_0 + amplitude * np.sin(omega * t_rel)

            theta = self.data.sensordata[pos_idx]
            theta_dot = self.data.sensordata[vel_idx]

            torque = self.k * (theta_ref - theta) - self.d * theta_dot

            if disable_other_joints:
                self._disable_other_joints(joint_name)
            self.data.ctrl[actuator_idx] = torque

            record.time.append(t_rel)
            record.theta.append(theta)
            record.theta_dot.append(theta_dot)
            record.theta_ref.append(theta_ref)
            record.torque.append(torque)

            mj.mj_step(self.model, self.data)

        # Restore simulation
        self._restore_contacts(orig_flags)
        self._restore_sim_state(saved_state)

        # ── Compute amplitude ratio & phase lag on steady-state portion ──
        t_arr = np.array(record.time)
        ref_arr = np.array(record.theta_ref) - theta_0  # remove DC
        pos_arr = np.array(record.theta) - theta_0

        settle_time = settle_cycles / drive_freq_hz
        mask = t_arr >= settle_time

        ref_ss = ref_arr[mask]
        pos_ss = pos_arr[mask]
        t_ss = t_arr[mask]

        # Amplitude ratio from peak values
        ref_amp = (np.max(ref_ss) - np.min(ref_ss)) / 2.0
        pos_amp = (np.max(pos_ss) - np.min(pos_ss)) / 2.0
        amplitude_ratio = pos_amp / ref_amp if ref_amp > 1e-9 else 0.0

        # Phase lag via cross-correlation
        # Normalise signals for correlation
        ref_norm = ref_ss - np.mean(ref_ss)
        pos_norm = pos_ss - np.mean(pos_ss)
        correlation = np.correlate(pos_norm, ref_norm, mode='full')
        lags = np.arange(-len(ref_norm) + 1, len(ref_norm))
        dt_avg = np.mean(np.diff(t_ss))
        best_lag_idx = np.argmax(correlation)
        time_lag = lags[best_lag_idx] * dt_avg  # seconds

        phase_lag_rad = omega * time_lag  # convert to radians
        # Normalise into [-π, π]; positive = output lags input
        phase_lag_rad = (phase_lag_rad + np.pi) % (2 * np.pi) - np.pi
        phase_lag_deg = np.degrees(phase_lag_rad)

        metrics = {
            'amplitude_ratio': amplitude_ratio,
            'phase_lag_rad': phase_lag_rad,
            'phase_lag_deg': phase_lag_deg,
            'drive_freq_hz': drive_freq_hz,
        }

        print(
            f"  {joint_name} @ {drive_freq_hz:.2f} Hz: "
            f"amp_ratio = {amplitude_ratio:.3f}, "
            f"phase_lag = {phase_lag_deg:+.1f}°"
        )

        return record, metrics

    # ─── CSV logging ────────────────────────────────────────────

    @staticmethod
    def save_record_csv(
        record: 'MusclePdController.TestRecord',
        filepath: str,
    ) -> None:
        """
        Write a TestRecord to CSV with columns:
        time, theta, theta_dot, theta_ref, torque
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'theta', 'theta_dot', 'theta_ref', 'torque'])
            for row in zip(
                record.time,
                record.theta,
                record.theta_dot,
                record.theta_ref,
                record.torque,
            ):
                writer.writerow(row)
        print(f"Saved test data → {filepath}")

    # ─── Plotting helpers ───────────────────────────────────────

    @staticmethod
    def plot_step_response(
        record: 'MusclePdController.TestRecord',
        joint_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot position & velocity transient for a step-response test."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(record.time, record.theta, label='θ')
        axes[0].axhline(record.theta_ref[0], color='r', ls='--', label='θ_ref')
        axes[0].set(title=f'{joint_name} - position', xlabel='Time [s]', ylabel='θ [rad]')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(record.time, record.theta_dot)
        axes[1].set(title=f'{joint_name} - velocity', xlabel='Time [s]', ylabel='θ̇ [rad/s]')
        axes[1].grid(True)

        axes[2].plot(record.time, record.torque)
        axes[2].set(title=f'{joint_name} - torque', xlabel='Time [s]', ylabel='τ [N·m]')
        axes[2].grid(True)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved plot → {save_path}")
        plt.show()

    @staticmethod
    def plot_frequency_response(
        record: 'MusclePdController.TestRecord',
        joint_name: str,
        metrics: dict,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot θ_ref vs θ and annotate amplitude ratio / phase lag."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].plot(record.time, record.theta_ref, label='θ_ref', alpha=0.7)
        axes[0].plot(record.time, record.theta, label='θ', alpha=0.9)
        axes[0].set(
            title=f'{joint_name} - frequency response @ {metrics["drive_freq_hz"]:.2f} Hz',
            xlabel='Time [s]', ylabel='Angle [rad]',
        )
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(record.time, record.torque)
        axes[1].set(title=f'{joint_name} - torque', xlabel='Time [s]', ylabel='τ [N·m]')
        axes[1].grid(True)

        fig.suptitle(
            f"Amp ratio = {metrics['amplitude_ratio']:.3f},  "
            f"Phase lag = {metrics['phase_lag_deg']:+.1f}°",
            fontsize=10,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved plot → {save_path}")
        plt.show()