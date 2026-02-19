"""
Muscle-like PD controller for quadruped knee joints.

Physics model
─────────────
Hip/thigh joints use stiff position-tracking PD control (high gains) so the
CPG can prescribe their trajectories directly.

Knee (calf) joints are modelled as passive spring-damper systems:

    τ = k · (θ_ref - θ) - d · θ̇

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

    def __init__(
        self,
        kp: float = 10.0,
        kd: float = 4.0,
        passive_kp: float = 40.0,
        passive_kd: float = 4.0,
    ) -> None:
        """
        Parameters
        ----------
        k : float           ← Knee spring stiffness  [N·m / rad].
        d : float           ← Knee viscous damping   [N·m·s / rad].
        passive_kp : float  ← Hip position gain (stiff tracker).
        passive_kd : float  ← Hip derivative gain (stiff tracker).
        """
        # Knee: passive spring-damper parameters
        if kd < 0.0:
            raise self.InvalidDampingError(
                f"Damping coefficient d must be ≥ 0, got {d}"
            )
        self.kp: float = kp   # spring stiffness
        self.kd: float = kd   # viscous damping

        # Hip / thigh: stiff PD tracker
        self.passive_kp: float = passive_kp
        self.passive_kd: float = passive_kd

        # Simulation handles (set in init_controller)
        self.model: Optional[mj.MjModel] = None
        self.data: Optional[mj.MjData] = None
        self.dt: Optional[float] = None

    # MuJoCo bootstrap
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


    # Joint inertia
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
        """ ζ = d / (2 · √(k · I)) 
        \nMust be called after init_controller (needs the MuJoCo model). """

        I = self._estimate_joint_inertia(joint_name)
        if self.kp <= 0:
            return float('inf')
        zeta = self.kd / (2.0 * np.sqrt(self.kp * I))
        return zeta

    # Core torque law 

    # def apply_pd(self, targets: dict[str, float]) -> None:
    #     """
    #     Apply control to all actuators listed in *targets*.

    #     * Hip / thigh joints → stiff PD position tracker
    #     * Calf (knee) joints → passive spring-damper:  τ = k(θ_ref - θ) - d·θ̇
    #     """
    #     for joint_name, actuator_idx in ACTUATOR_DICT.items():
    #         if joint_name not in targets:
    #             continue

    #         theta = self.data.sensordata[SENSOR_POS_DICT[joint_name]]
    #         theta_dot = self.data.sensordata[SENSOR_VEL_DICT[joint_name]]
    #         theta_ref = targets[joint_name]
            
    #         if 'calf' in joint_name:
    #             # Knee: passive spring-damper (muscle-like compliance)
    #             #    τ = k · (θ_ref − θ) − d · θ̇
    #             #  No velocity reference — the damper resists all motion.
    #             torque = self.p * (theta_ref - theta) - self.d * theta_dot
    #             print(f"Calf: {joint_name}, velocity={theta_dot}")
    #         else:
    #             # Hip / thigh: stiff PD tracker
    #             #    τ = kp · (θ_ref − θ) + kd · (0 − θ̇)
    #             torque = (
    #                 self.passive_kp * (theta_ref - theta)
    #                 + self.passive_kd * (0.0 - theta_dot)
    #             )

    #         self.data.ctrl[actuator_idx] = torque

    def apply_pd(self, targets: dict[str, float]) -> None:
        """
        Apply control to all actuators listed in *targets*.

        * Hip / thigh joints → stiff PD position tracker
        * Calf (knee) joints → passive spring-damper:  τ = k(θ_ref - θ) - d·θ̇
        """
        tau_max_hip = 23.7
        tau_max_knee = 45.43

        for joint_name, actuator_idx in ACTUATOR_DICT.items():
            if joint_name not in targets:
                # Calf (knee) control - coordinated with thigh position
                current_thigh_angle = self.data.sensordata[actuator_idx-1]
                calf_ref = self.get_knee_trajectory(current_thigh_angle)
                tau_calf = self.control_eq(actuator_idx, calf_ref, self.kp, self.kd)
                self.data.ctrl[actuator_idx] = np.clip(tau_calf, -tau_max_knee, tau_max_knee)
            elif 'thigh' in joint_name or 'hip' in joint_name:
                # Thigh / hip control
                thigh_ref = targets[joint_name]
                tau_thigh = self.control_eq(actuator_idx, thigh_ref, self.passive_kp, self.passive_kd)
                self.data.ctrl[actuator_idx] = np.clip(tau_thigh, -tau_max_hip, tau_max_hip)

    def control_eq(self, joint_id, ref, kp, kd):

        return - kp * (self.data.sensordata[joint_id] - ref) - kd * self.data.sensordata[joint_id + 12]
    
    def get_knee_trajectory(self, thigh_angle):
        """
        Generate knee angle based on thigh angle for natural walking motion.
        
        When thigh swings forward (larger angle ~1.5-2.0): knee bends more (more negative, ~-2.0 to -2.2) to lift foot
        When thigh is neutral/back (smaller angle ~0.5-1.0): knee extends (less negative, ~-1.4 to -1.6) for stance
        
        Args:
            thigh_angle: Current thigh joint angle in radians
            
        Returns:
            Target knee angle in radians
        """
        # Map thigh angle to knee angle
        # Thigh range: ~0.5 (back) to ~2.0 (forward)
        # Knee range: -1.4 (extended) to -2.2 (flexed)
        
        # Normalize thigh angle to 0-1 range
        thigh_min, thigh_max = 0.5, 1.5
        t = np.clip((thigh_angle - thigh_min) / (thigh_max - thigh_min), 0, 1)
        
        # When thigh forward (t=1): knee MUCH MORE flexed (-2.2) to lift foot high
        # When thigh back (t=0): knee extended (-1.4) for stance/push
        knee_extended = -1.4  # Less bent for stance
        knee_flexed = -2.2    # More bent for swing phase
        
        return knee_extended + t * (knee_flexed - knee_extended)

    # leg_order = [0, 1, 2, 3]  # FR, FL, RR, RL
    # # all 4 legs
    # for i, leg_id in enumerate(leg_order):
    #     hip_id = leg_id * 3 + 0     # hips: 0, 3, 6, 9
    #     thigh_id = leg_id * 3 + 1   # thighs: 1, 4, 7, 10
    #     calf_id = leg_id * 3 + 2    # calves: 2, 5, 8, 11

    #     tau_hip = self.control_eq(hip_id, 0)
    #     self.data.ctrl[hip_id] = np.clip(tau_hip, -tau_max_hip, tau_max_hip)

    #     # Thigh control - trot gait pattern
    #     thigh_ref = self.get_thigh_trot_trajectory(leg_id)
    #     thigh_ref = 0.8
    #     if leg_id == 2 or leg_id == 3:
    #         thigh_ref = 1.0
    #     tau_thigh = self.control_eq(thigh_id, thigh_ref)
    #     self.data.ctrl[thigh_id] = np.clip(tau_thigh, -tau_max_hip, tau_max_hip)
        
    #     # Calf (knee) control - coordinated with thigh position
    #     current_thigh_angle = self.data.sensordata[thigh_id]
    #     calf_ref = self.get_knee_trajectory(current_thigh_angle)
    #     tau_calf = self.control_eq(calf_id, calf_ref)
    #     self.data.ctrl[calf_id] = np.clip(tau_calf, -tau_max_knee, tau_max_knee)