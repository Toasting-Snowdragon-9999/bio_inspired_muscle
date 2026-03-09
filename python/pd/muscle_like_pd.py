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

        self.start_time = None
        self.settle_duration = 0.2
        self.stand_duration = 1.0


    # MuJoCo bootstrap
    def init_controller(self, model: mj.MjModel, data: mj.MjData) -> None:
        """Called once by MujocoSim before the simulation loop."""
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.start_time = data.time


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

    def apply_pd(self, targets: dict[str, float], phases) -> None:
        """
        Apply control to all actuators listed in *targets*.

        * Hip / thigh joints → stiff PD position tracker
        * Calf (knee) joints → passive spring-damper:  τ = k(θ_ref - θ) - d·θ̇
        """
        tau_max_hip = 23.7
        tau_max_knee = 45.43

        phases = np.array([x % (2*np.pi) for x in phases]) # wrap phases

        # actuator idx
        # hips: 0, 3, 6, 9
        # thighs: 1, 4, 7, 10
        # calves: 2, 5, 8, 11

        # current_time = self.data.time

        # # ---- SETTLE PHASE ----
        # if current_time - self.start_time < self.settle_duration:
        #     self.data.ctrl[:] = 0.0
        #     return

        # # ---- STAND PHASE ----
        # elif current_time - self.start_time < self.settle_duration + self.stand_duration:
        #     stand_mode = True
        # else:
        #     stand_mode = False

        # for joint_name, actuator_idx in ACTUATOR_DICT.items():
        #     if joint_name not in targets or 'calf' in joint_name:
        #         # Calf (knee) control - coordinated with thigh position
        #         current_thigh_angle = self.data.sensordata[actuator_idx-1]
        #         calf_ref = self.compute_knee_reference_from_phase(current_thigh_angle, joint_name)
        #         leg_idx = actuator_idx // 3
        #         if stand_mode:
        #             kp = 100
        #             kd = 20
        #         else:
        #             kp, kd = self.get_knee_impedance(phases[leg_idx], joint_name)

        #         tau_calf = self.control_eq(actuator_idx, calf_ref, kp, kd)
        #         self.data.ctrl[actuator_idx] = np.clip(tau_calf, -tau_max_knee, tau_max_knee)

        for joint_name, actuator_idx in ACTUATOR_DICT.items():
            if joint_name not in targets or 'calf' in joint_name:
                kp = 100
                kd = 20
                current_thigh_angle = self.data.sensordata[actuator_idx-1]
                calf_ref = self.compute_knee_reference_from_phase(current_thigh_angle, joint_name)
                tau_calf = self.control_eq(actuator_idx, calf_ref, kp, kd)
                self.data.ctrl[actuator_idx] = np.clip(tau_calf, -tau_max_knee, tau_max_knee)


            elif 'thigh' in joint_name or 'hip' in joint_name:
                # Thigh / hip control
                thigh_ref = targets[joint_name]
                # tau_thigh = self.control_eq(actuator_idx, thigh_ref, self.passive_kp, self.passive_kd)
                # if stand_mode:
                #     tau_thigh = self.control_eq(actuator_idx, thigh_ref, 200, 15)
                # else:
                #     tau_thigh = self.control_eq(actuator_idx, thigh_ref, self.passive_kp, self.passive_kd)

                tau_thigh = self.control_eq(actuator_idx, thigh_ref, self.passive_kp, self.passive_kd)

                self.data.ctrl[actuator_idx] = np.clip(tau_thigh, -tau_max_hip, tau_max_hip)

    def control_eq(self, joint_id, ref, kp, kd):

        return - kp * (self.data.sensordata[joint_id] - ref) - kd * self.data.sensordata[joint_id + 12]
    
    # def get_knee_impedance(self, phase):
    #     s = 0.5 * (1 + np.cos(phase - np.pi))
    #     kp_stance_max = 400 # 200-400
    #     kp_stance_mod = 180 # 80-180
    #     kp_stance_min = 60 # 20-60

    #     kd_stance_max = 10 # 6-10
    #     kd_stance_mod = 5 # 2-5 
    #     kd_stance_min = 1 # 0.3-1

    #     kp = kp_stance_min + (kp_stance_max - kp_stance_min) * s
    #     kd = kd_stance_min + (kd_stance_max - kd_stance_min) * s

    #     return kp, kd
    
    # def get_knee_impedance(self, phi):
    #     if phi <= np.pi:  # stance
    #         s = 0.5 * (1 + np.cos(phi - np.pi))
    #         d = 0.5 * (1 + np.cos(phi))

    #         kp = self.k_stance_min + \
    #             (self.k_stance_max - self.k_stance_min) * s

    #         kd = self.k_d_stance_min + \
    #             (self.k_d_stance_max - self.k_d_stance_min) * d
    #     else:  # swing
    #         kp = self.k_swing
    #         kd = self.k_d_swing

    #     return kp, kd

    # def get_knee_impedance(self, phase: float, joint_name: str):
    #     """
    #     Phase-dependent knee impedance with front/rear differentiation.
    #     """

    #     # ---- Front vs Rear tuning ----

    #     if joint_name in FRONT_LEGS:
    #         # Front legs: more vertical support, slightly more damping
    #         KP_STANCE_MIN = 180
    #         KP_STANCE_MAX = 420

    #         KD_STANCE_MIN = 8
    #         KD_STANCE_MAX = 18

    #         KP_SWING = 15
    #         KD_SWING = 2

    #     else:
    #         # Rear legs: more propulsion, slightly more compliant
    #         KP_STANCE_MIN = 130
    #         KP_STANCE_MAX = 380

    #         KD_STANCE_MIN = 5
    #         KD_STANCE_MAX = 12

    #         KP_SWING = 8
    #         KD_SWING = 1

    #     # ---- Phase logic ----

    #     if phase <= np.pi:
    #         stance_progress = phase / np.pi

    #         # Cosine peak at mid-stance
    #         s = 0.5 * (1 - np.cos(2 * np.pi * stance_progress))

    #         kp = KP_STANCE_MIN + (KP_STANCE_MAX - KP_STANCE_MIN) * s
    #         kd = KD_STANCE_MIN + (KD_STANCE_MAX - KD_STANCE_MIN) * s

    #     else:
    #         kp = KP_SWING
    #         kd = KD_SWING

    #     return kp, kd



    def compute_knee_reference_from_phase(self, phase, joint_name):

        # Define swing window: phase > π
        if phase > np.pi:
            swing_progress = (phase - np.pi) / np.pi
        else:
            swing_progress = 0.0

        swing_progress = np.clip(swing_progress, 0.0, 1.0)

        knee_stance = 0
        knee_swing  = 0
        if joint_name in FRONT_LEGS:
            knee_stance = -1.55 # more extended
            knee_swing  = -2.2
        else:
            knee_stance = -1.8 # more crouched
            knee_swing  = -2.4
        
        # with these settings: rear legs are:
        # ~0.25 rad (~14°) more flexed in stance
        # ~0.2 rad (~11°) more flexed in swing

        return knee_stance + swing_progress * (knee_swing - knee_stance)


