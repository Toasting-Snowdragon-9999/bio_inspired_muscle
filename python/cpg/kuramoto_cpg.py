import os, sys
import numpy as np
import mujoco as mj
from dataclasses import dataclass, field
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *
from pd.muscle_like_pd import MusclePdController

class KuramotoCpg:
    """
    Kuramoto (phase) oscillator CPG controller for MuJoCo quadruped.

    4 coupled oscillators, one per hip. Each has a single phase variable θ_i:
        dθ_i/dt = ω_i + Σ_j  w_ij · sin(θ_j - θ_i - φ_ij)

    Hip (thigh) output:  standing_target + amplitude * sin(θ_i)
    Knee (calf) output:  standing_target + amplitude * sin(θ_i + knee_phase_offset)

    Designed to be used as a MuJoCo controller via init_controller() / run().
    """

    @dataclass
    class Neuron:
        phase: float        # θ
        frequency: float    # ω
        amplitude: float    # A

    def __init__(self, gait: str = 'trot', frequency:float = 1.0, warmup_seconds: float = 2.0, pd_controller: MusclePdController = None) -> None:
        self.neurons_cnt = 4  # one oscillator per hip (FL, FR, RR, RL)
        self.model = None
        self.data = None
        self.dt = None  # read from MuJoCo model at init_controller
        self.warmup_seconds = warmup_seconds
        self.sim_time = 0.0

        self.coupling_weights = np.zeros((self.neurons_cnt, self.neurons_cnt))
        hip_coupling_strength = 1.5 * np.pi
        for i in range(self.neurons_cnt):
            for j in range(self.neurons_cnt):
                if i != j:
                    self.coupling_weights[i, j] = hip_coupling_strength

        # Knee/calf follows its corresponding hip, shifted by this offset
        self.knee_phase_offset = -np.pi

        self.gait_name = gait
        hip_phases = GAITS[gait]  # length 4
        self.phase_offsets = np.zeros((self.neurons_cnt, self.neurons_cnt))
        self._build_phase_offset_matrix(hip_phases)

        self.neurons = [
            self.Neuron(
                phase=hip_phases[i],
                frequency=frequency,
                amplitude=1.0,
            )
            for i in range(self.neurons_cnt)
        ]

        self.d_theta = np.zeros(self.neurons_cnt)

        self.hip_amplitude = 0.4
        self.knee_amplitude = 1.2

        self.standing_targets = {
            'front_left_thigh':  0.8,
            'front_right_thigh': 0.8,
            'rear_left_thigh':   1.0,
            'rear_right_thigh':  1.0,
            'front_left_calf':  -1.5,
            'front_right_calf': -1.5,
            'rear_left_calf':   -1.5,
            'rear_right_calf':  -1.5,
        }

        # Static target positions for non-CPG joints (abduction)
        self.static_targets = {
            'front_right_hip': 0.0,
            'front_left_hip': 0.0,
            'rear_right_hip': 0.0,
            'rear_left_hip': 0.0,
        }
        if pd_controller is None:
            # k, d are the knee spring–damper parameters;
            # passive_kp, passive_kd are the stiff hip PD gains.
            self.pd_controller = MusclePdController(
                k=40.0, d=4.0, passive_kp=40.0, passive_kd=4.0
            )
        else: 
            self.pd_controller = pd_controller

    def set_gait(self, gait_name: str):
        if gait_name not in GAITS:
            raise ValueError(
                f"Gait '{gait_name}' not recognized. "
                f"Available gaits: {list(GAITS.keys())}"
            )

        self.gait_name = gait_name
        gait_phases = GAITS[gait_name]
        self._build_phase_offset_matrix(gait_phases)

    def set_frequency(self, frequency: float, index: int = None):
        if index is None:
            for n in self.neurons:
                n.frequency = frequency
        else:
            self.neurons[index].frequency = frequency

    def _build_phase_offset_matrix(self, desired_phases: float):
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
                            self.coupling_weights[i, j] * np.sin(thetas[j] - thetas[i] - self.phase_offsets[i, j])
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

    def init_controller(self, model, data):
        """Called once by MujocoSim before the simulation loop."""
        self.model = model
        self.data = data
        self.dt = model.opt.timestep  # MuJoCo simulation dt
        self.sim_time = 0.0

        # Load the 'home' keyframe so the robot starts in a stable standing pose
        key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mj.mj_resetDataKeyframe(model, data, key_id)
            mj.mj_forward(model, data)
            print(f"Loaded 'home' keyframe (id={key_id})")
        else:
            print("Warning: 'home' keyframe not found, starting from default pose")

        self.pd_controller.init_controller(model=model, data=data)

    def run(self, model, data):
        """
        Called every MuJoCo timestep by mj.set_mjcb_control.
        Steps the CPG then applies PD control to all joints.
        """
        self.sim_time = data.time

        if self.sim_time < self.warmup_seconds:
            targets = dict(self.static_targets)
            targets.update(self.standing_targets)
            self.pd_controller.apply_pd(targets)
            return

        blend_duration = 1.0  # seconds
        t_since_warmup = self.sim_time - self.warmup_seconds
        blend = min(1.0, t_since_warmup / blend_duration)

        self.rk4_integration(self.dt)

        cpg_targets = {}
        for osc_idx in range(self.neurons_cnt):
            phase = self.neurons[osc_idx].phase
            
            # Thigh target: standing - amplitude * sin(phase)  (negative = forward)
            thigh_name = OSCILLATOR_TO_THIGH[osc_idx]
            thigh_stand = self.standing_targets[thigh_name]
            thigh_cpg = thigh_stand - self.hip_amplitude * np.sin(phase)
            cpg_targets[thigh_name] = thigh_stand + blend * (thigh_cpg - thigh_stand)

            calf_name = OSCILLATOR_TO_CALF[osc_idx]
            calf_stand = self.standing_targets[calf_name]
            knee_signal = min(0.0, np.sin(phase + self.knee_phase_offset))
            calf_cpg = calf_stand + self.knee_amplitude * knee_signal
            cpg_targets[calf_name] = calf_stand + blend * (calf_cpg - calf_stand)


        # 3. Build full target dict: static joints + CPG-driven joints
        targets = dict(self.static_targets)
        targets.update(cpg_targets)

        # 4. PD control
        self.pd_controller.apply_pd(targets)


    def get_oscillator_outputs(self) -> np.ndarray:
        """
        Return current output signal for each hip and its corresponding knee.
        Returns:
            hip_outputs:  np.array of shape (4,)  - -sin(θ_i) (matches hip output sign)
            knee_outputs: np.array of shape (4,)  - clamped knee signal
        """
        hip_out = np.zeros(self.neurons_cnt)
        knee_out = np.zeros(self.neurons_cnt)
        for i in range(self.neurons_cnt):
            phase = self.neurons[i].phase
            hip_out[i] = -np.sin(phase)
            knee_out[i] = min(0.0, np.sin(phase + self.knee_phase_offset))
        return hip_out, knee_out