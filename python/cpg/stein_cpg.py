import numpy as np
import mujoco as mj
from dataclasses import dataclass, field


class SteinCpg:
    """
    Stein (phase) oscillator CPG controller for MuJoCo quadruped.

    Each oscillator has a single phase variable θ_i:
        dθ_i/dt = ω_i + Σ_j  w_ij · sin(θ_j - θ_i - φ_ij) + b*f_i*cos(θ_i) # Although the external input is not used yet

    Output per oscillator uses stance/swing duty cycle:
        φ = θ mod 2π
        if φ < duty_cycle * 2π  →  stance phase
        else                    →  swing phase

    Designed to be used as a MuJoCo controller via init_controller() / run().
    """

    # ── Joint ranges (radians) ──────────────────────────────────────
    KNEE_POS_RANGE      = (-2.7227, -0.83776)
    FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)
    BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)
    ABDUCTION_POS_RANGE = (-1.0472, 1.0472)

    # ── Actuator / sensor index maps ────────────────────────────────
    ACTUATOR_DICT = {
        'front_right_hip': 0,   'front_right_thigh': 1,  'front_right_calf': 2,
        'front_left_hip': 3,    'front_left_thigh': 4,   'front_left_calf': 5,
        'rear_right_hip': 6,    'rear_right_thigh': 7,   'rear_right_calf': 8,
        'rear_left_hip': 9,     'rear_left_thigh': 10,   'rear_left_calf': 11,
    }

    SENSOR_POS_DICT = {
        'front_right_hip': 0,   'front_right_thigh': 1,  'front_right_calf': 2,
        'front_left_hip': 3,    'front_left_thigh': 4,   'front_left_calf': 5,
        'rear_right_hip': 6,    'rear_right_thigh': 7,   'rear_right_calf': 8,
        'rear_left_hip': 9,     'rear_left_thigh': 10,   'rear_left_calf': 11,
    }

    SENSOR_VEL_DICT = {
        'front_right_hip': 12,  'front_right_thigh': 13, 'front_right_calf': 14,
        'front_left_hip': 15,   'front_left_thigh': 16,  'front_left_calf': 17,
        'rear_right_hip': 18,   'rear_right_thigh': 19,  'rear_right_calf': 20,
        'rear_left_hip': 21,    'rear_left_thigh': 22,   'rear_left_calf': 23,
    }

    # ── CPG oscillator → joint mapping ──────────────────────────────
    # Neurons 0-3: hip (thigh) oscillators   FL, FR, RR, RL
    # Neurons 4-7: knee (calf) oscillators   FL, FR, RR, RL
    OSCILLATOR_TO_THIGH = {
        0: 'front_left_thigh',
        1: 'front_right_thigh',
        2: 'rear_right_thigh',
        3: 'rear_left_thigh',
    }

    OSCILLATOR_TO_CALF = {
        4: 'front_left_calf',
        5: 'front_right_calf',
        6: 'rear_right_calf',
        7: 'rear_left_calf',
    }

    # Which knee neuron is coupled to which hip neuron
    KNEE_TO_HIP = {4: 0, 5: 1, 6: 2, 7: 3}

    # ── Preset gaits ────────────────────────────────────────────────
    GAITS = {
        'trot':   np.array([0.0,       np.pi,     0.0,       np.pi]),
        'walk':   np.array([0.0,       np.pi/2,   np.pi,     3*np.pi/2]),
        'bound':  np.array([0.0,       0.0,       np.pi,     np.pi]),
        'pace':   np.array([0.0,       np.pi,     np.pi,     0.0]),
        'gallop': np.array([0.0,       0.0,       np.pi*0.8, np.pi*0.8]),
    } 

    @dataclass
    class Neuron:
        phase: float        # θ
        frequency: float    # ω
        amplitude: float    # A

    # ────────────────────────────────────────────────────────────────
    def __init__(self, gait='trot', frequency=1.0, warmup_seconds=2.0,
                 hip_knee_phase_offset=np.pi, duty_cycle=0.65):
        self.neurons_cnt = 8  # 0-3 hip, 4-7 knee
        self.model = None
        self.data = None
        self.dt = None  # read from MuJoCo model at init_controller
        self.warmup_seconds = warmup_seconds
        self.sim_time = 0.0
        self.hip_knee_phase_offset = hip_knee_phase_offset  # knee leads/lags hip
        self.duty_cycle = duty_cycle  # fraction of cycle in stance (0-1)

        self.coupling_weights = np.zeros((8, 8))

        # Hip-hip coupling (0–3) 
        hip_coupling_strength = 1.5 * np.pi

        for i in range(4):
            for j in range(4):
                if i != j:
                    self.coupling_weights[i,j] = hip_coupling_strength

        # Hip-knee diagonal coupling
        hip_knee_strength = 2 * np.pi

        for knee_idx, hip_idx in self.KNEE_TO_HIP.items():
            self.coupling_weights[knee_idx, hip_idx] = hip_knee_strength
            self.coupling_weights[hip_idx, knee_idx] = hip_knee_strength


        # Phase offset matrix (built from gait + hip-knee offset)
        self.gait_name = gait
        self.phase_offsets = np.zeros((self.neurons_cnt, self.neurons_cnt))

        hip_phases = self.GAITS[gait]  # length 4
        # Knee initial phases = hip phase + offset
        knee_phases = hip_phases + hip_knee_phase_offset
        all_phases = np.concatenate([hip_phases, knee_phases])

        self.neurons = [
            self.Neuron(
                phase=all_phases[i],
                frequency=frequency,
                amplitude=1.0,
            )
            for i in range(self.neurons_cnt)
        ]
        self._build_phase_offset_matrix(all_phases)

        self.kp = 40.0
        self.kd = 4.0

        # Static target positions for non-CPG joints
        self.static_targets = {
            # Abduction (hip roll) – keep legs straight under body
            'front_right_hip': 0.0,
            'front_left_hip': 0.0,
            'rear_right_hip': 0.0,
            'rear_left_hip': 0.0,
        }

        # Thigh (hip) stance/swing targets
        # During stance: hip sweeps backward (push-off), angle increases
        # During swing:  hip moves forward (leg recovery), angle decreases
        self.front_thigh_stance = (0.6, 1.1)  # (start, end) of stance: forward → back
        self.front_thigh_swing  = (1.1, 0.6)  # (start, end) of swing:  back → forward
        self.rear_thigh_stance  = (0.6, 1.3)  # rear: wider push-off
        self.rear_thigh_swing   = (1.3, 0.4)  # rear: much more forward swing to clear ground

        # Knee (calf) stance/swing targets
        # During stance: knee stays mostly extended (stable support)
        # During swing:  knee flexes mid-swing (sinusoidal), extends before touchdown
        self.front_calf_stance = (-1.7, -1.8)       # near-constant, slight flex
        self.front_calf_swing_base = -1.8
        self.front_calf_swing_amplitude = 0.5        # front knee sinusoidal flex depth
        self.front_calf_swing_lift = 0.2             # minimum immediate lift at swing start

        self.rear_calf_stance = (-1.3, -1.4)
        self.rear_calf_swing_base = -1.4
        self.rear_calf_swing_amplitude = 0.45        # rear knee sinusoidal flex depth
        self.rear_calf_swing_lift = 0.3              # immediate lift at swing start

        # Standing targets (used during warmup, matches home keyframe) 
        self.standing_targets = {
            'front_right_thigh': 0.9,
            'front_left_thigh':  0.9,
            'rear_right_thigh':  0.9,
            'rear_left_thigh':   0.9,
            'front_right_calf': -1.8,
            'front_left_calf':  -1.8,
            'rear_right_calf':  -1.4,
            'rear_left_calf':   -1.4,
        }

    def set_gait(self, gait_name):
        if gait_name not in self.GAITS:
            raise ValueError(
                f"Gait '{gait_name}' not recognized. "
                f"Available gaits: {list(self.GAITS.keys())}"
            )
        self.gait_name = gait_name
        hip_phases = self.GAITS[gait_name]
        knee_phases = hip_phases + self.hip_knee_phase_offset
        all_phases = np.concatenate([hip_phases, knee_phases])
        self._build_phase_offset_matrix(all_phases)
        for i in range(self.neurons_cnt):
            self.neurons[i].phase = all_phases[i]

    def set_frequency(self, frequency, index=None):
        if index is None:
            for n in self.neurons:
                n.frequency = frequency
        else:
            self.neurons[index].frequency = frequency

    def _build_phase_offset_matrix(self, desired_phases):
        n = self.neurons_cnt
        self.phase_offsets = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.phase_offsets[i, j] = desired_phases[j] - desired_phases[i]

    def rk4_integration(self, dt):
        """Advance all oscillator phases by one timestep dt using 4th-order Runge-Kutta."""
        def derivatives(thetas):
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

    def get_outputs(self):
        """Return CPG outputs y_i = A_i * max(0, cos(θ_i)). Legacy, prefer get_phase_state()."""
        return np.array([
            n.amplitude * max(0.0, np.cos(n.phase))
            for n in self.neurons
        ])

    def get_phase_state(self):
        """
        Compute stance/swing state for each oscillator.

        Returns a list of (is_stance, progress) tuples:
            is_stance: True if in stance phase, False if in swing
            progress:  [0, 1] how far through the current sub-phase
        """
        two_pi = 2 * np.pi
        stance_end = self.duty_cycle * two_pi
        states = []
        for n in self.neurons:
            phi = n.phase % two_pi
            if phi < stance_end:
                progress = phi / stance_end
                states.append((True, progress))
            else:
                swing_range = two_pi - stance_end
                progress = (phi - stance_end) / swing_range
                states.append((False, progress))
        return states

    def get_leg_phase_state(self):
        """
        Compute stance/swing state per LEG, driven only by hip oscillators.

        Returns a dict mapping leg index (0-3) to (is_stance, progress).
        Both hip and knee of the same leg share the same stance/swing state.
        """
        two_pi = 2 * np.pi
        stance_end = self.duty_cycle * two_pi
        leg_states = {}
        for leg_idx in range(4):
            phi = self.neurons[leg_idx].phase % two_pi  # hip oscillator
            if phi < stance_end:
                progress = phi / stance_end
                leg_states[leg_idx] = (True, progress)
            else:
                swing_range = two_pi - stance_end
                progress = (phi - stance_end) / swing_range
                leg_states[leg_idx] = (False, progress)
        return leg_states

    # ── MuJoCo controller interface ─────────────────────────────────

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

    def run(self, model, data):
        """
        Called every MuJoCo timestep by mj.set_mjcb_control.
        Steps the CPG then applies PD control to all joints.
        """
        self.sim_time = data.time

        # ── Warmup: hold standing pose, don't oscillate ─────────────
        if self.sim_time < self.warmup_seconds:
            targets = dict(self.static_targets)
            targets.update(self.standing_targets)
            self._apply_pd(data, targets)
            return

        # ── Blend: ramp CPG influence over 1 s after warmup ─────────
        blend_duration = 1.0  # seconds
        t_since_warmup = self.sim_time - self.warmup_seconds
        blend = min(1.0, t_since_warmup / blend_duration)

        # 1. Step the CPG forward
        self.rk4_integration(self.dt)
        leg_states = self.get_leg_phase_state()  # {0-3: (is_stance, progress)}

        # Leg index mapping: 0=FL, 1=FR, 2=RR, 3=RL

        # 2. Map hip oscillator states (0-3) → thigh joint targets
        cpg_targets = {}
        for osc_idx, joint_name in self.OSCILLATOR_TO_THIGH.items():
            is_stance, progress = leg_states[osc_idx]
            if 'front' in joint_name:
                stance_range = self.front_thigh_stance
                swing_range  = self.front_thigh_swing
            else:
                stance_range = self.rear_thigh_stance
                swing_range  = self.rear_thigh_swing

            if is_stance:
                cpg_target = stance_range[0] + progress * (stance_range[1] - stance_range[0])
            else:
                cpg_target = swing_range[0] + progress * (swing_range[1] - swing_range[0])

            stand_target = self.standing_targets[joint_name]
            cpg_targets[joint_name] = stand_target + blend * (cpg_target - stand_target)

        # 3. Map knee targets from the SAME leg's stance/swing (driven by hip)
        for osc_idx, joint_name in self.OSCILLATOR_TO_CALF.items():
            hip_idx = self.KNEE_TO_HIP[osc_idx]  # get corresponding hip
            is_stance, progress = leg_states[hip_idx]  # use HIP's state

            # Select front vs rear calf parameters
            if 'front' in joint_name:
                calf_stance = self.front_calf_stance
                swing_base = self.front_calf_swing_base
                swing_amp  = self.front_calf_swing_amplitude
                swing_lift = self.front_calf_swing_lift
            else:
                calf_stance = self.rear_calf_stance
                swing_base = self.rear_calf_swing_base
                swing_amp  = self.rear_calf_swing_amplitude
                swing_lift = self.rear_calf_swing_lift

            if is_stance:
                cpg_target = calf_stance[0] + progress * (calf_stance[1] - calf_stance[0])
            else:
                # Swing: constant lift offset + sinusoidal extra flex at mid-swing
                # swing_lift provides immediate ground clearance at progress=0
                # swing_amp adds peak flex at mid-swing on top of that
                cpg_target = swing_base - swing_lift - swing_amp * np.sin(np.pi * progress)

            stand_target = self.standing_targets[joint_name]
            cpg_targets[joint_name] = stand_target + blend * (cpg_target - stand_target)

        # 4. Build full target dict: static joints + CPG-driven joints
        targets = dict(self.static_targets)
        targets.update(cpg_targets)

        # 5. PD control
        self._apply_pd(data, targets)

    def _apply_pd(self, data, targets):
        """Apply PD control to all actuators towards the given targets."""
        for joint_name, actuator_idx in self.ACTUATOR_DICT.items():
            if joint_name not in targets:
                continue
            pos_idx = self.SENSOR_POS_DICT[joint_name]
            vel_idx = self.SENSOR_VEL_DICT[joint_name]

            current_pos = data.sensordata[pos_idx]
            current_vel = data.sensordata[vel_idx]
            target_pos = targets[joint_name]

            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel

            data.ctrl[actuator_idx] = (
                self.kp * position_error + self.kd * velocity_error
            )