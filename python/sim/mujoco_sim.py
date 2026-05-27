from pyexpat import model
import os, sys
import time
import traceback
from math import floor
from collections import deque
from scipy.spatial.transform import Rotation as R

import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import (
    LEG_LABELS, KNEE_POS_RANGE, FRONT_HIP_POS_RANGE,
    ABDUCTION_POS_RANGE, BACK_HIP_POS_RANGE, SENSOR_POS_DICT,
    SENSOR_VEL_DICT, ACTUATOR_DICT, NEURON_TO_FOOT_DICT
)
from shared_module.robot_state import Joint, RobotInterface, Foot, Hip, Thigh, RobotData

class MujocoSim:
    def __init__(self, model_path, robot_interface: RobotInterface, window_scale = 1.0, print_camera_config=0, render_hz: float = 60.0):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.robot_interface = robot_interface
        self.print_camera_config = print_camera_config
        self.render_hz = render_hz   # target render frequency; lower on slow PCs (e.g. 30.0)
        self.window_width = floor(1920 * window_scale)
        self.window_height = floor(1080 * window_scale)
        # Mouse interaction state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.height = 0.0
        self.use_direct = False
        self.follow_robot = True
        self.z_thresh = 0.025 

        # Will be set during init_graphics
        self.cam = None
        self.scene = None
        self.controller = None

        # Controller keyboard callback
        self.controller_keyboard_callback = None

        self.dispense_in_air = False

        # for computing CoT
        self._energy = 0.0
        self._start_x = None
        self._start_y = None
        self._robot_mass = None
        self.use_bias_compensation = True
        self.print_cot = False
        self._starting_pos = None

        # geom-id → Foot map for actual-contact detection. Built lazily on the
        # first call to `_sync_robot_interface` because the model is parsed
        # before MuJoCo finishes assigning ids in some test paths. See
        # `_ensure_foot_geom_map` below — used to populate
        # `robot_interface.contact` (the actual-contact mirror of
        # `expected_footfall`) every mj_step.
        self._foot_geom_to_foot: dict[int, Foot] = {}
        self.foot_fall_history = 3.0 # seconds
        self.foot_fall_comparison = None
        self.prev_expected_footfall = None

    @property
    def starting_pos(self):
        return self._starting_pos

    @starting_pos.setter
    def starting_pos(self, starting_state):
        self.data.qpos[0:3] = starting_state
        self._starting_pos = starting_state

    # @starting_pos.setter
    # def starting_pos(self, starting_state):

    #     print("\n========== SETTING SPAWN ==========")

    #     print("Requested starting_state:", starting_state)

    #     print("BEFORE:")
    #     print("qpos[0:7]:", self.data.qpos[0:7])
    #     print("qvel[0:6]:", self.data.qvel[0:6])

    #     # Apply position
    #     self.data.qpos[0:3] = starting_state

    #     # Optional but VERY important for debugging
    #     # Reset velocities to avoid explosive impulses
    #     self.data.qvel[:] = 0.0

    #     print("\nAFTER POSITION WRITE:")
    #     print("qpos[0:7]:", self.data.qpos[0:7])
    #     print("qvel[0:6]:", self.data.qvel[0:6])

    #     # Check terrain/body overlap after forward dynamics
    #     import mujoco as mj
    #     mj.mj_forward(self.model, self.data)

    #     print("\nAFTER mj_forward:")
    #     print("base xpos:", self.data.body("base_link").xpos)
    #     print("base quat:", self.data.qpos[3:7])
    #     print("ncon:", self.data.ncon)

    #     # Print active contacts
    #     for i in range(self.data.ncon):
    #         con = self.data.contact[i]
    #         print(
    #             f"contact {i}: "
    #             f"geom1={con.geom1}, "
    #             f"geom2={con.geom2}, "
    #             f"dist={con.dist}"
    #         )

    #     self._starting_pos = starting_state

    #     print("===================================\n")

    def get_model_and_data(self):
        return self.model, self.data

    def use_direct_control(self):
        self.use_direct = True

    def set_camera_follow(self, enabled: bool):
        """Enable or disable camera following the robot."""
        self.follow_robot = bool(enabled)

    def init_graphics(self):
        """Initialize GLFW window and visualization structures."""
        glfw.init()
        window = glfw.create_window(self.window_width, self.window_height, "MuJoCo Simulation", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        self.cam = mj.MjvCamera()

        opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        self.cam.azimuth=90
        self.cam.elevation=0.5 
        self.cam.distance=4
        mj.mjv_defaultOption(opt)
        
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Install GLFW callbacks
        glfw.set_key_callback(window, self.keyboard)
        glfw.set_cursor_pos_callback(window, self.mouse_move)
        glfw.set_mouse_button_callback(window, self.mouse_button)
        glfw.set_scroll_callback(window, self.scroll)
        glfw.focus_window(window)

        return window, self.cam, opt, self.scene, context

    def simulation_step(self, window, model, data, opt, scene, cam, context):
        """Perform one simulation step and render."""
        if self.follow_robot:
            base_pos = data.body("base_link").xpos
            cam.lookat[:] = base_pos

        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        if self.print_camera_config :
            print(f'cam.azimuth={cam.azimuth}; cam.elevation={cam.elevation}; cam.distance={cam.distance}')
            print(f'cam.lookat=np.array([{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}])')

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # ── Robot-interface synchronisation ──────────────────────────

    def _ensure_foot_geom_map(self) -> None:
        """Lazily build geom-id → Foot map. Called once on first sync.

        Walks every geom in the model, looks up its parent body, and if that
        body's name matches one of the `Foot` enum values (e.g. 'FL_foot'),
        records the geom id. Robust to models with multiple geoms per foot
        body — every such geom counts as a foot-contact source.
        """
        if self._foot_geom_to_foot:
            return
        body_to_foot = {foot.value: foot for foot in Foot}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            if body_name in body_to_foot:
                self._foot_geom_to_foot[geom_id] = body_to_foot[body_name]

    def _sync_robot_interface(self):
        """Copy MuJoCo sensor data → RobotInterface so controllers see fresh state."""
        ri = self.robot_interface
        ri.dt = self.model.opt.timestep

        pos = {joint: float(self.data.sensordata[idx])
               for joint, idx in SENSOR_POS_DICT.items()}

        vel = {joint: float(self.data.sensordata[idx])
               for joint, idx in SENSOR_VEL_DICT.items()}

        ri.joint_positions = pos
        ri.joint_velocities = vel
        ri.body_position = self.data.body("base_link").xpos.tolist()
        ri.body_orientation = self.data.body("base_link").xmat.tolist()  # rotation matrix as flat list
        ri.thigh_position = {
            thigh: self.data.body(thigh.value).xpos.tolist()
            for thigh in Thigh
        }
        ri.foot_positions = {
            foot: self.data.body(foot.value).xpos.tolist()
            for foot in Foot
        }
        ri.hip_position = {
            hip: self.data.body(hip.value).xpos.tolist()
            for hip in Hip
        }

        hip_stance = {
            hip: [pos[i] - ri.body_position[i] for i in range(3)]
            for hip, pos in ri.hip_position.items()
        }
        foot_stance = {
            foot: [pos[i] - ri.body_position[i] for i in range(3)]
            for foot, pos in ri.foot_positions.items()
        }
        ri.stance_positions = {
            foot: [
                foot_pos[i] - hip_stance[Hip[foot.name]][i] for i in range(3)
            ]
            for foot, foot_pos in foot_stance.items()
        }

        # Actual per-foot contact for the RL gait-error reward. We start every
        # foot at 0 and flip to 1 if any active contact pair touches that
        # foot's geom — `data.ncon` is the live contact count, `data.contact`
        # the array of contact records. Cheap O(ncon).
        # self._ensure_foot_geom_map()
        contact_state = {foot: 0 for foot in Foot}
        for foot in Foot:
            z = self.robot_interface.foot_positions[foot][2]
            if z < self.z_thresh:
                contact_state[foot] = 1
        # for c_idx in range(self.data.ncon):
        #     con = self.data.contact[c_idx]
        #     foot = self._foot_geom_to_foot.get(int(con.geom1))
        #     if foot is None:
        #         foot = self._foot_geom_to_foot.get(int(con.geom2))
        #     if foot is not None:
        #         contact_state[foot] = 1
        #         print(f"Contact detected for {foot.name} at time {self.data.time:.2f}s: geom1={con.geom1}, geom2={con.geom2}, dist={con.dist}")
        ri.contact = contact_state

    def _apply_controller_targets_directly(self):
        """Write RobotInterface.target_positions → data.ctrl (position actuators)."""
        targets = self.robot_interface.target_positions
        # current_joint_pos = self.robot_interface.joint_positions
        for joint, target_angle in targets.items():
            joint_name = (
                joint.name
                .replace('HIP', 'hip_joint')
                .replace('THIGH', 'thigh_joint')
                .replace('CALF', 'calf_joint')
            )
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            # start_angle = current_joint_pos[joint]
            # angle = (1 - alpha) * start_angle + alpha * target_angle
            self.data.qpos[qpos_adr] = target_angle

    def _apply_controller_targets_torque(self):
        """
        Apply torque commands from RobotInterface to MuJoCo actuators.
        """

        torques = self.robot_interface.target_torques

        for joint, tau_cmd in torques.items():

            joint_name = (
                joint.name
                .replace('HIP', 'hip_joint')
                .replace('THIGH', 'thigh_joint')
                .replace('CALF', 'calf_joint')
            )

            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

            dof_adr = self.model.jnt_dofadr[joint_id]

            actuator_idx = ACTUATOR_DICT[joint]

            # MuJoCo bias forces (gravity + coriolis)
            tau_bias = float(self.data.qfrc_bias[dof_adr]) if self.use_bias_compensation else 0.0

            # Apply controller torque + gravity compensation
            tau = tau_cmd + tau_bias

            # Respect actuator limits
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_idx]
            tau = np.clip(tau, ctrl_min, ctrl_max)

            # Send torque command
            self.data.ctrl[actuator_idx] = tau

    def set_bias_compensation(self, enabled: bool = True):
        self.use_bias_compensation = bool(enabled)

    def reset(self):
        """Reset any internal state in the MujocoSim instance (e.g. for a new sim run)."""
        self._energy = 0.0
        if self.controller is not None and hasattr(self.controller, 'reset'):
            self.controller.reset()
            self.robot_interface.cpg_alpha = 0.00
        self.starting_pos = self._starting_pos
        self.next_cot_print_time = self.sim_length if self.sim_length > 0 else None

    def reset_cot(self):    
        """Reset CoT tracking state (energy accumulator and start position)."""
        self._energy = 0.0
        self._start_x = self.data.qpos[0]
        self._start_y = self.data.qpos[1]
    
    def _save_data(self):
        """Save relevant data from the current timestep into the RobotInterface's data buffer."""
        if self.robot_interface.data_list is None:
            self.robot_interface.data_list = deque(maxlen=int(10 / self.robot_interface.dt))

        quat = self.data.body("base_link").xquat.copy()

        r = R.from_quat([
            quat[1],
            quat[2],
            quat[3],
            quat[0]
        ])

        roll, pitch, _ = r.as_euler('xyz', degrees=True)

        data_entry = RobotData(
            knee_torque={
                foot: float(
                    self.data.ctrl[
                        ACTUATOR_DICT[Joint[f"{foot.name}_CALF"]]
                    ]
                )
                for foot in Foot
            },

            cot=self.compute_CoT(),

            cpg_phases=self.robot_interface.cpg_phase.copy(),

            robot_velocity=float(self.data.qvel[0]),

            roll=float(roll),
            pitch=float(pitch),

            time=float(self.data.time)
        )
        self.robot_interface.data_list.append(data_entry)

    def sim(self, controller=None, sim_length=-1, slow_factor=1.0, warmup: float = 2.0):
        """
        Main simulation loop (frame-rate independent, deterministic physics).

        Args:
            controller: Object with a `run()` method.
            sim_length: Duration in seconds (negative = infinite).
            slow_factor: >1.0 = slow motion (visual only).
            warmup: Sim-seconds before energy/distance tracking begins.
                    Lets the robot settle from the initial drop so CoT
                    is not polluted by the transient.  Default 2.0 s.
        """
        window, cam, opt, scene, context = self.init_graphics()

        # --- Simulation parameters ---
        dt = self.model.opt.timestep                # Fixed physics timestep
        render_dt = 1.0 / self.render_hz            # Target render interval
        accumulator = 0.0

        # Set timestep for external interface
        self.robot_interface.dt = dt
        self.foot_fall_comparison = deque(maxlen=int(self.foot_fall_history / dt))  # 3 seconds of history for footfall comparison with robot_interface.expected_footfall

        # Register controller helpers
        if controller is not None:
            if hasattr(controller, 'keyboard_callback'):
                self.controller_keyboard_callback = controller.keyboard_callback
            if hasattr(controller, 'get_oscillator_outputs'):
                self._fig_controller = controller
            self.controller = controller

        # --- Control callback ---
        def _control_callback(model, data):
            try:
                self._sync_robot_interface()
                controller.run()
                if self.use_direct:
                    self._apply_controller_targets_directly()
                else:
                    self._apply_controller_targets_torque()
                
                if self.prev_expected_footfall is not None:
                    self.foot_fall_comparison.append({
                        "contact": self.robot_interface.contact.copy(),
                        "expected": self.prev_expected_footfall.copy()
                    })
                    # print(f"Footfall comparison at time {self.data.time:.2f}s: contact={self.robot_interface.contact}, expected={self.prev_expected_footfall}")
                    # print(f"footfall comparison {self.foot_fall_comparison[-1]}")  # Print the most recent comparison for debugging
                if self.robot_interface.expected_footfall is not None:
                    self.prev_expected_footfall = (
                        self.robot_interface.expected_footfall.copy()
                    )
                self._save_data()

            except Exception as e:
                print(f"Error in control callback at time {self.data.time:.2f}s: {e}")
                traceback.print_exc()

        mj.set_mjcb_control(_control_callback if controller is not None else None)

        # --- Init metrics ---
        if self._robot_mass is None:
            self._robot_mass = np.sum(self.model.body_mass)

        # Energy/distance tracking is deferred until after warmup
        self._energy = 0.0
        self._start_x = None
        self._start_y = None
        metrics_started = False
        self.next_cot_print_time = warmup + sim_length if sim_length > 0 else None
        self.sim_length = sim_length
        iteration = 1
        # --- Main loop ---
        while not glfw.window_should_close(window):
            frame_start = time.perf_counter()

            # Amount of simulation time we want to advance this frame
            # slow_factor > 1 → less sim-time per frame → slow motion
            sim_time_budget = render_dt / slow_factor
            accumulator += sim_time_budget

            # --- Physics stepping (fixed dt) ---
            while accumulator >= dt:
                mj.mj_step(self.model, self.data)
                # Start accumulating energy only after warmup period
                if not metrics_started and self.data.time >= warmup:
                    metrics_started = True
                    self._energy = 0.0
                    self._start_x = self.data.qpos[0]
                    self._start_y = self.data.qpos[1]
                if metrics_started:
                    self._accumulate_energy()
                accumulator -= dt

            # --- Exit condition ---
            if self.sim_length > 0 and self.data.time >= self.next_cot_print_time:
                break
                if iteration % 10 == 0:
                    # every 10 iteration increment freq
                    self.robot_interface.frequency += 0.1
                    print(f"Incrementing frequency to {self.robot_interface.frequency:.2f} Hz")
                if self.print_cot is False:
                    print(iteration, " CoT: ", self.compute_CoT())
                    self.reset_cot()
                    self.next_cot_print_time += self.sim_length
                    iteration += 1

                self.print_cot = True
            else: 
                self.print_cot = False

            # --- Rendering ---
            self.simulation_step(window, self.model, self.data, opt, scene, cam, context)

            if self.dispense_in_air:
                self.enable_air_mode(self.height)

            # --- Frame rate control (rendering only) ---
            elapsed = time.perf_counter() - frame_start
            sleep_time = render_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            # print foot z coordinate

        # --- Cleanup ---
        mj.set_mjcb_control(None)
        glfw.destroy_window(window)
        glfw.terminate()

    def compute_current_footfall(self):
        """Return the current expected footfall pattern as a set of Foot enums."""
        footfall = set()
        for foot, idx in SENSOR_POS_DICT.items():
            if foot.name.endswith("foot") and self.data.sensordata[idx] < 0.01:
                footfall.add(Foot[foot.name.upper()])
        return footfall

    def headless_sim(self, controller=None, sim_length: float = 10.0, warmup: float = 2.0) -> float | None:
        """
        Run the simulation as fast as possible with no visual output.
        Intended for parameter sweeps and CoT benchmarking.

        Args:
            controller:  Object with a `run()` method (same as sim()).
            sim_length:  Duration of the *measurement* phase in sim-seconds. Must be > 0.
            warmup:      Sim-seconds to run before energy/distance tracking begins.
                         Allows the robot to settle from the initial drop before CoT
                         is measured. Default 2.0 s. Set to 0 to disable.

        Returns:
            Cost of Transport (dimensionless), or None if the robot did not
            move forward during the measurement phase (e.g. fell over).
        """
        if sim_length <= 0:
            raise ValueError("sim_length must be positive for headless_sim.")

        # Reset simulation to t=0 so each call starts from the same initial
        # state regardless of previous runs on this instance.
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        # Reset controller internal state (CPG phases, IK warm-start, etc.)
        # so stale phase accumulation from a previous run doesn't carry over.
        if controller is not None and hasattr(controller, 'reset'):
            controller.reset()

        # Set timestep on robot_interface
        self.robot_interface.dt = self.model.opt.timestep

        # Build control callback: sync sensors → controller → apply torques
        def _control_callback(model, data):
            self._sync_robot_interface()
            controller.run()
            if self.use_direct:
                self._apply_controller_targets_directly()
            else:
                self._apply_controller_targets_torque()
            

        mj.set_mjcb_control(_control_callback if controller is not None else None)

        if self._robot_mass is None:
            self._robot_mass = np.sum(self.model.body_mass)

        # ── Warmup phase: controller runs but energy is NOT counted ──
        # This lets the robot settle from the initial drop before measurement.
        warmup_end = self.data.time + warmup
        while self.data.time < warmup_end:
            mj.mj_step(self.model, self.data)

        # ── Measurement phase: reset metrics, then accumulate ────────
        # _start_x is set here so distance is measured from the post-warmup
        # position, not the drop point.
        self._energy = 0.0
        self._start_x = self.data.qpos[0]
        self._start_y = self.data.qpos[1]
        measure_end = self.data.time + sim_length
        while self.data.time < measure_end:
            mj.mj_step(self.model, self.data)
            self._accumulate_energy()

        mj.set_mjcb_control(None)

        return self.compute_CoT()

    # Minimum foot-z (m) we want to see during swing. Below this, the
    # `swing_clearance` reward term penalises the optimiser — discourages
    # foot-dragging policies that exploit `velocity` without lifting.
    _SWING_MIN_CLEARANCE = 0.02

    def headless_sim_extended(self, controller=None, sim_length: float = 10.0,
                              warmup: float = 2.0) -> dict:
        """
        Run headless simulation and return extended metrics for RL training.

        Same warmup → measurement loop as headless_sim(), but also tracks
        forward velocity, average body tilt (roll + pitch), survival, and the
        per-step gait-quality terms consumed by sim/gait_env.py:_compute_reward
        (gait_error, slip, swing_clearance, ang_vel). All per-step sums are
        mean-reduced over `step_count` so the scalar magnitudes stay
        comparable across `--sim-length` settings.

        Returns:
            dict with keys:
                cot              — Cost of Transport (float or None if robot didn't move)
                distance         — forward distance travelled during measurement (m)
                velocity         — average forward velocity during measurement (m/s)
                avg_tilt         — mean(|roll| + |pitch|) during measurement (rad)
                survived         — True if body stayed above 0.15 m throughout
                gait_error       — mean over steps of Σ_foot (contact - expected_footfall)²
                slip             — mean over steps of Σ_foot |foot_vel_x| while in stance
                swing_clearance  — mean over steps of Σ_foot max(0, MIN_CLEARANCE - foot_z) while in swing
                ang_vel          — mean over steps of |base angular velocity| (rad/s)
        """
        if sim_length <= 0:
            raise ValueError("sim_length must be positive for headless_sim_extended.")

        # Reset simulation
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        if controller is not None and hasattr(controller, 'reset'):
            controller.reset()

        self.robot_interface.dt = self.model.opt.timestep

        # Control callback
        # Captures any exception raised inside the controller chain
        # (`_sync_robot_interface`, `controller.run`, torque application).
        # Without this, an exception escaping into MuJoCo's C-side callback
        # wrapper triggers `mju_error("Python exception raised")`, which
        # under the default user-error handler terminates the whole
        # subprocess with exit code 1 — masking the real Python traceback.
        # We stash the exception in a closure variable, return cleanly so
        # `mj_step` can finish, and re-raise from the outer loop where
        # `_run_simulation`'s try/except can catch it and log a traceback.
        _callback_exc: list[BaseException] = []

        def _control_callback(model, data):
            if _callback_exc:
                # Already failing this episode — short-circuit so we don't
                # spam tracebacks for every remaining step before the outer
                # loop notices and bails out.
                return
            try:
                self._sync_robot_interface()
                controller.run()
                if self.use_direct:
                    self._apply_controller_targets_directly()
                else:
                    self._apply_controller_targets_torque()
            except BaseException as exc:
                # Print immediately so the failure is visible even if the
                # outer loop swallows the re-raise (e.g. caller catches and
                # converts to a penalty dict). `print_exc` writes to the
                # *current* sys.stderr — which under gait_env's verbose=False
                # path is a StringIO, but verbose=True forwards to the real
                # stderr so the LogConsole sees it.
                traceback.print_exc()
                _callback_exc.append(exc)

        mj.set_mjcb_control(_control_callback if controller is not None else None)

        if self._robot_mass is None:
            self._robot_mass = np.sum(self.model.body_mass)

        # ── Warmup phase ──
        warmup_end = self.data.time + warmup
        while self.data.time < warmup_end:
            mj.mj_step(self.model, self.data)
            if _callback_exc:
                mj.set_mjcb_control(None)
                raise _callback_exc[0]

        # ── Measurement phase ──
        self._energy = 0.0
        self._start_x = self.data.qpos[0]
        self._start_y = self.data.qpos[1]
        tilt_sum = 0.0
        step_count = 0
        survived = True
        min_height = 0.15  # body z threshold for "fell over"

        # Per-step gait-quality accumulators. Initialised here (not at
        # construction) so a single MujocoSim can serve many episodes.
        gait_error_sum = 0.0
        slip_sum = 0.0
        clearance_sum = 0.0
        ang_vel_sum = 0.0
        # Previous foot x-positions for finite-difference foot velocity.
        # Seeded after the first step inside the loop below.
        prev_foot_x: dict[Foot, float] | None = None
        dt_sim = self.model.opt.timestep

        measure_end = self.data.time + sim_length
        while self.data.time < measure_end:
            mj.mj_step(self.model, self.data)
            if _callback_exc:
                # Controller raised — surface the original exception so the
                # caller (gait_env._run_simulation) can convert it to a
                # penalty dict and the search keeps going.
                mj.set_mjcb_control(None)
                raise _callback_exc[0]
            self._accumulate_energy()

            # Body tilt from rotation matrix
            xmat = self.data.body("base_link").xmat.reshape(3, 3)
            # Roll  = atan2(R[2,1], R[2,2])
            roll = np.arctan2(xmat[2, 1], xmat[2, 2])
            # Pitch = -asin(clamp(R[2,0], -1, 1))
            pitch = -np.arcsin(np.clip(xmat[2, 0], -1.0, 1.0))
            tilt_sum += abs(roll) + abs(pitch)
            step_count += 1

            # Survival check
            body_z = self.data.body("base_link").xpos[2]
            if body_z < min_height:
                survived = False

            # ── Per-step gait-quality terms ──
            # `_sync_robot_interface` already wrote `contact`, `expected_footfall`
            # and `foot_positions` into the RobotInterface for this tick.
            ri = self.robot_interface
            actual = ri.contact
            desired = ri.expected_footfall
            foot_pos = ri.foot_positions

            cur_foot_x = {foot: foot_pos[foot][0] for foot in Foot if foot in foot_pos}

            for foot in Foot:
                a = actual.get(foot, 0)
                d = desired.get(foot, 0)
                gait_error_sum += (a - d) ** 2

                # Foot vx via backward finite-diff. Skipped on the first tick
                # (prev_foot_x is None) — slip stays at 0 for that step.
                if prev_foot_x is not None and foot in prev_foot_x and foot in cur_foot_x:
                    foot_vx = (cur_foot_x[foot] - prev_foot_x[foot]) / dt_sim
                    if a == 1:  # only penalise sliding while in stance
                        slip_sum += abs(foot_vx)

                # Swing clearance: penalty grows as foot dips below the floor
                # of `_SWING_MIN_CLEARANCE` while in swing. World-z is used.
                if a == 0 and foot in foot_pos:
                    foot_z = foot_pos[foot][2]
                    if foot_z < self._SWING_MIN_CLEARANCE:
                        clearance_sum += (self._SWING_MIN_CLEARANCE - foot_z)

            prev_foot_x = cur_foot_x

            # Base angular velocity magnitude — qvel[3:6] are the free-joint
            # rotational dofs in world frame for the floating base.
            ang_vel_sum += float(np.linalg.norm(self.data.qvel[3:6]))

        mj.set_mjcb_control(None)

        # Compute metrics
        cot = self.compute_CoT()
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]
        x_dis = current_x - self._start_x if self._start_x is not None else 0.0
        y_dis = current_y - self._start_y if self._start_y is not None else 0.0
        distance = np.hypot(x_dis, y_dis)
        velocity = distance / sim_length if sim_length > 0 else 0.0
        avg_tilt = tilt_sum / step_count if step_count > 0 else 0.0

        # Mean-reduce per-step accumulators so reward magnitudes don't scale
        # with episode length.
        denom = float(step_count) if step_count > 0 else 1.0
        gait_error = gait_error_sum / denom
        slip = slip_sum / denom
        swing_clearance = clearance_sum / denom
        ang_vel = ang_vel_sum / denom

        return {
            "cot": cot,
            "distance": distance,
            "velocity": velocity,
            "avg_tilt": avg_tilt,
            "survived": survived,
            "gait_error": gait_error,
            "slip": slip,
            "swing_clearance": swing_clearance,
            "ang_vel": ang_vel,
        }

    def keyboard(self, window, key, scancode, act, mods):
        # Handle built-in keyboard commands
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.reset()

        # Pass keyboard event to controller if registered
        if self.controller_keyboard_callback is not None:
            self.controller_keyboard_callback(window, key, scancode, act, mods)

    def mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height,
                        dy/height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                        yoffset, self.scene, self.cam)

    def remove_gravity(self):
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]

    def enable_air_mode(self, height=1.0):
        if not self.dispense_in_air:
            self.height = height
            self.dispense_in_air = True
        self.model.opt.gravity[:] = [0, 0, 0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[0:6] = 0.0
        self.data.qpos[2] = height
        mj.mj_forward(self.model, self.data)

    def _accumulate_energy(self):
        """
        Accumulate positive mechanical work from actuators.
        """
        dt = self.model.opt.timestep

        tau = self.data.actuator_force.copy()

        # Actuator torques correspond to DOFs in qvel
        qdot = self.data.qvel[:len(tau)]

        mechanical_power = tau * qdot

        # Only count positive work (realistic motor assumption)
        positive_power = np.maximum(mechanical_power, 0.0)

        self._energy += np.sum(positive_power) * dt

    def compute_CoT(self):
        """
        Compute Cost of Transport (dimensionless).
        """
        if self._start_x is None or self._start_y is None:
            # print("Error: _start_x or _start_y is None, cannot compute CoT.")
            return None

        current_x = self.data.qpos[0]   # base x position
        current_y = self.data.qpos[1]   # base y position

        x_dis = current_x - self._start_x
        y_dis = current_y - self._start_y
        distance = np.hypot(x_dis, y_dis) # Does sqrt(x_dis² + y_dis²) to get straight-line distance in the horizontal plane

        if distance <= 0:
            # Robot did not move forward — CoT undefined (likely fell over)
            print("Warning: Robot did not move forward during measurement phase. CoT is undefined.")
            return None

        g = 9.81
        # CoT = E / (m * g * d)  — dimensionless cost of transport
        cot = self._energy / (self._robot_mass * g * distance)
        return cot
