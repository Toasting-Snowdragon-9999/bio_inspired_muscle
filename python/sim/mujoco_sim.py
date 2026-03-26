from pyexpat import model
import os, sys
from math import floor
import numpy as np
import mujoco as mj 
from mujoco.glfw import glfw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import (
    LEG_LABELS, KNEE_POS_RANGE, FRONT_HIP_POS_RANGE,
    ABDUCTION_POS_RANGE, BACK_HIP_POS_RANGE, SENSOR_POS_DICT,
    SENSOR_VEL_DICT, ACTUATOR_DICT, NEURON_TO_FOOT_DICT
)
from shared_module.robot_state import Joint, RobotInterface, Foot, Hip, Thigh

class MujocoSim:
    def __init__(self, model_path, robot_interface: RobotInterface, window_scale = 1.0, print_camera_config=0):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.robot_interface = robot_interface
        self.print_camera_config = print_camera_config
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

        # Will be set during init_graphics
        self.cam = None
        self.scene = None
        
        # Controller keyboard callback
        self.controller_keyboard_callback = None

        # Oscillator figure overlays
        self.fig_hip = None
        self.fig_knee = None
        self.fig_joints = None
        self.slide_joints = None
        self._joint_fig_pnt = 0
        self._joint_history = 300
        self._joint_indices = None
        self._fig_pnt = 0          # ring-buffer write index
        self._fig_controller = None
        self.dispense_in_air = False
        self.enabled_graph = False
        self.enabled_joint_graph = False
        self.enabled_joint_sliders = False

        # for computing CoT
        self._energy = 0.0
        self._start_x = None
        self._robot_mass = None
        self.use_bias_compensation = True

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

    # ── Oscillator figure overlay ───────────────────────────────────
    _FIG_HISTORY = 300          # number of data points shown (≈5 s at 60 fps)
    _LEG_COLORS = [             # FL=blue, FR=red, RR=green, RL=magenta
        (0.2, 0.4, 1.0),
        (1.0, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.8, 0.2, 0.8),
    ]

    def _init_figures(self):
        """Create two mjvFigure objects for hip / knee oscillator output."""
        labels = LEG_LABELS

        for attr, title in [('fig_hip', 'Hip oscillators'), ('fig_knee', 'Knee oscillators')]:
            fig = mj.MjvFigure()
            mj.mjv_defaultFigure(fig)
            fig.title = title
            fig.xlabel = 'Time'
            fig.flg_extend = 0       # fixed x-range, we scroll manually
            fig.range[0] = [0, self._FIG_HISTORY]  # x range
            fig.range[1] = [-1.2, 1.2]             # y range
            fig.gridsize = [5, 5]

            for i in range(4):
                fig.linergb[i] = list(self._LEG_COLORS[i])
                fig.linename[i] = labels[i]
                # initialise flat data
                for k in range(self._FIG_HISTORY):
                    fig.linedata[i][2 * k] = float(k)
                    fig.linedata[i][2 * k + 1] = 0.0
                fig.linepnt[i] = self._FIG_HISTORY

            setattr(self, attr, fig)

        self._fig_pnt = 0
    
    def _init_joint_figures(self):
        """Create figure for joint position visualization."""
        fig = mj.MjvFigure()
        mj.mjv_defaultFigure(fig)

        fig.title = "Joint Positions"
        fig.xlabel = "Time"
        fig.range[0] = [0, self._joint_history]
        fig.range[1] = [-2.5, 2.5]   # adjust for your joint limits
        fig.gridsize = [5, 5]

        # Get joint indices (exclude root free joint)
        self._joint_indices = []
        for i in range(self.model.njnt):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
            if name is not None and self.model.jnt_type[i] != mj.mjtJoint.mjJNT_FREE:
                self._joint_indices.append(i)

        # Setup lines
        for i, j_id in enumerate(self._joint_indices):
            fig.linergb[i] = [np.random.rand(), np.random.rand(), np.random.rand()]
            fig.linename[i] = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j_id)

            for k in range(self._joint_history):
                fig.linedata[i][2*k] = float(k)
                fig.linedata[i][2*k+1] = 0.0

            fig.linepnt[i] = self._joint_history

        self.fig_joints = fig
        self._joint_fig_pnt = 0
    
    def _init_joint_sliders(self):
        self.slide_joints = []
        self._joint_names = []
        self._joint_ranges = []

        # Exact joint order based on SENSOR_POS_DICT
        joint_order = list(SENSOR_POS_DICT.keys())

        for name in joint_order:
            self._joint_names.append(name)

            if "calf" in name:
                self._joint_ranges.append(KNEE_POS_RANGE)
            elif "thigh" in name:
                if "front" in name:
                    self._joint_ranges.append(FRONT_HIP_POS_RANGE)
                else:
                    self._joint_ranges.append(BACK_HIP_POS_RANGE)
            elif "hip" in name:
                self._joint_ranges.append(ABDUCTION_POS_RANGE)

        for i, name in enumerate(self._joint_names):
            fig = mj.MjvFigure()
            mj.mjv_defaultFigure(fig)

            fig.title = str(name)
            fig.xlabel = ""
            fig.range[0] = [0, 1]
            fig.range[1] = list(self._joint_ranges[i])
            fig.gridsize = [2, 5]

            fig.linergb[0] = [0.2, 0.7, 0.9]
            fig.linepnt[0] = 2

            # initialize at midpoint
            mid = np.mean(self._joint_ranges[i])

            fig.linedata[0][0] = 0.5
            fig.linedata[0][1] = self._joint_ranges[i][0]

            fig.linedata[0][2] = 0.5
            fig.linedata[0][3] = mid

            self.slide_joints.append(fig)

    def _update_figures(self):
        """Push latest oscillator outputs into the figure ring buffers."""
        if self._fig_controller is None or self.fig_hip is None:
            return
        if not hasattr(self._fig_controller, 'get_oscillator_outputs'):
            return

        hip_out, knee_out = self._fig_controller.get_oscillator_outputs()

        idx = self._fig_pnt % self._FIG_HISTORY
        for i in range(4):
            # shift all data left by one
            for k in range(self._FIG_HISTORY - 1):
                self.fig_hip.linedata[i][2 * k + 1] = self.fig_hip.linedata[i][2 * (k + 1) + 1]
                self.fig_knee.linedata[i][2 * k + 1] = self.fig_knee.linedata[i][2 * (k + 1) + 1]
            # write newest sample at the end
            self.fig_hip.linedata[i][2 * (self._FIG_HISTORY - 1) + 1] = float(hip_out[i])
            self.fig_knee.linedata[i][2 * (self._FIG_HISTORY - 1) + 1] = float(knee_out[i])

        self._fig_pnt += 1

    def _update_joint_figures(self):
        if self.fig_joints is None:
            return

        for line_i, j_id in enumerate(self._joint_indices):
            # Read joint angle directly from qpos using MuJoCo's qposadr
            addr = self.model.jnt_qposadr[j_id]
            value = self.data.qpos[addr]

            # shift left
            for k in range(self._joint_history - 1):
                self.fig_joints.linedata[line_i][2*k+1] = \
                    self.fig_joints.linedata[line_i][2*(k+1)+1]

            # write newest sample
            self.fig_joints.linedata[line_i][2*(self._joint_history-1)+1] = float(value)

        self._joint_fig_pnt += 1
    
    def _update_joint_sliders(self):
        if self.slide_joints is None:
            return

        sens = self.data.sensordata

        for i, name in enumerate(self._joint_names):
            sensor_idx = SENSOR_POS_DICT[name]
            joint_val = sens[sensor_idx]

            fig = self.slide_joints[i]

            # bottom = joint min
            fig.linedata[0][0] = 0.5
            fig.linedata[0][1] = self._joint_ranges[i][0]

            # top = current theta
            fig.linedata[0][2] = 0.5
            fig.linedata[0][3] = joint_val

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

        # ── Render oscillator figure overlays ──────────────────────
        if self.enabled_graph:
            self._update_figures()
            fig_w = viewport_width // 3
            fig_h = viewport_height // 4
            # Hip plot: bottom-right
            hip_rect = mj.MjrRect(viewport_width - fig_w, 0, fig_w, fig_h)
            mj.mjr_figure(hip_rect, self.fig_hip, context)
            # Knee plot: above the hip plot
            knee_rect = mj.MjrRect(viewport_width - fig_w, fig_h, fig_w, fig_h)
            mj.mjr_figure(knee_rect, self.fig_knee, context)
        
        if self.enabled_joint_graph:
            self._update_joint_figures()

            fig_w = viewport_width // 3
            fig_h = viewport_height // 4

            joint_rect = mj.MjrRect(0, 0, fig_w, fig_h)
            mj.mjr_figure(joint_rect, self.fig_joints, context)
        
        elif self.enabled_joint_sliders:
            self._update_joint_sliders()

            cols = 4
            slider_w = viewport_width // 10
            slider_h = viewport_height // 4

            for i, fig in enumerate(self.slide_joints):
                col = i % cols
                row = i // cols

                x = col * slider_w
                y = viewport_height - (row + 1) * slider_h

                rect = mj.MjrRect(x, y, slider_w, slider_h)
                mj.mjr_figure(rect, fig, context)


        glfw.swap_buffers(window)
        glfw.poll_events()

    # ── Robot-interface synchronisation ──────────────────────────

    def _sync_robot_interface(self):
        """Copy MuJoCo sensor data → RobotInterface so controllers see fresh state."""
        ri = self.robot_interface
        ri.dt = self.model.opt.timestep

        # QPOS_DICT = {}
        # for joint in Joint:
        #     joint_name = (
        #         joint.name
        #         .replace('HIP', 'hip_joint')
        #         .replace('THIGH', 'thigh_joint')
        #         .replace('CALF', 'calf_joint')
        #     )

        #     joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        #     QPOS_DICT[joint] = self.model.jnt_qposadr[joint_id]

        # pos = {
        #     joint: float(self.data.qpos[idx])
        #     for joint, idx in QPOS_DICT.items()
        # }

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

    def sim(self, controller=None, sim_length=-1, slow_factor=1.0):
        """
        Main simulation loop.
        Args:
            controller: An object with a `run()` method (no model/data args).
                        Communication goes through self.robot_interface.
            sim_length: Duration of the simulation in seconds. If negative,
                        runs indefinitely until window is closed.
            slow_factor: Slow-motion factor. 1.0 = real-time, 2.0 = half speed.
        """
        window, cam, opt, scene, context = self.init_graphics()

        # Set timestep on robot_interface once
        self.robot_interface.dt = self.model.opt.timestep

        # Register controller helpers (keyboard, oscillator overlay)
        if controller is not None:
            if hasattr(controller, 'keyboard_callback'):
                self.controller_keyboard_callback = controller.keyboard_callback
            if hasattr(controller, 'get_oscillator_outputs'):
                self._fig_controller = controller

        # Build mjcb_control callback: sync → run controller → apply targets
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

        self._energy = 0.0
        self._start_x = self.data.qpos[0]

        # Main loop
        while not glfw.window_should_close(window):
            time_prev = self.data.time

            # Simulate at 60 Hz, scaled by slow_factor
            while (self.data.time - time_prev < 1.0 / (60.0 * slow_factor)):
                mj.mj_step(self.model, self.data)
                self._accumulate_energy()

            if sim_length > 0 and self.data.time >= sim_length:
                break

            self.simulation_step(window, self.model, self.data, opt, scene, cam, context)

            if self.dispense_in_air:
                self.enable_air_mode(self.height)

        # Clear global MuJoCo callback before tearing down GLFW.
        # If left set, the closure referencing this sim's objects will be
        # invoked by MuJoCo during the next model load, causing a crash.
        mj.set_mjcb_control(None)
        glfw.destroy_window(window)
        glfw.terminate()


    def keyboard(self, window, key, scancode, act, mods):
        # Handle built-in keyboard commands
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        
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
    
    def enable_graph(self):
        if self.fig_hip is None:
            self._init_figures()
            self.enabled_graph = True
            # self.enabled_joint_graph = False
    
    def enable_joint_graph(self):
        if self.fig_joints is None:
            self._init_joint_figures()

        self.enabled_joint_graph = True
        self.enabled_joint_sliders = False


    def enable_joint_sliders(self):
        if self.slide_joints is None:
            self._init_joint_sliders()
 
        self.enabled_joint_sliders = True
        self.enabled_joint_graph = False

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
        if self._start_x is None:
            return None

        current_x = self.data.qpos[0]   # base x position
        distance = current_x - self._start_x

        if distance <= 0:
            # Robot did not move forward — CoT undefined (likely fell over)
            return None

        g = 9.81
        # CoT = E / (m * g * d)  — dimensionless cost of transport
        cot = self._energy / (self._robot_mass * g * distance)
        return cot

        g = 9.81
        cot = self._energy / (self._robot_mass * g * distance)

        return cot
