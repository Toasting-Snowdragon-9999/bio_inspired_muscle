import os, sys
from math import floor
import numpy as np
import mujoco as mj 
from mujoco.glfw import glfw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import LEG_LABELS

class MujocoSim:
    def __init__(self, model_path, window_scale = 1.0, print_camera_config=0):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.print_camera_config = print_camera_config
        self.window_width = floor(1920 * window_scale)
        self.window_height = floor(1080 * window_scale)
        # Mouse interaction state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        
        # Will be set during init_graphics
        self.cam = None
        self.scene = None
        
        # Controller keyboard callback
        self.controller_keyboard_callback = None

        # Oscillator figure overlays
        self.fig_hip = None
        self.fig_knee = None
        self._fig_pnt = 0          # ring-buffer write index
        self._fig_controller = None
        self.dispense_in_air = False
        self.enabled_graph = False

    def init_graphics(self):
        """Initialize GLFW window and visualization structures."""
        glfw.init()
        window = glfw.create_window(self.window_width, self.window_height, "MuJoCo Simulation", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        self.cam = mj.MjvCamera()
        opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(opt)
        
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Install GLFW callbacks
        glfw.set_key_callback(window, self.keyboard)
        glfw.set_cursor_pos_callback(window, self.mouse_move)
        glfw.set_mouse_button_callback(window, self.mouse_button)
        glfw.set_scroll_callback(window, self.scroll)

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

    def simulation_step(self, window, model, data, opt, scene, cam, context):
        """Perform one simulation step and render."""
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        if self.print_camera_config :
            print(f'cam.azimuth={cam.azimuth}; cam.elevation={cam.elevation}; cam.distance={cam.distance}')
            print(f'cam.lookat=np.array([{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}])')

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # ── Render oscillator figure overlays ───────────────────────
        if self.fig_hip is not None and self.enabled_graph:
            self._update_figures()
            fig_w = viewport_width // 3
            fig_h = viewport_height // 4
            # Hip plot: bottom-right
            hip_rect = mj.MjrRect(viewport_width - fig_w, 0, fig_w, fig_h)
            mj.mjr_figure(hip_rect, self.fig_hip, context)
            # Knee plot: above the hip plot
            knee_rect = mj.MjrRect(viewport_width - fig_w, fig_h, fig_w, fig_h)
            mj.mjr_figure(knee_rect, self.fig_knee, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    def sim(self, controller=None, sim_length=-1, slow_factor=1.0):
        """
        Main simulation loop.
        Args:
            controller: An object which has two methods init_controller(model, data) and control(model, data). The former is called once at the beginning of the simulation, and the latter is called at every simulation step.
            sim_length: Duration of the simulation in seconds. If negative, runs indefinitely until window is closed.
            slow_factor: Slow-motion factor. 1.0 = real-time, 2.0 = half speed, 5.0 = 5x slower, etc.
        """
        window, cam, opt, scene, context = self.init_graphics()

        # Initialize and set controller
        if controller is not None:
            controller.init_controller(self.model, self.data)
            # Register controller's keyboard callback if it has one
            if hasattr(controller, 'keyboard_callback'):
                self.controller_keyboard_callback = controller.keyboard_callback
            # Set up oscillator figure overlay if controller supports it
            # if hasattr(controller, 'get_oscillator_outputs'):
            #     self._fig_controller = controller
            #     self._init_figures()

        mj.set_mjcb_control(controller.run if controller is not None else None)

        # Main loop
        while not glfw.window_should_close(window):
            time_prev = self.data.time

            # Simulate at 60 Hz, scaled by slow_factor
            while (self.data.time - time_prev < 1.0 / (60.0 * slow_factor)):
                mj.mj_step(self.model, self.data)

            if sim_length > 0 and self.data.time >= sim_length:
                break

            self.simulation_step(window, self.model, self.data, opt, scene, cam, context)

            if self.dispense_in_air:
                self.enable_air_mode(0.5)

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
            self.dispense_in_air = True
        self.model.opt.gravity[:] = [0, 0, 0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[0:6] = 0.0
        self.data.qpos[2] = height
        mj.mj_forward(self.model, self.data)
    
    def enable_graph(self):
        if self.fig_hip is None:
            if hasattr(self.controller, 'get_oscillator_outputs'):
                self._fig_controller = self.controller
                self._init_figures()
            self.enabled_graph = True