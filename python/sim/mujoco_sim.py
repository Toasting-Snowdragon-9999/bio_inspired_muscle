import mujoco as mj 
from mujoco.glfw import glfw


class MujocoSim:
    def __init__(self, model_path, window_width=1200, window_height=900, print_camera_config=0):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.print_camera_config = print_camera_config
        self.window_width = window_width
        self.window_height = window_height
        
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

    def simulation_step(self, window, model, data, opt, scene, cam, context):
        """Perform one simulation step and render."""
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        if self.print_camera_config :
            print(f'cam.azimuth={cam.azimuth}; cam.elevation={cam.elevation}; cam.distance={cam.distance}')
            print(f'cam.lookat=np.array([{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}])')

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    def sim(self, controller=None, sim_length=-1):
        """
        Main simulation loop.
        Args:
            controller: An object which has two methods init_controller(model, data) and control(model, data). The former is called once at the beginning of the simulation, and the latter is called at every simulation step.
            sim_length: Duration of the simulation in seconds. If negative, runs indefinitely until window is closed.
        """
        window, cam, opt, scene, context = self.init_graphics()

        # Initialize and set controller
        if controller is not None:
            controller.init_controller(self.model, self.data)
            # Register controller's keyboard callback if it has one
            if hasattr(controller, 'keyboard_callback'):
                self.controller_keyboard_callback = controller.keyboard_callback
    
        mj.set_mjcb_control(controller.run if controller is not None else None)

        # Main loop
        while not glfw.window_should_close(window):
            time_prev = self.data.time

            # Simulate at 60 Hz
            while (self.data.time - time_prev < 1.0 / 60.0):
                mj.mj_step(self.model, self.data)

            if sim_length > 0 and self.data.time >= sim_length:
                break

            self.simulation_step(window, self.model, self.data, opt, scene, cam, context)

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