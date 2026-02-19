import os 
from mujoco_sim import MujocoSim
import glfw
import numpy as np


class BB8InteractiveController:
    # Cart motion limits (meters)
    CART_RANGE = (-0.05, 0.05)

    ACTUATOR_DICT = {
        'cart_x': 0,
        'cart_y': 1,
    }

    JOINT_DICT = {
        'cart_x': 'cart_x',
        'cart_y': 'cart_y',
    }

    def __init__(self):
        self.model = None
        self.data = None

        # Target cart offsets
        self.target = {
            'cart_x': 0.0,
            'cart_y': 0.0,
        }

        self.step_size = 0.002  # meters per key press

        # PD gains (gentle!)
        self.kp = 80.0
        self.kd = 15.0

    def init_controller(self, model, data):
        self.model = model
        self.data = data

        # Cache indices (MuJoCo-style, not sensor-based)
        self.qpos_x = model.joint('cart_x').qposadr
        self.qpos_y = model.joint('cart_y').qposadr

        self.qvel_x = model.joint('cart_x').dofadr
        self.qvel_y = model.joint('cart_y').dofadr

        self.act_x = model.actuator('cart_x').id
        self.act_y = model.actuator('cart_y').id

    def keyboard_callback(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        # Forward / backward
        if key == glfw.KEY_UP:
            self.target['cart_x'] += self.step_size
        elif key == glfw.KEY_DOWN:
            self.target['cart_x'] -= self.step_size

        # Left / right
        elif key == glfw.KEY_LEFT:
            self.target['cart_y'] += self.step_size
        elif key == glfw.KEY_RIGHT:
            self.target['cart_y'] -= self.step_size

        # Reset
        elif key == glfw.KEY_R:
            self.target['cart_x'] = 0.0
            self.target['cart_y'] = 0.0

        # Clamp
        self.target['cart_x'] = np.clip(
            self.target['cart_x'],
            self.CART_RANGE[0],
            self.CART_RANGE[1]
        )

        self.target['cart_y'] = np.clip(
            self.target['cart_y'],
            self.CART_RANGE[0],
            self.CART_RANGE[1]
        )

    def run(self, model, data):
        # Read current state
        x  = data.qpos[self.qpos_x].item()
        y  = data.qpos[self.qpos_y].item()

        dx = data.qvel[self.qvel_x].item()
        dy = data.qvel[self.qvel_y].item()

        # PD control (now guaranteed scalars)
        ux = self.kp * (self.target['cart_x'] - x) - self.kd * dx
        uy = self.kp * (self.target['cart_y'] - y) - self.kd * dy

        # Apply control
        data.ctrl[self.act_x] = ux
        data.ctrl[self.act_y] = uy



def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'bb-8_scene.xml')
    sim = MujocoSim(xml_path)
    controller = BB8InteractiveController()
    
    sim.sim(controller=controller, sim_length=-1)  # Run until window closed

if __name__ == "__main__":
    main()
