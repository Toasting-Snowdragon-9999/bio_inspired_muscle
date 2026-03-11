import os
import sys

from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait

def main():
    starting_state = State(mode=Mode.MOVING, gait=Gait.TROT, frequency=0.7)
    robot_interface = RobotInterface(starting_state)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    sim.use_pd_control()
    # sim.enable_air_mode(0.5)
    # sim.enable_graph()
    # sim.enable_joint_sliders()
    # sim.enable_joint_graph()
    step_height = {
        Foot.FL: 0.1,
        Foot.FR: 0.1,
        Foot.RL: 0.1,
        Foot.RR: 0.1
    }
    controller = IKController(robot_interface=robot_interface, stride_length=0.1, step_height=step_height)
    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

if __name__ == "__main__":
    main()

