import os
import sys

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import OvalOffset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

def oval_traj_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.8), trajectory_method=TrajectoryMethod.OVAL)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_stairs.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    
    oval_offsets = {
        # FRONT
        OvalOffset.X_FFORE:   0.08,
        OvalOffset.X_FHIND:   0.08,
        OvalOffset.Z_FTOP:    0.13,
        OvalOffset.Z_FBOTTOM: 0.01,   # IMPORTANT (stance_depth!)

        # REAR
        OvalOffset.X_RFORE:   0.06,
        OvalOffset.X_RHIND:   0.06,
        OvalOffset.Z_RTOP:    0.11,
        OvalOffset.Z_RBOTTOM: 0.01,
    }

    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, oval_offset=oval_offsets)
    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)
    
    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def egg_traj_test():
    starting_state = State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.8)
    robot_interface = RobotInterface(starting_state, trajectory_method=TrajectoryMethod.EGG)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_stairs.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    # sim.enable_air_mode(0.5)
    # sim.enable_graph()
    # sim.enable_joint_sliders()
    # sim.enable_joint_graph()
    step_height = {
        Foot.FL: 0.13,
        Foot.FR: 0.13,
        Foot.RL: 0.11,
        Foot.RR: 0.11
    }

    step_width = {
        Foot.FL: 0.08,
        Foot.FR: 0.08,
        Foot.RL: 0.06,
        Foot.RR: 0.06
    }
    
    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=step_width, step_height=step_height, params=params, use_adaptive_pd=True)
    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def main():
    egg_traj_test()
    return
    oval_traj_test()

if __name__ == "__main__":
    main()

