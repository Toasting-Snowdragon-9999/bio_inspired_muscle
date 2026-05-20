import os
import sys
from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig, OvalOffset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod
from shared_module.settings_loader import load_settings_from_file

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

def elip_traj_test():
    try: 
        gait = Gait.TROT
        cfg, freq, duty_factor = load_settings_from_file(gait)
        robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=gait, frequency=freq), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=duty_factor)
        # robot_interface.enable_cpg = False
        # xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
        xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise.xml')

        sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
        # sim.enable_air_mode(0.5)
        # # Hard
        # params = (
        #     0.2,  # a: learning rate of impedance adaptation
        #     5.0,  # b: sensitivity of impedance adaptation to velocity error
        #     0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
        # )
        # Soft
        # params = (
        #     0.1,  # a: learning rate of impedance adaptation
        #     1.5,  # b: sensitivity of impedance adaptation to velocity error
        #     0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
        # )
        # Best 
        params = (
            0.1,  # a: learning rate of impedance adaptation
            20.0,  # b: sensitivity of impedance adaptation to velocity error
            0.07  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
        )
        controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=False, ellipsoid_config=cfg)
        
    except Exception as e:
        print(f"Error during setup: {e}")
        return

    try:
        sim.sim(controller=controller, sim_length=8, slow_factor=1.0)
    except Exception as e:
        print(f"Error during simulation: {e}")

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def main():
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()

