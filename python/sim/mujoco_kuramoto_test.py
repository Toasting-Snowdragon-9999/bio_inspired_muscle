import os
import sys
from enum import Enum
from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig, OvalOffset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod
from shared_module.settings_loader import load_settings_from_file

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

class Terrain(Enum):
    flat = "Flat"
    tiny_hills = "Tiny Hills"
    rough = "Rough"
    very_rough = "Very Rough"
    large_hills = "Large Hills"
    tricky = "Tricky"

def set_spawn(sim, terrain):
    starting_state = None
    if terrain == Terrain.flat:
        starting_state = [0.0, 0.0, 0.455]

    elif terrain == Terrain.tiny_hills:
        # starting_state = [4.5, 2.0, 0.8] # starting walk
        # starting_state = [-5.5, -3.5, 0.9] # Longest walk
        # starting_state = [-5.5, -7.5, 1.1] # second Longest walk
        # starting_state = [-5.5, 4.5, 0.8] # third longest walk
        starting_state = [-4.5, -4.5, 0.8] # third longest walk

    elif terrain == Terrain.rough:
        starting_state = [-9.5, 0.0, 0.6] # Longest walk
        # starting_state = [-9.5, -8.0, 0.6] # short walk
        # starting_state = [-9.5, 4.5, 0.6] # second Longest walk
        # starting_state = [-8.5, -4.5, 0.6] # short walk
        # starting_state = [-2.5, -4.5, 0.6] # short walk
        # starting_state = [0.0, 0.0, 0.6] # medium walk
        # starting_state = [2.0, -4.5, 0.6] # From y = -2.5 -> -0.7

    elif terrain == Terrain.very_rough:
        starting_state = [-2.5, 1.4, 0.6] # From y = -2.5 -> -0.7

    elif terrain == Terrain.large_hills:
        starting_state = [0.0, 0.0, 0.8]

    elif terrain == Terrain.tricky:
        starting_state = [-1.5, 0.0, 0.5]

    else:
        starting_state = [0.0, 0.0, 0.5]

    if starting_state is None:
        raise ValueError(f"Invalid terrain: {terrain}")

    sim.starting_pos = starting_state
    return starting_state[2]

def get_sim_xml(terrain):
    if terrain == Terrain.flat:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    elif terrain == Terrain.tiny_hills:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise_tiny_mountain.xml')
    elif terrain == Terrain.rough:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise_rough.xml')
    elif terrain == Terrain.very_rough:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise_very_rough.xml')
    elif terrain == Terrain.tricky:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise_tricky.xml')
    elif terrain == Terrain.large_hills:
        return os.path.join(os.path.dirname(__file__), 'go2', 'scene_perlin_noise_large_mountain.xml')

def elip_traj_test():
    try: 
        gait = Gait.TROT
        terrain = Terrain.very_rough
        cfg, freq, duty_factor = load_settings_from_file(gait)
        robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=gait, frequency=freq), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=duty_factor)
        # robot_interface.enable_cpg = False
        
        xml_path = get_sim_xml(terrain)
        sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
        z_pos = set_spawn(sim, terrain)
        # sim.enable_air_mode(z_pos)
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
        controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
        
    except Exception as e:
        print(f"Error during setup: {e}")
        return

    try:
        sim.sim(controller=controller, sim_length=4, slow_factor=1.0)
    except Exception as e:
        print(f"Error during simulation: {e}")

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def main():
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()

