import os
import sys

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig, OvalOffset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

def get_cfg(gait: Gait) -> EllipsoidConfig:
    if gait == Gait.WALK: # IDEAL
        return EllipsoidConfig(
            front_x_fore   = 0.14,
            front_x_hind   = 0.08,
            front_z_top    = 0.10,
            front_z_bottom = 0.02,
            front_rotation = 0.05,
            front_skew     = 0.00,

            rear_x_fore    = 0.06,
            rear_x_hind    = 0.14,
            rear_z_top     = 0.10,
            rear_z_bottom  = 0.02,
            rear_rotation  = -0.05,
            rear_skew      = -0.00,
        )

    elif gait == Gait.TROT: # IDEAL
        return EllipsoidConfig(
            front_x_fore   = 0.122961,
            front_x_hind   = 0.095000,
            front_z_top    = 0.103564,
            front_z_bottom = 0.025000,
            front_rotation = 0.045610,
            front_skew     = 0.035000,

            rear_x_fore    = 0.075000,
            rear_x_hind    = 0.121565,
            rear_z_top     = 0.095000,
            rear_z_bottom  = 0.005000,
            rear_rotation  = -0.060026,
            rear_skew      = -0.015000,
        )
    elif gait == Gait.CANTER:
        return EllipsoidConfig(
            front_x_fore   = 0.14,
            front_x_hind   = 0.07,
            front_z_top    = 0.12,
            front_z_bottom = 0.02,
            front_rotation = 0.10,
            front_skew     = 0.02,

            rear_x_fore    = 0.06,
            rear_x_hind    = 0.14,
            rear_z_top     = 0.10,
            rear_z_bottom  = 0.03,
            rear_rotation  = -0.05,
            rear_skew      = -0.03,
        )
    elif gait == Gait.AMBLE:
        return EllipsoidConfig(
            front_x_fore   = 0.14,
            front_x_hind   = 0.08,
            front_z_top    = 0.12,
            front_z_bottom = 0.02,
            front_rotation = 0.05,
            front_skew     = -0.03,

            rear_x_fore    = 0.06,
            rear_x_hind    = 0.14,
            rear_z_top     = 0.10,
            rear_z_bottom  = 0.02,
            rear_rotation  = -0.05,
            rear_skew      = -0.00,
        )

def elip_traj_test():
    freq = 1.675
    duty_factor = 0.4
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.AMBLE, frequency=freq), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=duty_factor)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    # sim.enable_air_mode(0.5)
    cfg = get_cfg(robot_interface.current_gait)

    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
    sim.sim(controller=controller, sim_length=-1, slow_factor=2.0)
    
    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def main():
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()

