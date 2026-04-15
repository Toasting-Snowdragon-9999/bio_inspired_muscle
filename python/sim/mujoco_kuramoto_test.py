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
    if gait == Gait.WALK:
        return EllipsoidConfig(
            front_x_fore   = 0.10,  # forward reach — front legs (m)
            front_x_hind   = 0.04,  # rearward reach — front legs (m)
            front_z_top    = 0.06,   # swing height — front legs (m)
            front_z_bottom = 0.03,   # stance depth — front legs (m)
            front_rotation = 0.00,   # ~11° forward tilt — front legs
            front_skew     = 0.00,   # x-displacement (m) — shift ellipse fwd/back, front legs

            rear_x_fore    = 0.04,  # forward reach — rear legs (m)
            rear_x_hind    = 0.10,  # rearward reach — rear legs (m)
            rear_z_top     = 0.06,   # swing height — rear legs (m)
            rear_z_bottom  = 0.03,   # stance depth — rear legs (m)
            rear_rotation  = -0.00,   # ~11° forward tilt — rear legs
            rear_skew      = -0.00,   # x-displacement (m) — shift ellipse fwd/back, rear legs
        )
    
    elif gait == Gait.TROT:
        return EllipsoidConfig(
            front_x_fore   = 0.14,  
            front_x_hind   = 0.08,
            front_z_top    = 0.10,
            front_z_bottom = 0.02,
            front_rotation = 0.05,
            front_skew     = 0.00,

            rear_x_fore    = 0.06,
            rear_x_hind    = 0.12,
            rear_z_top     = 0.10,
            rear_z_bottom  = 0.02,
            rear_rotation  = -0.05,
            rear_skew      = -0.00,
        )
    
    elif gait == Gait.BOUND:
        return EllipsoidConfig(
            front_x_fore   = 0.16,  # forward reach — front legs (m)
            front_x_hind   = 0.06,  # rearward reach — front legs (m)
            front_z_top    = 0.09,   # swing height — front legs (m)
            front_z_bottom = 0.02,   # stance depth — front legs (m)
            front_rotation = 0.20,   # ~11° forward tilt — front legs
            front_skew     = 0.00,   # x-displacement (m) — shift ellipse fwd/back, front legs

            rear_x_fore    = 0.04,  # forward reach — rear legs (m)
            rear_x_hind    = 0.16,  # rearward reach — rear legs (m)
            rear_z_top     = 0.09,   # swing height — rear legs (m)
            rear_z_bottom  = 0.02,   # stance depth — rear legs (m)
            rear_rotation  = -0.20,   # ~11° forward tilt — rear legs
            rear_skew      = -0.00,   # x-displacement (m) — shift ellipse fwd/back, rear legs
        )

def elip_traj_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.95), trajectory_method=TrajectoryMethod.ELLIPSOID)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    # sim.enable_air_mode(0.5)
    cfg = get_cfg(robot_interface.current_gait)
    cfg = EllipsoidConfig(
            front_x_fore   = 0.14,  
            front_x_hind   = 0.08,
            front_z_top    = 0.10,
            front_z_bottom = 0.04,
            front_rotation = 0.05,
            front_skew     = 0.05,

            rear_x_fore    = 0.06,
            rear_x_hind    = 0.14,
            rear_z_top     = 0.08,
            rear_z_bottom  = 0.02,
            rear_rotation  = -0.05,
            rear_skew      = -0.00,
        )
    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
    sim.sim(controller=controller, sim_length=5, slow_factor=2.0)
    
    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def oval_traj_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.8999999999999999999999999999999999999999999999999999), trajectory_method=TrajectoryMethod.OVAL)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    sim.enable_air_mode(0.5)
    oval_offsets = {
        # FRONT
        OvalOffset.X_FFORE:   0.15,   # front: forward reach (m)
        OvalOffset.X_FHIND:   0.15,   # front: rearward reach (m)
        OvalOffset.Z_FTOP:    0.05,   # front: swing height (m)
        OvalOffset.Z_FBOTTOM: 0.02,   # IMPORTANT (stance_depth!)

        # REAR
        OvalOffset.X_RFORE:   0.15,
        OvalOffset.X_RHIND:   0.15,
        OvalOffset.Z_RTOP:    0.05,
        OvalOffset.Z_RBOTTOM: 0.02,
    }

    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, oval_offset=oval_offsets)
    sim.sim(controller=controller, sim_length=-1, slow_factor=4.0)
    
    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

def main():
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()

