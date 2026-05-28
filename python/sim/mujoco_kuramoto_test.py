import os
import sys

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig, OvalOffset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod
from trajectory_recorder import TrajectoryRecorder

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

def elip_traj_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.95), trajectory_method=TrajectoryMethod.ELLIPSOID, duty_factor=0.5)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    # sim.enable_air_mode(0.5)
    # cfg = EllipsoidConfig( # WALK - 0.24 duty, 1.4 freq
    #     front_x_fore   = 0.14,  # forward reach — front legs (m)
    #     front_x_hind   = 0.08,  # rearward reach — front legs (m)
    #     front_z_top    = 0.10,   # swing height — front legs (m)
    #     front_z_bottom = 0.02,   # stance depth — front legs (m)
    #     front_rotation = 0.05,   # ~11° forward tilt — front legs
    #     front_skew     = 0.00,   # x-displacement (m) — shift ellipse fwd/back, front legs

    #     rear_x_fore    = 0.06,  # forward reach — rear legs (m)
    #     rear_x_hind    = 0.14,  # rearward reach — rear legs (m)
    #     rear_z_top     = 0.10,   # swing height — rear legs (m)
    #     rear_z_bottom  = 0.02,   # stance depth — rear legs (m)
    #     rear_rotation  = -0.05,   # ~11° forward tilt — rear legs
    #     rear_skew      = -0.00,   # x-displacement (m) — shift ellipse fwd/back, rear legs
    # )

    cfg = EllipsoidConfig( # TROT - 0.5 duty, 1.95 freq
        front_x_fore   = 0.122961,  # forward reach — front legs (m)
        front_x_hind   = 0.095,  # rearward reach — front legs (m)
        front_z_top    = 0.103564,   # swing height — front legs (m)
        front_z_bottom = 0.025,   # stance depth — front legs (m)
        front_rotation = 0.045610,   # ~11° forward tilt — front legs
        front_skew     = 0.035,   # x-displacement (m) — shift ellipse fwd/back, front legs

        rear_x_fore    = 0.075,  # forward reach — rear legs (m)
        rear_x_hind    = 0.121565,  # rearward reach — rear legs (m)
        rear_z_top     = 0.095,   # swing height — rear legs (m)
        rear_z_bottom  = 0.005,   # stance depth — rear legs (m)
        rear_rotation  = -0.060026,   # ~11° forward tilt — rear legs
        rear_skew      = -0.015,   # x-displacement (m) — shift ellipse fwd/back, rear legs
    )

    # cfg = EllipsoidConfig( # GALLOP 4 freq, 0.6 duty
    #     front_x_fore   = 0.20,   # more forward reach (lead leg effect)
    #     front_x_hind   = 0.15,   # moderate backward
    #     front_z_top    = 0.12,   # slightly higher than trot (more flight)
    #     front_z_bottom = 0.02,   # keep low stance
    #     front_rotation = 0.15,   # more forward tilt than trot
    #     front_skew     = 0.02,  # forward bias (important for asymmetry)

    #     rear_x_fore    = 0.05,   # less forward than front
    #     rear_x_hind    = 0.20,   # strong push-off
    #     rear_z_top     = 0.10,   # similar to front but slightly lower
    #     rear_z_bottom  = 0.03,  # slightly shallower stance
    #     rear_rotation  = -0.15, # stronger backward tilt (propulsion)
    #     rear_skew      = 0.03,  # slight backward bias
    # )

    # cfg = EllipsoidConfig( # GALLOP 4 freq, 0.6 duty
    #     front_x_fore   = 0.20,   # more forward reach (lead leg effect)
    #     front_x_hind   = 0.15,   # moderate backward
    #     front_z_top    = 0.12,   # slightly higher than trot (more flight)
    #     front_z_bottom = 0.02,   # keep low stance
    #     front_rotation = 0.15,   # more forward tilt than trot
    #     front_skew     = 0.02,  # forward bias (important for asymmetry)

    #     rear_x_fore    = 0.05,   # less forward than front
    #     rear_x_hind    = 0.20,   # strong push-off
    #     rear_z_top     = 0.10,   # similar to front but slightly lower
    #     rear_z_bottom  = 0.03,  # slightly shallower stance
    #     rear_rotation  = -0.14, # stronger backward tilt (propulsion)
    #     rear_skew      = 0.03,  # slight backward bias
    # )

    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )

    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
    recorder = TrajectoryRecorder(robot_interface)
    controller.enable_turn = True

    # Wrap controller.run
    original_run = controller.run

    def wrapped_run():
        original_run()
        recorder.record(controller.last_foot_targets)

    controller.run = wrapped_run

    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)
    
    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

    recorder.trim(1000) # remove swirly stuff in the beginning

    # recorder.plot()
    # recorder.plot_overlay()
    # recorder.plot_3d()
    print(recorder.compute_error())

def oval_traj_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.8), trajectory_method=TrajectoryMethod.OVAL)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_stairs.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    sim.enable_air_mode(0.5)
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
    sim.sim(controller=controller, sim_length=-1, slow_factor=4.0)
    
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
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()

