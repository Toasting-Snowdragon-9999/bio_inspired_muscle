import os
import sys
import argparse

from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ellipsoid trajectory sim with configurable EllipsoidConfig parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Front legs (FL, FR) ──────────────────────────────────────────────────
    p.add_argument("--freq",   type=float, default=1.95,  help="Gait frequency (Hz)")
    p.add_argument("--front_x_fore",   type=float, default=0.14,  help="Front: forward reach (m)")
    p.add_argument("--front_x_hind",   type=float, default=0.08,  help="Front: rearward reach (m)")
    p.add_argument("--front_z_top",    type=float, default=0.10,  help="Front: swing height (m)")
    p.add_argument("--front_z_bottom", type=float, default=0.04,  help="Front: stance depth (m)")
    p.add_argument("--front_rotation", type=float, default=0.05,  help="Front: ellipse rotation (rad)")
    p.add_argument("--front_skew",     type=float, default=0.05,  help="Front: x-displacement (m)")
    # ── Rear legs (RL, RR) ────────────────────────────────────────────────────
    p.add_argument("--rear_x_fore",    type=float, default=0.06,  help="Rear:  forward reach (m)")
    p.add_argument("--rear_x_hind",    type=float, default=0.14,  help="Rear:  rearward reach (m)")
    p.add_argument("--rear_z_top",     type=float, default=0.08,  help="Rear:  swing height (m)")
    p.add_argument("--rear_z_bottom",  type=float, default=0.02,  help="Rear:  stance depth (m)")
    p.add_argument("--rear_rotation",  type=float, default=-0.05, help="Rear:  ellipse rotation (rad)")
    p.add_argument("--rear_skew",      type=float, default=0.00,  help="Rear:  x-displacement (m)")
    return p.parse_args()


def elip_traj_test():
    args = parse_args()

    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=args.freq), trajectory_method=TrajectoryMethod.ELLIPSOID)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    # sim.enable_air_mode(0.5)
    cfg = EllipsoidConfig(
        front_x_fore   = args.front_x_fore,
        front_x_hind   = args.front_x_hind,
        front_z_top    = args.front_z_top,
        front_z_bottom = args.front_z_bottom,
        front_rotation = args.front_rotation,
        front_skew     = args.front_skew,
        rear_x_fore    = args.rear_x_fore,
        rear_x_hind    = args.rear_x_hind,
        rear_z_top     = args.rear_z_top,
        rear_z_bottom  = args.rear_z_bottom,
        rear_rotation  = args.rear_rotation,
        rear_skew      = args.rear_skew,
    )
    params = (
        0.2,  # a: learning rate of impedance adaptation
        5.0,  # b: sensitivity of impedance adaptation to velocity error
        0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
    )
    controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, ellipsoid_config=cfg)
    sim.sim(controller=controller, sim_length=5, slow_factor=2.0)
    
    cot = sim.compute_CoT()
    return cot

def main():
    cot = elip_traj_test()
    print("Cost of Transport:", cot)
    
    

if __name__ == "__main__":
    main()

