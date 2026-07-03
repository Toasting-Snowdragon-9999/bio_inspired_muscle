"""@brief Standalone demo: exercise the Levenberg-Marquardt IK solver inside the MuJoCo sim.

Entry-point script that builds a per-foot IK solver, wraps it in ``TestInvSim``, and
runs a rendered air-mode simulation that drives each foot toward a target Cartesian position.
"""
import os
import sys
import numpy as np

from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inverse_kinematics.simple_inv import LevenbegMarquardtIK
from shared_module.robot_state import Foot, Joint, RobotInterface, State, Mode, Gait

class TestInvSim:
    """Test the inverse kinematic solver in this simulation environment.

    @brief Controller that solves per-foot IK each tick and publishes the resulting joint targets.
    """
    def __init__(
        self,
        model,
        data,
        robot_interface: RobotInterface,
        solvers: dict[Foot, LevenbegMarquardtIK],
        target_pos: dict[Foot, list[float]],
    ):
        """@brief Cache model/data handles, IK solvers, targets, and joint-address lookups.

        @param model: MuJoCo MjModel for the loaded scene.
        @param data: MuJoCo MjData state container associated with the model.
        @param robot_interface: Shared RobotInterface that receives the solved target positions.
        @param solvers: Mapping from Foot to its Levenberg-Marquardt IK solver.
        @param target_pos: Mapping from Foot to its desired Cartesian goal position [x, y, z].
        """
        self.model = model
        self.data = data
        self.robot_interface = robot_interface
        self.solvers = solvers
        self.target_pos = target_pos
        self._tick = 0
        self._in_run = False
        self.foot_body_ids = {foot: self.model.body(foot.value).id for foot in self.target_pos}
        self.prev_q = {foot: self.data.qpos.copy() for foot in self.target_pos}
        self.joint_qposadr = {
            Joint.FR_HIP: int(self.model.joint("FR_hip_joint").qposadr[0]),
            Joint.FR_THIGH: int(self.model.joint("FR_thigh_joint").qposadr[0]),
            Joint.FR_CALF: int(self.model.joint("FR_calf_joint").qposadr[0]),
            Joint.FL_HIP: int(self.model.joint("FL_hip_joint").qposadr[0]),
            Joint.FL_THIGH: int(self.model.joint("FL_thigh_joint").qposadr[0]),
            Joint.FL_CALF: int(self.model.joint("FL_calf_joint").qposadr[0]),
            Joint.RR_HIP: int(self.model.joint("RR_hip_joint").qposadr[0]),
            Joint.RR_THIGH: int(self.model.joint("RR_thigh_joint").qposadr[0]),
            Joint.RR_CALF: int(self.model.joint("RR_calf_joint").qposadr[0]),
            Joint.RL_HIP: int(self.model.joint("RL_hip_joint").qposadr[0]),
            Joint.RL_THIGH: int(self.model.joint("RL_thigh_joint").qposadr[0]),
            Joint.RL_CALF: int(self.model.joint("RL_calf_joint").qposadr[0]),
        }

    def run(self) -> None:
        """@brief Solve IK for each configured foot and write joint targets to RobotInterface.

        Solve IK for each configured foot and write joint targets to RobotInterface.
        """
        if self._in_run:
            return

        self._in_run = True
        q_seed = self.data.qpos.copy()
        residuals = {}
        try:
            for foot, goal in self.target_pos.items():
                try:
                    q_seed, residual = self.solvers[foot].calculate(
                        goal=np.asarray(goal, dtype=float),
                        init_q=q_seed,
                        body_id=self.foot_body_ids[foot],
                        max_iter=12,
                    )
                    self.prev_q[foot] = q_seed.copy()
                    residuals[foot] = residual
                except Exception as exc:
                    # Keep controller alive inside mjcb_control and report which leg failed.
                    print(f"[TestInvSim] IK failed for {foot.name}: {exc}")
                    q_seed = self.prev_q[foot].copy()
                    residuals[foot] = float("inf")

            self.robot_interface.target_positions = {
                joint: float(q_seed[qpos_idx])
                for joint, qpos_idx in self.joint_qposadr.items()
            }

            if self._tick % 200 == 0:
                print("current foot positions:")
                for foot, body_id in self.foot_body_ids.items():
                    foot_pos = self.data.body(body_id).xpos
                    print(f"  {foot.name}: {foot_pos}")
                print(f"[TestInvSim] t={self.data.time:.2f}s max_residual={max(residuals.values()):.6f}")
            self._tick += 1
        finally:
            self._in_run = False


def main():
    """@brief Entry point: build per-foot IK solvers and run the air-mode IK tracking demo."""

    # target_pos = {
    #     Foot.FL: [0.1795716643722928, 0.14149099862255338, 0.18806348777447116], 
    #     Foot.FR: [0.1795716705870714, -0.141490825432425, 0.18806342628408043], 
    #     Foot.RL: [-0.26887864336080886, 0.14201085282060438, 0.19723781364348866], 
    #     Foot.RR: [-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]
    # }

    target_pos = {
        Foot.FL: [0.0, 0.0, -0.0], 
        Foot.FR: [0.0, -0.0, -0.0], 
        Foot.RL: [-0.0, -0.0, -0.0], 
        Foot.RR: [-0.0, -0.0, -0.0]
    }
    

    starting_state = State(mode=Mode.MOVING, gait=Gait.TROT, frequency=0.7)
    robot_interface = RobotInterface(starting_state)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
    model, data = sim.get_model_and_data()
    ik_solvers: dict[Foot, LevenbegMarquardtIK] = {}

    for foot in target_pos:
        ik_solvers[foot] = LevenbegMarquardtIK(
            model=model,
            data=data,
            step_size=0.1,
            tol=1e-4,
            damping=0.01,
        )

    sim.enable_air_mode()
    # sim.enable_graph()
    # sim.enable_joint_sliders()
    # sim.enable_joint_graph()
    controller = TestInvSim(
        model=model,
        data=data,
        robot_interface=robot_interface,
        solvers=ik_solvers,
        target_pos=target_pos,
    )
    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)

if __name__ == "__main__":
    main()
