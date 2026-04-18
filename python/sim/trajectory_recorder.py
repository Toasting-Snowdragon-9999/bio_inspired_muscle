import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.robot_state import Foot, Hip
from inverse_kinematics.inverse_kin import HIP_OFFSETS

class TrajectoryRecorder:
    """
    Records and compares:
    - actual foot trajectories (from simulation)
    - desired trajectories (from controller / trajectory builder)

    Everything is stored in HIP frame (x, z).
    """

    def __init__(self, robot_interface):
        self.robot_interface = robot_interface

        self.actual = {foot: [] for foot in Foot}
        self.desired = {foot: [] for foot in Foot}

    def reset(self):
        self.actual = {foot: [] for foot in Foot}
        self.desired = {foot: [] for foot in Foot}

    # =========================
    # RECORDING
    # =========================

    def record_actual(self):
        """Record actual (simulated) foot positions."""
        for foot in Foot:
            hip = Hip[foot.name]

            foot_pos = np.array(self.robot_interface.foot_positions[foot])
            hip_pos  = np.array(self.robot_interface.hip_position[hip])

            rel = foot_pos - hip_pos

            # ✅ store FULL 3D
            self.actual[foot].append([rel[0], rel[1], rel[2]])

    def record_desired(self, foot_targets):
        for foot in Foot:
            pos = foot_targets[foot]

            # ✅ store FULL 3D
            self.desired[foot].append([
                float(pos.x),
                float(pos.y),
                float(pos.z)
            ])

    def record(self, foot_targets):
        """
        Convenience: record both at once.
        """
        self.record_actual()
        self.record_desired(foot_targets)

    # =========================
    # DATA
    # =========================

    def get_data(self):
        return {
            foot: {
                "actual": np.array(self.actual[foot]),
                "desired": np.array(self.desired[foot])
            }
            for foot in Foot
        }
    
    def get_global_trajectories(self, use_desired=False):
        """
        Convert hip-relative trajectories → body-relative trajectories.
        """

        source = self.desired if use_desired else self.actual

        global_traj = {}

        for foot in Foot:
            traj = np.array(source[foot])

            if len(traj) == 0:
                global_traj[foot] = traj
                continue

            hip_offset = HIP_OFFSETS[foot]

            # Expand offset to match trajectory shape
            global_traj[foot] = traj + hip_offset

        return global_traj
    
    def trim(self, skip=100):
        """
        Remove first N samples from all recorded data.
        """
        for foot in Foot:
            self.actual[foot] = self.actual[foot][skip:]
            self.desired[foot] = self.desired[foot][skip:]

    # =========================
    # PLOTTING
    # =========================

    def plot(self, title="Actual vs Desired Trajectories"):
        data = self.get_data()

        plt.figure(figsize=(10, 8))

        for i, foot in enumerate(Foot):
            traj_a = data[foot]["actual"]
            traj_d = data[foot]["desired"]

            plt.subplot(2, 2, i + 1)

            if len(traj_d) > 0:
                plt.plot(traj_d[:, 0], traj_d[:, 2], '--', label="desired")

            if len(traj_a) > 0:
                plt.plot(traj_a[:, 0], traj_a[:, 2], label="actual")

            plt.title(foot.name)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.grid()
            plt.legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_overlay(self, title="Overlay (All Feet)"):
        data = self.get_data()

        plt.figure(figsize=(6, 6))

        for foot in Foot:
            traj_a = data[foot]["actual"]
            traj_d = data[foot]["desired"]

            if len(traj_a) > 0:
                plt.plot(traj_a[:, 0], traj_a[:, 2], label=f"{foot.name}-actual")

            if len(traj_d) > 0:
                plt.plot(traj_d[:, 0], traj_d[:, 2], '--', label=f"{foot.name}-desired")

        plt.xlabel("X")
        plt.ylabel("Z")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_3d(self, title="Feet Trajectories (Body Frame)", use_desired=False):
        """
        Plot trajectories in body frame so feet are relative to each other.
        """

        trajs = self.get_global_trajectories(use_desired=use_desired)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for foot in Foot:
            traj = trajs[foot]

            if len(traj) == 0:
                continue

            ax.plot(
                traj[:, 0],  # x
                traj[:, 1],  # y
                traj[:, 2],  # z
                label=foot.name
            )

            # mark starting point
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=30)

        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (lateral)")
        ax.set_zlabel("Z (height)")
        ax.set_title(title)

        ax.legend()
        ax.invert_zaxis()

        # nicer viewing angle
        ax.view_init(elev=25, azim=120)

        plt.tight_layout()
        plt.show()

    # =========================
    # METRICS (🔥 very useful)
    # =========================

    def compute_error(self):
        """
        Compute mean tracking error per foot.
        """
        errors = {}

        for foot in Foot:
            a = np.array(self.actual[foot])
            d = np.array(self.desired[foot])

            if len(a) == 0 or len(d) == 0:
                errors[foot] = None
                continue

            n = min(len(a), len(d))
            err = np.linalg.norm(a[:n] - d[:n], axis=1)

            errors[foot] = {
                "mean": float(np.mean(err)),
                "max": float(np.max(err))
            }

        return errors
