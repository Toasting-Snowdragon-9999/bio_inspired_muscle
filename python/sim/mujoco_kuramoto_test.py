"""@brief mujoco_kuramoto_test.py — mujoco kuramoto test."""
import os
import sys
from enum import Enum
import traceback
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mujoco_sim import MujocoSim
from cpg.trajectory_builder import EllipsoidConfig, OvalOffset
import json
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod
from shared_module.settings_loader import load_settings_from_file

# TROT: freq = 2.2 Hz 
# BOUND: freq = 5.0 Hz

base_size = 20
plt.rcParams.update({
    'font.size': base_size,        # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size + 2,  # X tick size
    'ytick.labelsize': base_size + 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})

os.environ["G_MESSAGES_DEBUG"] = "none"

class Terrain(Enum):
    """@brief Terrain — terrain."""
    flat = "Flat"
    tiny_hills = "Tiny Hills"
    rough = "Rough"
    very_rough = "Very Rough"
    large_hills = "Large Hills"
    tricky = "Tricky"

def set_spawn(sim, terrain):
    """
    @brief Set spawn.
    @param sim:
    @param terrain:
    @return
    """
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
    """
    @brief Get sim xml.
    @param terrain:
    @return
    """
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

def compute_contact_statistics(footfall_comparison):
    """
    @brief Compute contact statistics.
    @param footfall_comparison:
    @return
    """
    stats = {}

    for foot in Foot:
        actual = np.array([
            sample["contact"][foot]
            for sample in footfall_comparison
        ])

        expected = np.array([
            sample["expected"][foot]
            for sample in footfall_comparison
        ])

        correct = np.sum(actual == expected)
        total = len(actual)

        accuracy = correct / total

        mae = np.mean(np.abs(actual - expected))

        false_positive = np.sum((actual == 1) & (expected == 0))
        false_negative = np.sum((actual == 0) & (expected == 1))

        stats[foot] = {
            "accuracy": accuracy,
            "mae": mae,
            "false_positive": false_positive,
            "false_negative": false_negative
        }

        print(f"\n{foot.name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"False positives: {false_positive}")
        print(f"False negatives: {false_negative}")

    return stats

def compute_duty_cycles(footfall_comparison, dt):
    """
    @brief Compute duty cycles.
    @param footfall_comparison:
    @param dt:
    @return
    """
    if len(footfall_comparison) == 0:
        return {}
    
    contact_history = defaultdict(list)
    for sample in footfall_comparison:
        contact = sample["contact"]
        for foot in Foot:
            contact_history[foot].append(
                int(contact.get(foot, 0))
            )
    duty_cycles = {}

    for foot in Foot:
        contacts = np.array(contact_history[foot])
        stance_samples = np.sum(contacts == 0)
        total_samples = len(contacts)
        if total_samples == 0:
            duty_cycles[foot] = 0.0
            continue
        print(f"\n{foot.name}")
        print(f"Total samples: {total_samples}")
        stance_time = stance_samples * dt
        total_time = total_samples * dt
        duty_cycle = stance_time / total_time
        duty_cycles[foot] = duty_cycle
        print(f"Duty cycle for {foot.name}: {duty_cycle:.2f}")
    return duty_cycles

def plot_footfall(footfall_comparison, dt):
    """
    @brief Plot footfall.
    @param footfall_comparison:
    @param dt:
    """
    if footfall_comparison is not None:
        time = np.arange(len(footfall_comparison)) * dt

        fig, axs = plt.subplots(4, 1, figsize=(12, 4), sharex=True)

        for idx, foot in enumerate(Foot):

            actual = [
                sample["contact"][foot]
                for sample in footfall_comparison
            ]

            expected = [
                sample["expected"][foot]
                for sample in footfall_comparison
            ]

            axs[idx].step(
                time,
                actual,
                label=f"{foot.name} Actual",
                linewidth=2,
                where='post'
            )

            axs[idx].step(
                time,
                expected,
                '--',
                label=f"{foot.name} Expected",
                linewidth=2,
                where='post'
            )

            axs[idx].set_ylim(-0.1, 1.1)
            axs[idx].set_yticks([0, 1])
            axs[idx].set_ylabel(foot.name)
            axs[idx].grid(True)
            axs[idx].legend(loc="upper right")

        axs[-1].set_xlabel("Time [s]")

        plt.suptitle("Expected vs Actual Foot Contact")
        plt.tight_layout()
        plt.show()

def save_robot_data(robot_data_list, filename="robot_data.json"):
    """
    Save a list of RobotData entries to a JSON file.

    Parameters
    ----------
    robot_data_list : list[RobotData]
        List of RobotData dataclass instances.

    filename : str
        Output filename.
    """

    serializable_data = []

    for entry in robot_data_list:

        entry_dict = asdict(entry)

        # Convert Foot enum keys into strings
        entry_dict["knee_torque"] = {
            foot.name: value
            for foot, value in entry.knee_torque.items()
        }

        entry_dict["cpg_phases"] = {
            foot.name: value
            for foot, value in entry.cpg_phases.items()
        }

        serializable_data.append(entry_dict)

    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Saved {len(serializable_data)} entries to '{filename}'")

def graph_roll_pitch(robot_data_list, tmin=None, tmax=None):
    """
    Graph the roll and pitch of the robot over time.

    Parameters
    ----------
    robot_data_list : list[RobotData]
        List of RobotData dataclass instances.

    tmin : float, optional
        Start of the time window [s]. Entries before this are dropped.
        ``None`` means start from the first entry.

    tmax : float, optional
        End of the time window [s]. Entries after this are dropped.
        ``None`` means run to the last entry.
    """

    windowed = [
        entry for entry in robot_data_list
        if (tmin is None or entry.time >= tmin)
        and (tmax is None or entry.time <= tmax)
    ]

    if not windowed:
        print(f"No robot data in window tmin={tmin}, tmax={tmax}")
        return

    time = [entry.time for entry in windowed]
    roll = [entry.roll for entry in windowed]
    pitch = [entry.pitch for entry in windowed]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(time, roll, label="Roll", color="tab:blue", linewidth=2)
    axs[0].set_ylabel("Roll [deg]")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    axs[1].plot(time, pitch, label="Pitch", color="tab:orange", linewidth=2)
    axs[1].set_ylabel("Pitch [deg]")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    axs[-1].set_xlabel("Time [s]")

    plt.suptitle("Robot Roll and Pitch")
    plt.tight_layout()
    plt.show()

def elip_traj_test():
    """@brief Elip traj test."""
    try: 
        gait = Gait.TROT

        terrain = Terrain.flat
        cfg, freq, duty_factor, params = load_settings_from_file(gait)
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

        controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=False, ellipsoid_config=cfg)
        
    except Exception as e:
        print(f"Error during setup: {e}")
        traceback.print_exc()
        return

    try:
        sim.sim(controller=controller, sim_length=13, slow_factor=1.0)
    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()

    cot = sim.compute_CoT()
    print("Cost of Transport:", cot)
    oiac_filename = "data/oiac_data.json"
    pd_filename = "data/pd_data.json"
    save_robot_data(sim.robot_interface.data_list, filename=pd_filename)
    graph_roll_pitch(sim.robot_interface.data_list, tmin=6.0, tmax=12.0)

    # footfall_comparison = sim.foot_fall_comparison
    # if footfall_comparison is not None:
    #     compute_duty_cycles(footfall_comparison, sim.robot_interface.dt)
    #     plot_footfall(footfall_comparison, sim.robot_interface.dt)
    #     compute_contact_statistics(footfall_comparison)

def main():
    """@brief Main."""
    elip_traj_test()
    return
    

if __name__ == "__main__":
    main()