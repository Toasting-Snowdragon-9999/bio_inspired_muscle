import json
import matplotlib.pyplot as plt
import numpy as np


def load_robot_data(filename):
    """
    Load robot data JSON file.
    """

    with open(filename, "r") as f:
        return json.load(f)


def moving_average(data, window=10):
    """
    Simple smoothing for cleaner plots.
    """

    if window <= 1:
        return np.array(data)

    return np.convolve(
        data,
        np.ones(window) / window,
        mode='same'
    )

def compute_metrics(name,
                    roll,
                    pitch,
                    velocity,
                    cot,
                    torque):

    roll = np.array(roll)
    pitch = np.array(pitch)
    velocity = np.array(velocity)
    cot = np.array(cot)
    torque = np.array(torque)

    metrics = {

        "Mean Velocity": np.mean(velocity),
        "Velocity Std": np.std(velocity),

        "Mean CoT": np.mean(cot),

        "Roll RMS": np.sqrt(np.mean(roll ** 2)),
        "Pitch RMS": np.sqrt(np.mean(pitch ** 2)),

        "Max Roll": np.max(np.abs(roll)),
        "Max Pitch": np.max(np.abs(pitch)),

        "Mean Torque": np.mean(np.abs(torque)),
        "Torque RMS": np.sqrt(np.mean(torque ** 2)),
    }

    print(f"\n========== {name} ==========")

    for key, value in metrics.items():
        print(f"{key:<20}: {value:.4f}")

    return metrics

def plot_pd_vs_oiac(pd_file, oiac_file, smooth_window=10):

    pd_data = load_robot_data(pd_file)
    oiac_data = load_robot_data(oiac_file)

    foot = "RR"  # Front Right foot for torque and phase comparison
    # -----------------------------
    # Extract data
    # -----------------------------

    pd_time = [d["time"] for d in pd_data]
    oiac_time = [d["time"] for d in oiac_data]

    # Stability
    pd_roll = [d["roll"] for d in pd_data]
    oiac_roll = [d["roll"] for d in oiac_data]

    pd_pitch = [d["pitch"] for d in pd_data]
    oiac_pitch = [d["pitch"] for d in oiac_data]

    # Velocity
    pd_velocity = [d["robot_velocity"] for d in pd_data]
    oiac_velocity = [d["robot_velocity"] for d in oiac_data]

    # CoT
    pd_cot = [
        float(d["cot"]) if d["cot"] is not None else 0.0
        for d in pd_data
    ]

    oiac_cot = [
        float(d["cot"]) if d["cot"] is not None else 0.0
        for d in oiac_data
    ]

    # Knee torque
    pd_fl_torque = [
        d["knee_torque"][foot]
        for d in pd_data
    ]

    oiac_fl_torque = [
        d["knee_torque"][foot]
        for d in oiac_data
    ]

    # CPG phases
    pd_fl_phase = [
        d["cpg_phases"][foot]
        for d in pd_data
    ]

    oiac_fl_phase = [
        d["cpg_phases"][foot]
        for d in oiac_data
    ]

    # -----------------------------
    # Optional smoothing
    # -----------------------------

    pd_roll = moving_average(pd_roll, smooth_window)
    oiac_roll = moving_average(oiac_roll, smooth_window)

    pd_pitch = moving_average(pd_pitch, smooth_window)
    oiac_pitch = moving_average(oiac_pitch, smooth_window)

    pd_velocity = moving_average(pd_velocity, smooth_window)
    oiac_velocity = moving_average(oiac_velocity, smooth_window)

    pd_cot = moving_average(pd_cot, smooth_window)
    oiac_cot = moving_average(oiac_cot, smooth_window)

    pd_fl_torque = moving_average(pd_fl_torque, smooth_window)
    oiac_fl_torque = moving_average(oiac_fl_torque, smooth_window)

    # -----------------------------
    # Quantitative evaluation
    # -----------------------------

    compute_metrics(
        "PD",
        pd_roll,
        pd_pitch,
        pd_velocity,
        pd_cot,
        pd_fl_torque
    )

    compute_metrics(
        "OIAC",
        oiac_roll,
        oiac_pitch,
        oiac_velocity,
        oiac_cot,
        oiac_fl_torque
    )

    # -----------------------------
    # Plot configuration
    # -----------------------------

    roll_plot = {
        "title": "Roll Stability",
        "ylabel": "Roll [deg]",
        "pd": pd_roll,
        "oiac": oiac_roll,
    }

    pitch_plot = {
        "title": "Pitch Stability",
        "ylabel": "Pitch [deg]",
        "pd": pd_pitch,
        "oiac": oiac_pitch,
    }

    velocity_plot = {
        "title": "Forward Velocity",
        "ylabel": "Velocity [m/s]",
        "pd": pd_velocity,
        "oiac": oiac_velocity,
    }

    cot_plot = {
        "title": "Cost of Transport",
        "ylabel": "CoT",
        "pd": pd_cot,
        "oiac": oiac_cot,
    }

    torque_plot = {
        "title": f"{foot} Knee Torque",
        "ylabel": "Torque",
        "pd": pd_fl_torque,
        "oiac": oiac_fl_torque,
    }

    phase_plot = {
        "title": f"{foot} CPG Phase",
        "ylabel": "Phase [rad]",
        "pd": pd_fl_phase,
        "oiac": oiac_fl_phase,
    }

    plot_configs = [
        phase_plot,
        torque_plot,
        cot_plot,
        velocity_plot,
        roll_plot,
        pitch_plot,
    ]
    # -----------------------------
    # Create plots
    # -----------------------------

    fig, axs = plt.subplots(
        len(plot_configs),
        1,
        figsize=(14, 3 * len(plot_configs)),
        sharex=True
    )

    # Handle case with only one subplot
    if len(plot_configs) == 1:
        axs = [axs]

    for ax, cfg in zip(axs, plot_configs):

        ax.plot(pd_time, cfg["pd"], label="PD")
        ax.plot(oiac_time, cfg["oiac"], label="OIAC")

        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"])

        ax.grid(True)
        ax.legend()

        ax.set_xlim(3, 10)

    axs[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    oiac_filename = "data/oiac_data.json"
    pd_filename = "data/pd_data.json"
    plot_pd_vs_oiac(
        pd_file=pd_filename,
        oiac_file=oiac_filename,
        smooth_window=15
    )