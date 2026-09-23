"""Standalone CPG / trajectory visualizations driven by saved gait settings.

This is a slimmed-down rework of ``test.py``. Every function here is
self-contained and parameterized by a :class:`Gait`. Parameters
(frequency, duty factor, ellipsoid shape) are pulled from
``data/.settings/.<GAIT>.ini`` via :func:`load_settings_from_file`, so the
plots always reflect the tuned values on disk instead of hardcoded constants.

Each function also accepts optional ``freq`` / ``duty`` / ``cfg`` overrides:
pass a value to use it, leave it ``None`` to take the value from the settings
file. See :func:`_load_params`.

Run directly to render all three views for a default gait::

    python test_parts.py
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from kuramoto_cpg import KuramotoCpg
from trajectory_builder import TrajectoryBuilder, GaitScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.global_constants import (
    NEURON_TO_FOOT_DICT, FRONT_HIP_POS_RANGE, BACK_HIP_POS_RANGE, KNEE_POS_RANGE
)
from shared_module.robot_state import RobotInterface, State, Gait, Mode, Foot, TrajectoryMethod
from shared_module.settings_loader import load_settings_from_file
from inverse_kinematics.inverse_kin import forward_kinematics

base_size = 18
plt.rcParams.update({
    'font.size': base_size,            # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size - 2,  # X tick size
    'ytick.labelsize': base_size - 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})


def _load_params(gait, freq=None, duty=None, cfg=None):
    """Resolve (cfg, freq, duty) for a gait from settings, honoring overrides.

    Reads ``data/.settings/.<GAIT>.ini`` via :func:`load_settings_from_file`,
    then replaces any field for which the caller passed a non-``None`` value.
    """
    file_cfg, file_freq, file_duty, _ = load_settings_from_file(gait)

    cfg = cfg if cfg is not None else file_cfg
    freq = freq if freq is not None else file_freq
    duty = duty if duty is not None else file_duty

    return cfg, freq, duty


def _polygon_area(xs, zs):
    """Enclosed area of a closed polygon via the shoelace formula.

    ``xs``/``zs`` are the ordered vertices of the loop (the loop is treated as
    closed, i.e. the last point connects back to the first). Returns the
    unsigned area, so traversal direction does not matter.
    """
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    return 0.5 * abs(np.sum(xs * np.roll(zs, -1) - np.roll(xs, -1) * zs))


def _reachable_workspace_xz(hip_range, n=120):
    """Boundary of the sagittal-plane (X–Z) reachable foot workspace at q1=0.

    The leg is a 2-link planar mechanism in the X–Z plane: the foot position is
    ``forward_kinematics([0, q2, q3])`` projected to (x, z), depending only on the
    thigh angle ``q2`` (``hip_range``) and knee angle ``q3`` (``KNEE_POS_RANGE``).
    Because the knee range never crosses full extension, that map does not fold,
    so the boundary of the reachable region is exactly the image of the joint
    rectangle's perimeter. We walk the four edges of ``[hip_range] × [knee_range]``
    and map each sample through FK to get a closed boundary curve.

    Note: this is the q1 = 0 (abduction-neutral) sagittal slice. Sweeping the
    abduction joint would thicken the region slightly in z; for visualising how
    much margin the foot trajectory has, the slice is the relevant view.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Closed-loop x and z coordinates of the workspace boundary (m).
    """
    q2_lo, q2_hi = hip_range
    q3_lo, q3_hi = KNEE_POS_RANGE

    q2_sweep = np.linspace(q2_lo, q2_hi, n)
    q3_sweep = np.linspace(q3_lo, q3_hi, n)

    # Perimeter of the (q2, q3) rectangle, traversed as a closed loop.
    edges = (
        [(q2, q3_lo) for q2 in q2_sweep] +
        [(q2_hi, q3) for q3 in q3_sweep] +
        [(q2, q3_hi) for q2 in q2_sweep[::-1]] +
        [(q2_lo, q3) for q3 in q3_sweep[::-1]]
    )

    xs, zs = [], []
    for q2, q3 in edges:
        # d_y is irrelevant at q1 = 0: its contribution to z is d_y·sin(0) = 0.
        pos = forward_kinematics(np.array([0.0, q2, q3]), 0.0)
        xs.append(pos[0])
        zs.append(pos[2])

    return np.array(xs), np.array(zs)


def compute_trajectory_area(gait, freq=None, duty=None, cfg=None, dt=0.001, verbose=True):
    """Compute the enclosed area (m²) of each foot's trajectory loop.

    Each foot trajectory is a closed loop in the X–Z plane over one gait cycle.
    This builds the same CPG + trajectory stack as :func:`show_trajectory`,
    records exactly one gait cycle per foot (after a short warmup so the
    oscillators have settled), and returns the shoelace area per foot.

    Parameters and overrides behave like the other functions in this module
    (values come from the gait's settings file unless passed explicitly).

    Returns
    -------
    dict[Foot, float]
        Enclosed trajectory area in m², keyed by foot.
    """
    cfg, freq, duty = _load_params(gait, freq, duty, cfg)

    robot_interface = RobotInterface(
        starting_state=State(gait=gait, mode=Mode.MOVING, frequency=freq),
        trajectory_method=TrajectoryMethod.ELLIPSOID,
    )
    robot_interface.active_gait = gait
    robot_interface.active_traj_params = cfg
    robot_interface.duty_factor = duty
    robot_interface.dt = dt

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface, ellipsoid_config=cfg, duty_factor=duty)

    steps_per_cycle = int(round((1.0 / freq) / dt))

    # Warm up a few cycles so the oscillators settle onto their limit cycle,
    # then record exactly one cycle to get a cleanly closed loop.
    for _ in range(3 * steps_per_cycle):
        cpg.run()

    loops = {foot: {"x": [], "z": []} for foot in Foot}
    for _ in range(steps_per_cycle):
        cpg.run()
        phase = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        traj, _ = builder.build_ellipsoid_trajectory(phase, vel)
        for foot in Foot:
            loops[foot]["x"].append(traj[foot].x)
            loops[foot]["z"].append(traj[foot].z)

    areas = {
        foot: _polygon_area(loops[foot]["x"], loops[foot]["z"])
        for foot in Foot
    }

    if verbose:
        print(f"\nTrajectory area — {gait.name} "
              f"[freq={freq:.2f} Hz, duty={duty:.2f}]")
        print("-" * 40)
        for foot in Foot:
            print(f"{foot.name:>3}   {areas[foot]:.6f} m^2")
        print("-" * 40)

    return areas


def show_trajectory(gait, freq=None, duty=None, cfg=None, seconds=5.0, dt=0.001,
                    show_workspace=True, focus_on_trajectory=True):
    """Show the ellipsoid foot trajectory for a gait, sourced from settings.

    Builds the CPG + trajectory builder from the gait's ``.ini`` (frequency,
    duty factor, ellipsoid shape) and plots the last two gait cycles as front /
    rear XZ panels, with the stance position marked on each leg.

    Parameters
    ----------
    show_workspace : bool
        Shade the reachable kinematic workspace (the q1=0 sagittal slice, see
        :func:`_reachable_workspace_xz`) behind each trajectory so you can see
        how much joint-limit margin the foot loop has.
    focus_on_trajectory : bool
        When ``True`` (and the workspace is shown), the axes stay zoomed on the
        trajectory; the workspace fill extends past the view. Pass ``False`` to
        zoom out and see the entire reachable region (the trajectory looks tiny).
    """
    cfg, freq, duty = _load_params(gait, freq, duty, cfg)

    robot_interface = RobotInterface(
        starting_state=State(gait=gait, mode=Mode.MOVING, frequency=freq),
        trajectory_method=TrajectoryMethod.ELLIPSOID,
    )
    robot_interface.active_gait = gait
    robot_interface.active_traj_params = cfg
    robot_interface.duty_factor = duty
    robot_interface.dt = dt

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface, ellipsoid_config=cfg, duty_factor=duty)

    steps = int(seconds / dt)
    foot_trajectories = []
    for _ in range(steps):
        cpg.run()
        phase = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        traj, _ = builder.build_ellipsoid_trajectory(phase, vel)
        foot_trajectories.append(traj)

    # Show only the last 2 seconds for a clean shape.
    cycles_to_show = int(2.0 / dt)
    display_traj = foot_trajectories[-cycles_to_show:]

    # Area is computed over exactly one gait cycle so the shoelace sees a single
    # closed loop (display_traj spans several cycles, which would multiply it).
    steps_per_cycle = int(round((1.0 / freq) / dt))
    area_traj = foot_trajectories[-steps_per_cycle:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f'Ellipsoid Foot Trajectories — {gait.name}  '
        f'[freq={freq:.2f} Hz, duty={duty:.2f}]'
    )

    groups = [
        (axes[0], [Foot.FL, Foot.FR], 'Front Legs', FRONT_HIP_POS_RANGE),
        (axes[1], [Foot.RL, Foot.RR], 'Rear Legs', BACK_HIP_POS_RANGE),
    ]
    linestyles = ['-', '--']
    for ax, feet, title, hip_range in groups:
        if show_workspace:
            wx, wz = _reachable_workspace_xz(hip_range)
            ax.fill(wx, wz, color='gray', alpha=0.12, zorder=0,
                    label='reachable (q1=0)')
            ax.plot(wx, wz, color='gray', linewidth=1.0, alpha=0.4, zorder=0)
        for foot, ls in zip(feet, linestyles):
            x = [t[foot].x for t in display_traj]
            z = [t[foot].z for t in display_traj]
            area = _polygon_area([t[foot].x for t in area_traj],
                                 [t[foot].z for t in area_traj])
            ax.plot(x, z, linestyle=ls, label=f'{foot.name} (A={area:.4f} m²)')
            # Blue dot at the stationary (stance) position.
            sx = builder.stance_positions[foot].x
            sz = builder.stance_positions[foot].z
            ax.plot(sx, sz, 'o', color='blue', markersize=7, zorder=5,
                    label='stance' if foot == feet[0] else None)
        ax.set_title(title)
        ax.set_xlabel('X — forward / back (m)')
        ax.set_ylabel('Z — height (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    # Keep the view zoomed on the trajectory; otherwise the (much larger)
    # reachable workspace dominates and the foot loops shrink to specks.
    if show_workspace and focus_on_trajectory:
        all_x = [t[foot].x for t in display_traj for foot in Foot]
        all_z = [t[foot].z for t in display_traj for foot in Foot]
        cx = 0.5 * (min(all_x) + max(all_x))
        cz = 0.5 * (min(all_z) + max(all_z))
        half = 0.5 * max(max(all_x) - min(all_x), max(all_z) - min(all_z))
        half = max(half, 1e-3) * 2.2  # padding to reveal the nearby reach limit
        for ax in axes:  # sharey: both axes track the same window
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cz - half, cz + half)

    plt.tight_layout()
    plt.show()


def show_cpg_output(gait, freq=None, duty=None, dt=0.001, seconds=2.0):
    """Plot the duty-warped CPG neuron outputs (cos θ) over time for a gait.

    Frequency and duty factor are read from the gait's settings file unless
    overridden. Uses ``builder.apply_duty_factor`` so the warp matches the one
    used by the trajectory builder.
    """
    _, freq, duty = _load_params(gait, freq, duty)

    robot_interface = RobotInterface(
        starting_state=State(gait=gait, mode=Mode.MOVING, frequency=freq)
    )
    robot_interface.duty_factor = duty
    robot_interface.dt = dt

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface, duty_factor=duty)

    steps = int(seconds / dt)
    output = {}
    for step in range(steps):
        cpg.run()
        theta = cpg.get_phase_outputs()
        warped = np.array([builder.apply_duty_factor(p) for p in theta])
        output[step] = np.cos(warped)

    time = np.arange(steps) * dt
    plt.figure(figsize=(10, 6))
    for i in range(cpg.neurons_cnt):
        foot = NEURON_TO_FOOT_DICT[i]
        # Neurons 3/4 dashed so they stay visible when overlapping 1/2.
        linestyle = '--' if i >= 2 else '-'
        plt.plot(time, [output[step][i] for step in range(steps)],
                 label=f'{foot.name}', linestyle=linestyle)

    plt.title(f'Kuramoto CPG Outputs — {gait.name} '
              f'[freq={freq:.2f} Hz, duty={duty:.2f}]')
    plt.xlabel('Time (seconds)')
    plt.ylabel(r'$\cos(\theta)$')
    plt.legend(loc='center right')
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_single_gait_footfall(gait, freq=None, duty=None, dt=0.002, n_cycles=4):
    """Render the footfall (stance/swing) raster for a gait from settings.

    Runs the CPG and the gait scheduler, logging per-foot contact, then draws a
    black/white raster (1 = stance, 0 = swing) with vertical lines marking gait
    cycles. Frequency and duty factor come from the gait's settings file unless
    overridden.
    """
    _, freq, duty = _load_params(gait, freq, duty)

    feet_order = [Foot.FL, Foot.FR, Foot.RR, Foot.RL]

    robot_interface = RobotInterface(
        starting_state=State(gait=gait, mode=Mode.MOVING, frequency=freq)
    )
    robot_interface.dt = dt
    robot_interface.duty_factor = duty

    cpg = KuramotoCpg(robot_interface)
    scheduler = GaitScheduler(robot_interface)

    # Warmup so the oscillators settle before we record.
    warmup_steps = int(2.0 / (freq * dt))
    for _ in range(warmup_steps):
        cpg.run()

    record_steps = int(n_cycles / (freq * dt))
    contact_log = {f: [] for f in feet_order}
    for _ in range(record_steps):
        cpg.run()
        phases = cpg.get_phase_outputs()
        _, contact = scheduler.compute(phases)
        for foot in feet_order:
            contact_log[foot].append(contact[foot])

    # Rows = feet, cols = time. Flip so FL ends up on top.
    data = np.array([contact_log[f] for f in feet_order])[::-1]

    plt.figure(figsize=(10, 3))
    plt.imshow(data, aspect='auto', cmap='gray', interpolation='nearest')
    plt.yticks(range(len(feet_order)), [f.name for f in feet_order[::-1]])
    plt.xlabel("Time (samples)")
    plt.title(f"Footfall Pattern — {gait.name} "
              f"[freq={freq:.2f} Hz, duty={duty:.2f}]")

    samples_per_cycle = int(1 / (freq * dt))
    for c in range(1, n_cycles):
        plt.axvline(c * samples_per_cycle, color='red', linestyle='--', alpha=0.3)

    plt.colorbar(label="Contact (1=stance, 0=swing)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gait = Gait.GALLOP  

    # print("=" * 50)
    # print(f"CPG output — {gait.name}")
    # show_cpg_output(gait)

    print("=" * 50)
    print(f"Trajectory — {gait.name}")
    show_trajectory(gait)

    # print("=" * 50)
    # print(f"Footfall — {gait.name}")
    # plot_single_gait_footfall(gait)

    # print("=" * 50)
    compute_trajectory_area(gait)
