import sys, os
import numpy as np 
import matplotlib.pyplot as plt
from kuramoto_cpg import KuramotoCpg
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import NEURON_TO_FOOT_DICT
from trajectory_builder import TrajectoryBuilder, Coordinate, OvalOffset, EllipsoidConfig, GaitScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import RobotInterface, State, Gait, Mode, Foot, TrajectoryMethod
from shared_module.settings_loader import load_settings_from_file

base_size = 18
plt.rcParams.update({
    'font.size': base_size,        # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size - 2,  # X tick size
    'ytick.labelsize': base_size - 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})

def apply_duty_factor(phase: float, duty: float) -> float:
    """
    Warp phase according to duty factor.
    Input:  phase in radians [0, 2π)
    Output: warped phase in radians [0, 2π)
    """
    phi = (phase % (2 * np.pi)) / (2 * np.pi)

    if phi < duty:
        gait_phi = 0.5 * (phi / duty)
    else:
        gait_phi = 0.5 + 0.5 * ((phi - duty) / (1 - duty))


    return gait_phi * 2 * np.pi

def test_cpg_output(gait=None, freq=None, duty=None):

    robot_interface = RobotInterface(starting_state=State(gait=gait, mode=Mode.MOVING, frequency=freq))
    robot_interface.dt = 0.001
    cpg = KuramotoCpg(robot_interface)
    second = 2.0 
    steps = int((second) / robot_interface.dt)
    output = {}
    for step in range(steps):
        cpg.run()
        theta = cpg.get_phase_outputs()
        theta_w = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_w[i] = apply_duty_factor(theta[i], duty)
        c = np.cos(theta_w)
        output[step] = c
    time = np.arange(steps) * robot_interface.dt
    plt.figure(figsize=(10, 6))
    for i in range(cpg.neurons_cnt):
        foot = NEURON_TO_FOOT_DICT[i]
        # Neurons 3 and 4 (index 2, 3) are dashed so they remain visible when overlapping 1 and 2
        linestyle = '--' if i >= 2 else '-'
        plt.plot(time, [output[step][i] for step in range(steps)], label=f'{foot.name}', linestyle=linestyle)
    plt.title('Kuramoto CPG Neuron Outputs Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel(r'$\cos(\theta)$')
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.grid()
    plt.show()

def test_gait_transtion():
    """Visualise CPG outputs while the gait blends from WALK to AMBLE.

    Mirrors :func:`test_cpg_output` (CPG output traces vs time) but, after a
    short warmup, advances a ``blending_factor`` by ``blend_rate * dt`` and
    rebuilds the CPG's pairwise phase-offset matrix every step from a
    linearly interpolated ``WALK -> AMBLE`` phase tuple. The Kuramoto
    coupling then smoothly steers the oscillators toward the new
    relative-phase pattern.

    Why we update the offset matrix directly instead of calling
    :meth:`KuramotoCpg.set_gait`: ``set_gait`` snaps each neuron's phase to
    ``gait.value[i]`` on every call, which would teleport oscillators on
    every blend step and erase the very transition we're trying to plot.
    """
    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.WALK, mode=Mode.MOVING, frequency=0.5)
    )
    robot_interface.dt = 0.001
    cpg = KuramotoCpg(robot_interface)

    # Timeline: warmup at WALK, then ramp blend toward AMBLE until it saturates.
    seconds = 6.0
    warmup_seconds = 1.0
    blend_rate = 0.4   # per second; saturates blending_factor at 1.0 in 2.5 s
    steps = int(seconds / robot_interface.dt)
    warmup_steps = int(warmup_seconds / robot_interface.dt)

    walk_phases = np.array(Gait.WALK.value, dtype=float)
    amble_phases = np.array(Gait.AMBLE.value, dtype=float)

    output = {}
    blend_log = np.zeros(steps)
    blending_factor = 0.0

    def interp_angle(a, b, s):

        delta = (b - a + np.pi) % (2*np.pi) - np.pi

        return a + s * delta
    
    for step in range(steps):
        # Advance the blend after warmup. Clamp at 1.0 so we settle at AMBLE
        # for the tail of the recording instead of overshooting.
        if step >= warmup_steps and blending_factor < 1.0:
            blending_factor = min(1.0, blending_factor + blend_rate * robot_interface.dt)
            blend_phases = np.array([
                interp_angle(a, b, blending_factor)
                for a, b in zip(walk_phases, amble_phases)
            ])
            cpg._build_phase_offset_matrix(blend_phases)
        blend_log[step] = blending_factor
        cpg.run()
        output[step] = cpg.get_oscillator_outputs()

    time = np.arange(steps) * robot_interface.dt

    fig, (ax_out, ax_blend) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    for i in range(cpg.neurons_cnt):
        foot = NEURON_TO_FOOT_DICT[i]
        # Match test_cpg_output's convention: dash neurons 3/4 so they stay
        # legible when overlapping 1/2 mid-blend.
        linestyle = '--' if i >= 2 else '-'
        ax_out.plot(
            time,
            [output[step][i] for step in range(steps)],
            label=f'{foot.name}',
            linestyle=linestyle,
        )
    ax_out.axvline(warmup_seconds, color='gray', linestyle=':', alpha=0.7,
                   label='blend start')
    ax_out.set_title('Kuramoto CPG Outputs — WALK to AMBLE blend')
    ax_out.set_ylabel('Neuron output')
    ax_out.legend(loc='upper right')
    ax_out.grid()

    ax_blend.plot(time, blend_log, color='tab:purple', linewidth=1.5)
    ax_blend.axvline(warmup_seconds, color='gray', linestyle=':', alpha=0.7)
    ax_blend.set_xlabel('Time (seconds)')
    ax_blend.set_ylabel('blend (WALK→AMBLE)')
    ax_blend.set_ylim(-0.05, 1.05)
    ax_blend.grid()

    plt.tight_layout()
    plt.show()


def test_duty_factor():
    robot_interface = RobotInterface(
        starting_state=State(
            gait=Gait.TROT,
            mode=Mode.MOVING,
            frequency=0.5
        )
    )
    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)
    traj = TrajectoryBuilder(robot_interface)

    second = 5.0
    steps = int(second / robot_interface.dt)

    output = {}

    for step in range(steps):
        cpg.run()

        phases = cpg.get_phase_outputs()   # <-- IMPORTANT: use phases, not outputs
        warped = [traj.apply_duty_factor(p) for p in phases]

        output[step] = warped

    time = np.arange(steps) * robot_interface.dt

    plt.figure(figsize=(10, 6))
    for i in range(cpg.neurons_cnt):
        foot = NEURON_TO_FOOT_DICT[i]
        linestyle = '--' if i >= 2 else '-'

        plt.plot(
            time,
            [output[step][i] for step in range(steps)],
            label=f'{foot.name}',
            linestyle=linestyle
        )

    plt.title('Duty Factor Warped Phases')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Warped Phase (rad)')
    plt.legend()
    plt.grid()
    plt.show()

def test_trajectory_builder(gait=None, freq_in=None, duty_in=None):

    if gait is None:
        raise ValueError("Gait cannot be None")
    conf, freq, duty = load_settings_from_file(gait)

    if freq_in is not None:
        freq = freq_in
    if duty_in is not None:
        duty = duty_in
        conf

    robot_interface = RobotInterface(starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=freq))
    robot_interface.active_gait = gait
    robot_interface.active_traj_params = conf
    robot_interface.duty_factor = duty
    robot_interface.dt = 0.001
    cpg = KuramotoCpg(robot_interface)
    
    step_height = {Foot.FL: 0.14, Foot.FR: 0.14, Foot.RL: 0.1, Foot.RR: 0.1}
    builder = TrajectoryBuilder(robot_interface, width=0.1, height=step_height)

    second = 20.0 
    steps = int(second / robot_interface.dt)
    foot_trajectories = {}
    for step in range(steps):
        cpg.run()
        output = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        foot_trajectories[step], _ = builder.build_trajectory(output, vel)

    # Plot the foot trajectories — only the last 2 gait cycles (2 s at 0.5 Hz)
    # so the dashed pattern on RL/RR remains visible instead of being filled in
    # by thousands of overlapping line segments.
    cycles_to_show = int(2.0 / robot_interface.dt)  # 2 seconds of data
    display_traj = dict(list(foot_trajectories.items())[-cycles_to_show:])
    plt.figure(figsize=(10, 6))
    for i, foot in enumerate([Foot.FL, Foot.FR, Foot.RL, Foot.RR]):
        x = [pos[foot].x for pos in display_traj.values()]
        z = [pos[foot].z for pos in display_traj.values()]
        linestyle = '--' if i % 2 == 1 else '-'
        plt.plot(x, z, label=f'{foot.name} Foot Trajectory', linestyle=linestyle)
    plt.title('Foot Trajectories from Trajectory Builder')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.legend()
    plt.grid()
    plt.show()

def test_3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5)
    )

    robot_interface.foot_positions = {
        Foot.FL: [0.1795716643722928, 0.14149099862255338, 0.18806348777447116],
        Foot.FR: [0.1795716705870714, -0.141490825432425, 0.18806342628408043],
        Foot.RL: [-0.26887864336080886, 0.14201085282060438, 0.19723781364348866],
        Foot.RR: [-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]
    }

    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface)

    seconds = 20.0
    steps = int(seconds / robot_interface.dt)

    foot_trajectories = []

    for _ in range(steps):
        cpg.run()
        phase = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        traj, _ = builder.build_trajectory(phase, vel)
        foot_trajectories.append(traj)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for foot in [Foot.FL, Foot.FR, Foot.RL, Foot.RR]:
        xs = [traj[foot].x for traj in foot_trajectories]
        ys = [traj[foot].y for traj in foot_trajectories]
        zs = [traj[foot].z for traj in foot_trajectories]

        ax.plot(xs, ys, zs, label=f"{foot.name}")

    ax.set_title("3D Foot Trajectories")
    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Y (lateral)")
    ax.set_zlabel("Z (height)")
    ax.legend()
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1,1,1])
    plt.show()

def test_3d_direction():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5)
    )

    robot_interface.foot_positions = {
        Foot.FL: [0.1795716643722928, 0.14149099862255338, 0.18806348777447116],
        Foot.FR: [0.1795716705870714, -0.141490825432425, 0.18806342628408043],
        Foot.RL: [-0.26887864336080886, 0.14201085282060438, 0.19723781364348866],
        Foot.RR: [-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]
    }

    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface)

    seconds = 20.0
    steps = int(seconds / robot_interface.dt)

    foot_trajectories = []

    for _ in range(steps):
        cpg.run()
        phase = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        traj, _ = builder.build_trajectory(phase, vel)
        foot_trajectories.append(traj)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        Foot.FL: "tab:blue",
        Foot.FR: "tab:orange",
        Foot.RL: "tab:green",
        Foot.RR: "tab:red",
    }
    arrow_stride = max(1, steps // 120)
    arrow_length = 0.02

    for foot in [Foot.FL, Foot.FR, Foot.RL, Foot.RR]:
        xs = np.array([traj[foot].x for traj in foot_trajectories])
        ys = np.array([traj[foot].y for traj in foot_trajectories])
        zs = np.array([traj[foot].z for traj in foot_trajectories])

        ax.plot(xs, ys, zs, label=f"{foot.name}", color=colors[foot], linewidth=1.6)

        for i in range(0, len(xs) - 1, arrow_stride):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            dz = zs[i + 1] - zs[i]
            norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            if norm == 0.0:
                continue
            ax.quiver(
                xs[i], ys[i], zs[i],
                dx / norm, dy / norm, dz / norm,
                length=arrow_length,
                normalize=True,
                color=colors[foot],
                linewidth=0.8,
            )

    ax.set_title("3D Foot Trajectories with Direction Arrows")
    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Y (lateral)")
    ax.set_zlabel("Z (height)")
    ax.legend()

    plt.show()

def test_phase_space():
    import matplotlib.pyplot as plt
    import numpy as np

    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5)
    )

    robot_interface.foot_positions = {
        Foot.FL: [0.1795716643722928, 0.14149099862255338, 0.18806348777447116],
        Foot.FR: [0.1795716705870714, -0.141490825432425, 0.18806342628408043],
        Foot.RL: [-0.26887864336080886, 0.14201085282060438, 0.19723781364348866],
        Foot.RR: [-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]
    }

    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface)

    seconds = 10
    steps = int(seconds / robot_interface.dt)

    phases = []
    xs = []
    zs = []

    for _ in range(steps):
        cpg.run()
        theta = cpg.get_phase_outputs()
        traj = builder.build_trajectory(theta)

        phases.append(theta[0])      # examine FL leg
        xs.append(traj[Foot.FL].x)
        zs.append(traj[Foot.FL].z)

    phases = np.unwrap(np.array(phases))

    fig, ax = plt.subplots(1,3, figsize=(15,4))

    ax[0].plot(phases, xs)
    ax[0].set_title("Phase vs X")
    ax[0].set_xlabel("θ")
    ax[0].set_ylabel("x")

    ax[1].plot(phases, zs)
    ax[1].set_title("Phase vs Z")
    ax[1].set_xlabel("θ")
    ax[1].set_ylabel("z")

    ax[2].plot(xs, zs)
    ax[2].set_title("Foot trajectory (X–Z)")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("z")

    for a in ax:
        a.grid()

    plt.show()

def generate_foot_traj_for_IK():
    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5)
    )

    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface)

    seconds = 20.0
    steps = int(seconds / robot_interface.dt)

    foot_trajectories = []

    for _ in range(steps):
        cpg.run()
        phase = cpg.get_phase_outputs()
        vel = cpg.get_phase_velocities()
        traj, _ = builder.build_trajectory(phase, vel)
        foot_trajectories.append(traj)

    pretty_print_foot_positions("Foot position 0", foot_trajectories[steps // 10])
    
def pretty_print_foot_positions(title: str, foot_positions: dict[Foot, Coordinate]):
    print(f"\n{title}")
    print("-" * 40)

    for foot in Foot:
        pos = foot_positions[foot]
        print(
            f"{foot.name:>3}  "
            f" [{float(pos.x): .4f},  "
            f"{float(pos.y): .4f},  "
            f"{float(pos.z): .4f}]"
        )

    print("-" * 40)

def test_foot_positions():
    builder = TrajectoryBuilder()
    foot_targets = {
        Foot.FL: Coordinate(-0.018626816173289977, 0.14199272947995584, 0.271932972842175),
        Foot.FR: Coordinate(-0.01862681953944406, -0.1420072706603052, 0.27193297619052603),
        Foot.RL: Coordinate(-0.41655953150783753, 0.14199273012130323, 0.2828365249660655),
        Foot.RR: Coordinate(-0.4165598918564412, -0.14200727017798084, 0.2828368957807831)
    }

    foot_target = builder.transform_to_hip_coordinates(foot_targets)
    print(foot_target)


def test_ellipsoid_traj():
    """Visualise the asymmetric ellipsoid foot trajectory for all four legs.
    The left subplot shows front legs, the right shows rear legs.
    Tweak EllipsoidConfig parameters to explore the trajectory shape.
    Supports asymmetric fore/hind reach, top/bottom height, rotation, skew,
    and duty factor just like the oval trajectory."""

    robot_interface = RobotInterface(
        starting_state=State(gait=Gait.AMBLE, mode=Mode.MOVING, frequency=1.75),
        trajectory_method=TrajectoryMethod.ELLIPSOID
    )
    robot_interface.dt = 0.001

    cpg = KuramotoCpg(robot_interface)

    # EllipsoidConfig now has independent front_* / rear_* parameters.
    # Front legs (FL, FR) and rear legs (RL, RR) can have different shapes.
    cfg = EllipsoidConfig(
            front_x_fore   = 0.123,  
            front_x_hind   = 0.095,
            front_z_top    = 0.104,
            front_z_bottom = 0.025,
            # front_rotation = 0.082300,
            # front_skew     = -0.022200,
            front_rotation = 0.0,
            front_skew     = 0.0,

            rear_x_fore    = 0.075,
            rear_x_hind    = 0.122,
            rear_z_top     = 0.095,
            rear_z_bottom  = 0.005,
            # rear_rotation  = -0.065400,
            # rear_skew      = -0.035000,
            rear_rotation=0.0,
            rear_skew=0.0
        )
    # duty_factor: fraction of cycle in stance (0.5 = symmetric, 0.7 = longer stance)
    builder = TrajectoryBuilder(robot_interface, ellipsoid_config=cfg, duty_factor=0.4)
    planner = GaitScheduler(robot_interface)
    # Simulate 5 s, collect trajectories
    seconds = 5.0
    steps   = int(seconds / robot_interface.dt)
    foot_trajectories = []
    phase_zero_points = {foot: [] for foot in Foot}
    threshold = 0.01

    for _ in range(steps):
        cpg.run()
        phase = cpg.get_phase_outputs()
        
        vel   = cpg.get_phase_velocities()
        # phase_norm, contact = planner.compute(phase)
        # traj, _ = builder.new_build_ellipsoid_trajectory(phase_norm, contact, vel)
        traj, _ = builder.build_ellipsoid_trajectory(phase, vel)
        foot_trajectories.append(traj)
        for i, foot in enumerate([Foot.FL, Foot.FR, Foot.RL, Foot.RR]):
            if -threshold < phase[i] < threshold:
                phase_zero_points[foot].append((traj[foot].x, traj[foot].z))

    # Show only the last 2 gait cycles for a clean shape
    cycles_to_show = int(2.0 / robot_interface.dt)
    display_traj = foot_trajectories[-cycles_to_show:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f'Ellipsoid Foot Trajectories (X–Z)  '
        f'[front: x_fore={cfg.front_x_fore}, x_hind={cfg.front_x_hind}, '
        f'z_top={cfg.front_z_top}, z_bot={cfg.front_z_bottom}, '
        f'rot={cfg.front_rotation:.2f}rad | '
        f'rear: x_fore={cfg.rear_x_fore}, z_top={cfg.rear_z_top}, '
        f'rot={cfg.rear_rotation:.2f}rad | duty={robot_interface.duty_factor}]'
    )

    groups = [
        (axes[0], [Foot.FL, Foot.FR], 'Front Legs'),
        (axes[1], [Foot.RL, Foot.RR], 'Rear Legs'),
    ]
    linestyles = ['-', '--']
    for ax, feet, title in groups:
        for foot, ls in zip(feet, linestyles):
            x = [t[foot].x for t in display_traj]
            z = [t[foot].z for t in display_traj]
            ax.plot(x, z, linestyle=ls, label=foot.name)
            # Blue dot at the stationary (stance) position — zero offset maps to
            # builder.stance_positions[foot] after transform_relative_world_to_hip
            sx = builder.stance_positions[foot].x
            sz = builder.stance_positions[foot].z
            ax.plot(sx, sz, 'o', color='blue', markersize=7, zorder=5,
                    label='stance' if foot == feet[0] else None)

            if phase_zero_points[foot]:
                gx, gz = zip(*phase_zero_points[foot])
                ax.scatter(gx, gz, color='lime', s=30, zorder=6,
                        label='phase ≈ 0' if foot == feet[0] else None)
        ax.set_title(title)
        ax.set_xlabel('X — forward / back (m)')
        ax.set_ylabel('Z — height (m)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def plot_single_gait_footfall():
    gait = Gait.AMBLE
    feet_order = [Foot.FL, Foot.FR, Foot.RR, Foot.RL]
    duty_factor = 0.40  
    frequency = 1.75000
    dt = 0.002
    duty = duty_factor
    n_cycles = 4

    robot_interface = RobotInterface(
        starting_state=State(gait=gait, mode=Mode.MOVING, frequency=frequency)
    )
    robot_interface.dt = dt
    robot_interface.duty_factor = duty

    cpg = KuramotoCpg(robot_interface)
    scheduler = GaitScheduler(robot_interface)

    # Warmup
    warmup_steps = int(2.0 / (frequency * dt))
    for _ in range(warmup_steps):
        cpg.run()

    # Record
    record_steps = int(n_cycles / (frequency * dt))
    contact_log = {f: [] for f in feet_order}

    for _ in range(record_steps):
        cpg.run()
        phases = cpg.get_phase_outputs()
        _, contact = scheduler.compute(phases)

        for foot in feet_order:
            contact_log[foot].append(contact[foot])

    # Convert to matrix (rows = feet, cols = time)
    data = np.array([contact_log[f] for f in feet_order])

    # Flip so FL is top
    data = data[::-1]

    # Plot as image
    plt.figure(figsize=(10, 3))
    plt.imshow(data, aspect='auto', cmap='gray', interpolation='nearest')

    plt.yticks(
        range(len(feet_order)),
        [f.name for f in feet_order[::-1]]
    )

    plt.xlabel("Time (samples)")
    plt.title(f"Footfall Pattern — {gait.name}")

    # Optional: vertical lines for cycles
    samples_per_cycle = int(1 / (frequency * dt))
    for c in range(1, n_cycles):
        plt.axvline(c * samples_per_cycle, color='red', linestyle='--', alpha=0.3)

    plt.colorbar(label="Contact (1=stance, 0=swing)")
    plt.tight_layout()
    plt.show()


def test_main():
    print("==================================================")
    print("Testing footfall pattern")
    #plot_single_gait_footfall()

    print("==================================================")
    print("Testing ellipsoid trajectory")
    test_trajectory_builder(gait=Gait.TROT)
    # freq = 1.40
    # duty = 0.20
    # gait = Gait.WALK
    # test_cpg_output(gait, freq, duty)
    # freq = 1.750
    # duty = 0.40
    # gait = Gait.AMBLE
    # test_cpg_output(gait, freq, duty)
    # freq = 2.20
    # duty = 0.50
    # gait = Gait.TROT
    # test_cpg_output(gait, freq, duty)
    # freq = 2.350
    # duty = 0.45
    # gait = Gait.CANTER
    # test_cpg_output(gait, freq, duty)
    # freq = 2.5
    # duty = 0.65
    # gait = Gait.GALLOP
    # test_cpg_output(gait, freq, duty)


    #test_gait_transtion()
    test_ellipsoid_traj()

    print("==================================================")
    return

    print("==================================================")
    print("Testing oval trajectory")
    test_oval_traj()
    print("==================================================")
    return

    generate_foot_traj_for_IK()
    print("==================================================")
    print("Testing CPG output")
    test_cpg_output()
    print("==================================================")

    print("==================================================")
    print("Testing Trajectory Builder output")
    test_trajectory_builder()
    print("==================================================")

    print("==================================================")
    print("Testing Trajectory Builder output")
    test_3d_direction()
    print("==================================================")

    print("==================================================")
    test_3d()
    print("==================================================")
    print("==================================================")
    test_3d_direction()
    print("==================================================")
    print("==================================================")
    test_phase_space()
    print("==================================================")

if __name__ == '__main__':
    test_main()
