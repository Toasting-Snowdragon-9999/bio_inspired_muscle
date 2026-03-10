import sys, os
import numpy as np 
import matplotlib.pyplot as plt
from kuramoto_cpg import KuramotoCpg
from trajectory_builder import TrajectoryBuilder, Coordinate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import RobotInterface, State, Gait, Mode, Foot

def test_cpg_output():
    robot_interface = RobotInterface(starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5))
    robot_interface.dt = 0.001
    cpg = KuramotoCpg(robot_interface)
    second = 20.0 
    steps = int(second / robot_interface.dt)
    output = {}
    for step in range(steps):
        cpg.run()
        output[step] = cpg.get_phase_outputs()
    time = np.arange(steps) * robot_interface.dt
    plt.figure(figsize=(10, 6))
    for i in range(cpg.neurons_cnt):
        plt.plot(time, [output[step][i] for step in range(steps)], label=f'Neuron {i+1}')
    plt.title('Kuramoto CPG Neuron Outputs Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Neuron Output')
    plt.legend()
    plt.grid()
    plt.show()

def test_trajectory_builder():
    robot_interface = RobotInterface(starting_state=State(gait=Gait.TROT, mode=Mode.MOVING, frequency=0.5))

    robot_interface.dt = 0.001
    cpg = KuramotoCpg(robot_interface)
    builder = TrajectoryBuilder(robot_interface)

    second = 20.0 
    steps = int(second / robot_interface.dt)
    foot_trajectories = {}
    for step in range(steps):
        cpg.run()
        output = cpg.get_phase_outputs()
        foot_trajectories[step] = builder.build_trajectory(output)

    # Plot the foot trajectories
    plt.figure(figsize=(10, 6))
    for foot in [Foot.FL, Foot.FR, Foot.RL, Foot.RR]:
        x = [pos[foot].x for pos in foot_trajectories.values()]
        z = [pos[foot].z for pos in foot_trajectories.values()]
        plt.plot(x, z, label=f'{foot.name} Foot Trajectory')
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
        traj = builder.build_trajectory(phase)
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
        traj = builder.build_trajectory(phase)
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
        traj = builder.build_trajectory(phase)
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

def test_main():
    generate_foot_traj_for_IK()
    return
    print("==================================================")
    print("Testing CPG output")
    test_cpg_output()
    print("==================================================")
    print("==================================================")
    print("Testing Trajectory Builder output")
    test_trajectory_builder()
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
