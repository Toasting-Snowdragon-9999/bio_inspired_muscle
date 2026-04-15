import os
import sys
from multiprocessing import Lock

from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cpg.trajectory_builder import OvalOffset

from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait, TrajectoryMethod

# WALK: Lowest CoT: 0.8135 at 2.20 Hz
# TROT: 
# BOUND: Lowest CoT: 1.8478 at 3.00 Hz
# PACE: 
# GALLOP:

def main():
    cot_values = {} # Freq : CoT

    # Sweep 0.5 – 3.0 Hz in 0.1 Hz steps (26 points).
    # Frequencies above ~3 Hz are outside the Go2's stable trot range
    # and will cause the robot to fall, giving meaningless CoT readings.
    frequencies = [round(0.5 + i * 0.1, 2) for i in range(26)]

    for i, freq in enumerate(frequencies):
        print(f"Running test iteration {i+1}/{len(frequencies)}  (freq={freq:.1f} Hz)...")
        starting_state = State(mode=Mode.MOVING, gait=Gait.BOUND, frequency=freq)
        robot_interface = RobotInterface(starting_state)

        xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
        sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
        # sim.enable_air_mode(0.5)
        # sim.enable_graph()
        # sim.enable_joint_sliders()
        # sim.enable_joint_graph()
        step_height = {
            Foot.FL: 0.14,
            Foot.FR: 0.14,
            Foot.RL: 0.1,
            Foot.RR: 0.1
        }
        
        params = (
            0.2,  # a: learning rate of impedance adaptation
            5.0,  # b: sensitivity of impedance adaptation to velocity error
            0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
        )

        controller = IKController(robot_interface=robot_interface, stride_length=0.1, step_height=step_height, params=params, use_adaptive_pd=True)
        cot = sim.headless_sim(controller=controller, sim_length=5)

        if cot is None:
            # Robot fell or did not move forward — skip this frequency
            print(f"  → CoT = N/A (robot fell or did not advance)")
        else:
            print(f"  → CoT = {cot:.4f}")
            cot_values[freq] = cot

    # ── Results ──────────────────────────────────────────────────────────
    print("\n=== CoT sweep results ===")
    print(f"{'Freq (Hz)':>10}  {'CoT':>10}")
    print("-" * 24)
    for f, c in sorted(cot_values.items()):
        print(f"{f:>10.2f}  {c:>10.4f}")

    best_freq = min(cot_values, key=cot_values.get)
    print(f"\nLowest CoT: {cot_values[best_freq]:.4f} at {best_freq:.2f} Hz")

def quick_test():
    robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=Gait.TROT, frequency=1.2), trajectory_method=TrajectoryMethod.OVAL)

    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)
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
    max_freq = 2.2
    min_freq = 1.6
    step_size = 0.05
    best_COT = 999
    best_freq = 0
    robot_interface.frequency = min_freq
    for _ in range(int((max_freq - min_freq) / step_size) + 1):
        robot_interface.frequency += step_size
        cot = sim.headless_sim(controller=controller, sim_length=3)
        if cot is None:
            print("Freq doesnt work: ", robot_interface.frequency)
            continue
        if cot < best_COT:
            best_COT = cot
            best_freq = robot_interface.frequency

    print("Cost of Transport:", best_COT)
    print("Best Frequency:", best_freq)
    print("min freq:", min_freq, " max_freq:", max_freq, " step_size:", step_size)

    test_result = open("python/sim/cot_test.txt", "a")
    test_result.write(f"[{robot_interface.current_gait}] Best CoT: {best_COT:.4f} at {best_freq:.2f} Hz\n")
    test_result.close()

def get_offset(gait):
    if gait == Gait.WALK:
        return {
            OvalOffset.X_FFORE: 0.0,
            OvalOffset.X_FHIND: 0.5,
            OvalOffset.Z_FTOP: 0.75,
            OvalOffset.Z_FBOTTOM: 0.25,
            
            OvalOffset.X_RFORE: 0.05,
            OvalOffset.X_RHIND: 0.15,
            OvalOffset.Z_RTOP: 0.05,
            OvalOffset.Z_RBOTTOM: 0.02,

        }
    elif gait == Gait.TROT:
        return {
            OvalOffset.X_FFORE: 0.15,
            OvalOffset.X_FHIND: 0.15,
            OvalOffset.Z_FTOP: 0.05,
            OvalOffset.Z_FBOTTOM: 0.02,
            
            OvalOffset.X_RFORE: 0.15,
            OvalOffset.X_RHIND: 0.15,
            OvalOffset.Z_RTOP: 0.05,
            OvalOffset.Z_RBOTTOM: 0.02,

        }
    elif gait == Gait.BOUND:
        return {
            OvalOffset.X_FFORE: 0.2,
            OvalOffset.X_FHIND: 0.1,
            OvalOffset.Z_FTOP: 0.05,
            OvalOffset.Z_FBOTTOM: 0.02,
            
            OvalOffset.X_RFORE: 0.1,
            OvalOffset.X_RHIND: 0.2,
            OvalOffset.Z_RTOP: 0.05,
            OvalOffset.Z_RBOTTOM: 0.02,

        }
    elif gait == Gait.PACE:
        return {
            OvalOffset.X_FFORE: 0.1,
            OvalOffset.X_FHIND: 0.2,
            OvalOffset.Z_FTOP: 0.05,
            OvalOffset.Z_FBOTTOM: 0.02,
            
            OvalOffset.X_RFORE: 0.2,
            OvalOffset.X_RHIND: 0.1,
            OvalOffset.Z_RTOP: 0.05,
            OvalOffset.Z_RBOTTOM: 0.02,

        }
    elif gait == Gait.GALLOP:
        return {
            OvalOffset.X_FFORE: 0.2,
            OvalOffset.X_FHIND: 0.1,
            OvalOffset.Z_FTOP: 0.05,
            OvalOffset.Z_FBOTTOM: 0.02,
            
            OvalOffset.X_RFORE: 0.1,
            OvalOffset.X_RHIND: 0.2,
            OvalOffset.Z_RTOP: 0.05,
            OvalOffset.Z_RBOTTOM: 0.02,

        }
    else:
        raise ValueError(f"Unsupported gait: {gait}")


def multi_thread_test():
    file_lock = Lock()

    def safe_write(log_line):
        with file_lock:
            with open("python/sim/cot_test.txt", "a") as f:
                f.write(log_line)

    def sim(oval_offset, gait, min_freq, max_freq, step_size):
        robot_interface = RobotInterface(starting_state=State(mode=Mode.MOVING, gait=gait, frequency=1.2), trajectory_method=TrajectoryMethod.OVAL)

        xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
        sim = MujocoSim(xml_path, robot_interface=robot_interface, window_scale=2.0)


        params = (
            0.2,  # a: learning rate of impedance adaptation
            5.0,  # b: sensitivity of impedance adaptation to velocity error
            0.05  # k: baseline stiffness (added to adapted stiffness to prevent singularity when error is near zero)
        )

        controller = IKController(robot_interface=robot_interface, stride_length=None, step_height=None, params=params, use_adaptive_pd=True, oval_offset=oval_offsets)
        max_freq = 2.2
        min_freq = 1.6
        step_size = 0.05
        best_COT = 999
        best_freq = 0
        robot_interface.frequency = min_freq
        for _ in range(int((max_freq - min_freq) / step_size) + 1):
            robot_interface.frequency += step_size
            cot = sim.headless_sim(controller=controller, sim_length=3)
            if cot is None:
                print("Freq doesnt work: ", robot_interface.frequency)
                continue
            if cot < best_COT:
                best_COT = cot
                best_freq = robot_interface.frequency

        print("Cost of Transport:", best_COT)
        print("Best Frequency:", best_freq)
        print("min freq:", min_freq, " max_freq:", max_freq, " step_size:", step_size)

        test_result = open("python/sim/cot_test.txt", "a")
        test_result.write(f"[{robot_interface.current_gait}] Best CoT: {best_COT:.4f} at {best_freq:.2f} Hz\n")
        test_result.close()
    
    gaits = [Gait.WALK, Gait.TROT, Gait.BOUND, Gait.PACE, Gait.GALLOP]

if __name__ == "__main__":
    quick_test()

