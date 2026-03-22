import os
import sys

from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from controllers.ik_controller import IKController
from shared_module.robot_state import Foot, RobotInterface, State, Mode, Gait

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
        sim.sim(controller=controller, sim_length=5, slow_factor=1.0)

        cot = sim.compute_CoT()
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

if __name__ == "__main__":
    main()

