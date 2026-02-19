import os
import sys
from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cpg.kuramoto_cpg import KuramotoCpg

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, window_scale=2.0)
    # sim.enable_air_mode()
    # sim.enable_graph()
    controller = KuramotoCpg(gait='trot', frequency=0.7)
    sim.sim(controller=controller, sim_length=-1, slow_factor=1.0)

if __name__ == "__main__":
    main()

