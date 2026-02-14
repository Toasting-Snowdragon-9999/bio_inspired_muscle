import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpg'))

from mujoco_sim import MujocoSim
from stein_cpg import SteinCpg


def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path)

    controller = SteinCpg(gait='trot', frequency=0.7)

    sim.sim(controller=controller, sim_length=30)


if __name__ == "__main__":
    main()
