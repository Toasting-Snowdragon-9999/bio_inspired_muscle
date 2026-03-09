import os
import sys
from mujoco_sim import MujocoSim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cpg.kuramoto_cpg import KuramotoCpg
from pd.muscle_like_pd import MusclePdController

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene.xml')
    sim = MujocoSim(xml_path, window_scale=2.0, print_camera_config=0)
    # sim.enable_air_mode()
    # sim.enable_graph()
    pd_controller = MusclePdController(kp=40, kd=4, passive_kp=40, passive_kd=4)
    cpg_controller = KuramotoCpg(gait='trot', frequency=0.7, pd_controller=pd_controller)
    sim.sim(controller=cpg_controller, sim_length=-1, slow_factor=10.0)

if __name__ == "__main__":
    main()

