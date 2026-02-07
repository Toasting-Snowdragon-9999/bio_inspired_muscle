import os 
from mujoco_sim import MujocoSim

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'go2', 'scene_terrain.xml')
    sim = MujocoSim(xml_path)
    sim.sim()  # Run simulation for 10 seconds

if __name__ == "__main__":
    main()
