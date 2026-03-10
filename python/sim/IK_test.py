# # import os
# # import sys
# # import numpy as np
# # import mujoco as mj

# # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# # from mujoco_sim import MujocoSim
# # from shared_module.robot_state import RobotInterface, Joint
# # from inverse_kinematics.inverse_kin import (
# #     LevenbergMarquardtIK,
# #     forward_kinematics,
# #     Leg,
# #     LEG_CONFIG
# # )

# # # -------------------------------------------------
# # # Create robot interface (not used for control here)
# # # -------------------------------------------------

# # starting_state = {j: 0.0 for j in Joint}
# # robot_interface = RobotInterface(starting_state)

# # # -------------------------------------------------
# # # Load MuJoCo model
# # # -------------------------------------------------

# # model_path = os.path.join(os.path.dirname(__file__), "go2/scene.xml")
# # sim = MujocoSim(model_path, robot_interface)

# # mj.mj_resetData(sim.model, sim.data)

# # # -------------------------------------------------
# # # Desired foot positions (stable stance)
# # # -------------------------------------------------

# # # foot_targets = {
# # #     Leg.FL: np.array([ 0.15,  0.0955, -0.27]),
# # #     Leg.FR: np.array([ 0.15, -0.0955, -0.27]),
# # #     Leg.RL: np.array([-0.15,  0.0955, -0.27]),
# # #     Leg.RR: np.array([-0.15, -0.0955, -0.27]),
# # # }
# # foot_targets = {
# #     Leg.FL: np.array([0.1795716643722928, 0.14149099862255338, 0.18806348777447116]),
# #     Leg.FR: np.array([0.1795716705870714, -0.141490825432425, 0.18806342628408043]),
# #     Leg.RL: np.array([-0.26887864336080886, 0.14201085282060438, 0.19723781364348866]),
# #     Leg.RR: np.array([-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]),
# # }
# # # fra chris
# # # 'FL_foot': [0.1795716643722928, 0.14149099862255338, 0.18806348777447116], 
# # # 'FR_foot': [0.1795716705870714, -0.141490825432425, 0.18806342628408043], 
# # # 'RL_foot': [-0.26887864336080886, 0.14201085282060438, 0.19723781364348866], 
# # # 'RR_foot': [-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]}


# # # -------------------------------------------------
# # # Solve IK
# # # -------------------------------------------------

# # joint_targets = {}

# # print("\nSolving IK\n")

# # for leg in Leg:

# #     solver = LevenbergMarquardtIK(leg)
# #     goal = foot_targets[leg]

# #     print(f"{leg.name} target:", goal)

# #     result = solver.calculate(goal)

# #     config = LEG_CONFIG[leg]

# #     q = np.array([result[j] for j in config['joints']])

# #     fk = forward_kinematics(q, solver.d_y)
# #     error = np.linalg.norm(fk - goal)

# #     print("  solved q:", q)
# #     print("  fk:", fk)
# #     print("  error:", error)

# #     for joint, angle in result.items():
# #         joint_targets[joint] = angle

# #     print()

# # # -------------------------------------------------
# # # Apply IK pose directly to MuJoCo
# # # -------------------------------------------------

# # print("Applying pose directly to simulation...\n")

# # for joint, angle in joint_targets.items():

# #     joint_name = (
# #         joint.name
# #         .replace('HIP', 'hip_joint')
# #         .replace('THIGH', 'thigh_joint')
# #         .replace('CALF', 'calf_joint')
# #     )

# #     joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

# #     qpos_adr = sim.model.jnt_qposadr[joint_id]

# #     sim.data.qpos[qpos_adr] = angle

# # # update kinematics (no physics)
# # mj.mj_forward(sim.model, sim.data)
# # print("base position:", sim.data.qpos[0:3])

# # # -------------------------------------------------
# # # Print base height
# # # -------------------------------------------------

# # base_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "base_link")
# # com = sim.data.subtree_com[base_id]

# # print("Base COM height:", com[2])

# # # -------------------------------------------------
# # # Start viewer
# # # -------------------------------------------------

# # print("\nStarting viewer...\n")
# # sim.sim()

# import os
# import sys
# import numpy as np
# import mujoco as mj

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from mujoco_sim import MujocoSim
# from shared_module.robot_state import RobotInterface, Joint
# from inverse_kinematics.inverse_kin import (
#     LevenbergMarquardtIK,
#     forward_kinematics,
#     Leg,
#     LEG_CONFIG
# )

# # -------------------------------------------------
# # Create robot interface (not used for control here)
# # -------------------------------------------------

# starting_state = {j: 0.0 for j in Joint}
# robot_interface = RobotInterface(starting_state)

# # -------------------------------------------------
# # Load MuJoCo model
# # -------------------------------------------------

# model_path = os.path.join(os.path.dirname(__file__), "go2/scene.xml")
# sim = MujocoSim(model_path, robot_interface)

# mj.mj_resetData(sim.model, sim.data)

# # -------------------------------------------------
# # Desired foot positions (stable stance)
# # -------------------------------------------------

# foot_targets = {
#     Leg.FL: np.array([0.1795716643722928, 0.14149099862255338, 0.18806348777447116]),
#     Leg.FR: np.array([0.1795716705870714, -0.141490825432425, 0.18806342628408043]),
#     Leg.RL: np.array([-0.26887864336080886, 0.14201085282060438, 0.19723781364348866]),
#     Leg.RR: np.array([-0.26887863626710007, -0.14201177697582473, 0.1972380948189839]),
# }

# # -------------------------------------------------
# # Solve IK
# # -------------------------------------------------

# joint_targets = {}

# print("\nSolving IK\n")

# for leg in Leg:

#     solver = LevenbergMarquardtIK(leg)
#     goal = foot_targets[leg]

#     print(f"{leg.name} target:", goal)

#     init = np.array([0.0, -0.8, 1.6])
#     result = solver.calculate(goal, init_q=init)

#     config = LEG_CONFIG[leg]

#     q = np.array([result[j] for j in config['joints']])

#     fk = forward_kinematics(q, solver.d_y)
#     error = np.linalg.norm(fk - goal)

#     print("  solved q:", q)
#     print("  fk:", fk)
#     print("  error:", error)

#     for joint, angle in result.items():
#         joint_targets[joint] = angle

#     print()

# # -------------------------------------------------
# # Read current joint positions
# # -------------------------------------------------

# current_joint_pos = {}

# for joint in joint_targets:

#     joint_name = (
#         joint.name
#         .replace('HIP', 'hip_joint')
#         .replace('THIGH', 'thigh_joint')
#         .replace('CALF', 'calf_joint')
#     )

#     joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

#     qpos_adr = sim.model.jnt_qposadr[joint_id]

#     current_joint_pos[joint] = sim.data.qpos[qpos_adr]

# # -------------------------------------------------
# # Move robot gradually to IK pose
# # -------------------------------------------------

# print("Moving robot to IK pose...\n")

# steps = 200

# for i in range(steps):

#     alpha = (i + 1) / steps

#     for joint, target_angle in joint_targets.items():

#         joint_name = (
#             joint.name
#             .replace('HIP', 'hip_joint')
#             .replace('THIGH', 'thigh_joint')
#             .replace('CALF', 'calf_joint')
#         )

#         joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

#         qpos_adr = sim.model.jnt_qposadr[joint_id]

#         start_angle = current_joint_pos[joint]

#         angle = (1 - alpha) * start_angle + alpha * target_angle

#         sim.data.qpos[qpos_adr] = angle

#     mj.mj_forward(sim.model, sim.data)

# # -------------------------------------------------
# # Print base position and COM height
# # -------------------------------------------------

# print("base position:", sim.data.qpos[0:3])

# base_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "base_link")
# com = sim.data.subtree_com[base_id]

# print("Base COM height:", com[2])

# # -------------------------------------------------
# # Start viewer
# # -------------------------------------------------

# print("\nStarting viewer...\n")

# sim.sim()

import os
import sys
import numpy as np
import mujoco as mj

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mujoco_sim import MujocoSim
from shared_module.robot_state import RobotInterface, Joint
from inverse_kinematics.inverse_kin import (
    LevenbergMarquardtIK,
    forward_kinematics,
    convert_frame,
    Leg,
    LEG_CONFIG
)

# -------------------------------------------------
# Create robot interface
# -------------------------------------------------

starting_state = {j: 0.0 for j in Joint}
robot_interface = RobotInterface(starting_state)

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------

model_path = os.path.join(os.path.dirname(__file__), "go2/scene.xml")
sim = MujocoSim(model_path, robot_interface)

mj.mj_resetData(sim.model, sim.data)

sim.model.opt.gravity[:] = 0

# -------------------------------------------------
# Desired foot positions (MuJoCo frame)
# -------------------------------------------------

# foot_targets = {
#     Leg.FL: np.array([ 0.18,  0.14, 0.19]),
#     Leg.FR: np.array([ 0.18, -0.14, 0.19]),
#     Leg.RL: np.array([-0.27,  0.14, 0.20]),
#     Leg.RR: np.array([-0.27, -0.14, 0.20]),
# }
foot_targets = {
    Leg.FL: np.array([ 0.20,  0.14149, 0.28]),
    Leg.FR: np.array([ 0.20, -0.14149, 0.28]),
    Leg.RL: np.array([-0.20,  0.14149, 0.28]),
    Leg.RR: np.array([-0.20, -0.14149, 0.28]),
}
foot_targets = {
    Leg.FL: np.array([ 0, 0, 0]),
    Leg.FR: np.array([ 0, 0, 0]),
    Leg.RL: np.array([ 0, 0, 0]),
    Leg.RR: np.array([ 0, 0, 0]),
}
stance_positions = {
    Leg.FL: np.array([-0.01458281454414101, 0.09520361349694849, -0.31155118257490244]),
    Leg.FR: np.array([-0.014582808367942734, -0.0952034741060266, -0.311551225724358]),
    Leg.RL: np.array([-0.07616549739558984, 0.0955063386302885, -0.3023549187505239]),
    Leg.RR: np.array([-0.076165490580238, -0.09550685969947971, -0.30235475609234186]),
}
stance_positions_hipf = {
    Leg.FL: np.array([ 0.01359,  0.10217, -0.26305]),
    Leg.FR: np.array([ 0.01359, -0.10217, -0.26305]),
    Leg.RL: np.array([-0.04041,  0.09929, -0.26944]),
    Leg.RR: np.array([-0.04041, -0.09929, -0.26944]),
}
ninety_degree_stance = {
    Leg.FL: np.array([-0.22359950480560226, 0.09550000000006144, -0.21716310380818882]),
    Leg.FR: np.array([-0.22359950482700078, -0.09550000000001409, -0.2171631037861585]), 
    Leg.RL: np.array([-0.22359950482717342, 0.09549999999997426, -0.2171631037859984]), 
    Leg.RR: np.array([-0.23417369519362705, -0.0955000000000057, -0.2057163889540201]),
}
ninety_degree_stance_new = {
    Leg.FL: np.array([-0.005036816173289977, 0.24416272947995582, 0.008882972842174985]),
    Leg.FR: np.array([-0.005036819539444062, -0.2441772706603052, 0.008882976190526026]),
    Leg.RL: np.array([-0.45696953150783753, 0.24128273012130325, 0.01339652496606547]),
    Leg.RR: np.array([-0.4569698918564412, -0.24129727017798086, 0.013396895780783069]),
}
from_cpg = {
        Leg.FL: np.array([-0.1864,  0.1022, -0.2627]),
        Leg.FR: np.array([ 0.2136, -0.1022, -0.2631]),
        Leg.RL: np.array([ 0.1596,  0.0993, -0.2694]),
        Leg.RR: np.array([-0.2404, -0.0993, -0.2691]),
    }
# template = {
#         Leg.FL: np.array(),
#         Leg.FR: np.array(),
#         Leg.RL: np.array(),
#         Leg.RR: np.array(),
#     }
from_cpg1 = {
        Leg.FL: np.array([-0.0864,   0.1022,  -0.2629]),
        Leg.FR: np.array([ 0.1136,  -0.1022,  -0.2631]),
        Leg.RL: np.array([ 0.0596,   0.0993,  -0.2694]),
        Leg.RR: np.array([-0.1404,  -0.0993,  -0.2693]),
    }
# FL   [-0.0864,   0.1022,  -0.2629]
# FR   [ 0.1136,  -0.1022,  -0.2631]
# RL   [ 0.0596,   0.0993,  -0.2694]
# RR   [-0.1404,  -0.0993,  -0.2693]

COM_OFFSET_X = 0.01687
for leg in foot_targets:
    foot_targets[leg][0] -= COM_OFFSET_X

# -------------------------------------------------
# Solve IK
# -------------------------------------------------

joint_targets = {}

print("\nSolving IK\n")

for leg in Leg:

    solver = LevenbergMarquardtIK(leg)
    config = LEG_CONFIG[leg]

    goal_world = from_cpg1[leg]

    # --- convert MuJoCo → IK frame ---
    goal = convert_frame(goal_world, leg)

    # keep foot in sagittal plane
    goal[1] = solver.d_y

    print(f"{leg.name} target:", goal)

    # init = initial_guess[leg]
    if leg in (Leg.RL, Leg.RR):
        init = np.array([0.00156, -0.79, -1.50])
    else:
        init = np.array([0.00156, 0.79, -1.50])

    result = solver.calculate(goal, init_q=init)

    q = np.array([result[j] for j in config['joints']])

    fk = forward_kinematics(q, solver.d_y)
    error = np.linalg.norm(fk - goal)

    print("  solved q:", q)
    print("  fk:", fk)
    print("  error:", error)

    for joint, angle in result.items():
        joint_targets[joint] = angle

    print()

# -------------------------------------------------
# Read current joint positions
# -------------------------------------------------

current_joint_pos = {}

for joint in joint_targets:

    joint_name = (
        joint.name
        .replace('HIP', 'hip_joint')
        .replace('THIGH', 'thigh_joint')
        .replace('CALF', 'calf_joint')
    )

    joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

    qpos_adr = sim.model.jnt_qposadr[joint_id]

    current_joint_pos[joint] = sim.data.qpos[qpos_adr]

# -------------------------------------------------
# Move robot gradually to IK pose
# -------------------------------------------------

print("Moving robot to IK pose...\n")

steps = 200

for i in range(steps):

    alpha = (i + 1) / steps

    for joint, target_angle in joint_targets.items():

        joint_name = (
            joint.name
            .replace('HIP', 'hip_joint')
            .replace('THIGH', 'thigh_joint')
            .replace('CALF', 'calf_joint')
        )

        joint_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)

        qpos_adr = sim.model.jnt_qposadr[joint_id]

        start_angle = current_joint_pos[joint]

        angle = (1 - alpha) * start_angle + alpha * target_angle

        sim.data.qpos[qpos_adr] = angle

    mj.mj_forward(sim.model, sim.data)

for leg in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
    hip_body = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, leg.replace("_foot", "_hip"))
    hip_pos = sim.data.xpos[hip_body]
    print(leg.replace("_foot", "_hip"), hip_pos)

    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, leg)
    pos = sim.data.xpos[body_id]
    print(leg, pos)

# -------------------------------------------------
# Print COM and support polygon
# -------------------------------------------------

base_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, "base_link")
com = sim.data.subtree_com[base_id]

print("\n--- COM ---")
print("COM position:", com)
print("COM projection (x,y):", com[0], com[1])

print("\n--- Foot targets ---")
for leg, pos in foot_targets.items():
    print(leg.name, ":", pos)

# compute center of support polygon
center_x = np.mean([p[0] for p in foot_targets.values()])
center_y = np.mean([p[1] for p in foot_targets.values()])

print("\nSupport polygon center:", center_x, center_y)

# -------------------------------------------------
# Start viewer
# -------------------------------------------------

print("\nStarting viewer...\n")
hip_body = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"{leg.name.lower()}_hip")
hip_pos = sim.data.xpos[hip_body]
print(hip_pos)

sim.sim()