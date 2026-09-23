"""@brief Project-wide constants: joint position ranges and midpoints, CPG neuron counts,
and the mapping dictionaries between neurons, feet, joints, MuJoCo actuators/sensors, and labels."""

import numpy as np
try:
    from robot_state import RobotState, State, Joint, Gait, Foot
except ModuleNotFoundError:
    from shared_module.robot_state import RobotState, State, Joint, Gait, Foot

# ── Joint ranges (radians) ──────────────────────────────────────
KNEE_POS_RANGE      = (-2.7227, -0.83776)
FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)
BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)
ABDUCTION_POS_RANGE = (-1.0472, 1.0472)

MID_KNEE_POS = np.mean(KNEE_POS_RANGE)
MID_BACK_HIP_POS = np.mean(BACK_HIP_POS_RANGE)
MID_FRONT_HIP_POS = np.mean(FRONT_HIP_POS_RANGE)
MID_ABDUCTION_POS = np.mean(ABDUCTION_POS_RANGE)

NEURON_CNT = 4

NEURON_TO_FOOT_DICT = {
    0: Foot.FL,
    1: Foot.FR,
    2: Foot.RR,
    3: Foot.RL
}

FOOT_TO_JOINT_DICT = {
    Foot.FL: [Joint.FL_HIP, Joint.FL_THIGH, Joint.FL_CALF],
    Foot.FR: [Joint.FR_HIP, Joint.FR_THIGH, Joint.FR_CALF],
    Foot.RR: [Joint.RR_HIP, Joint.RR_THIGH, Joint.RR_CALF],
    Foot.RL: [Joint.RL_HIP, Joint.RL_THIGH, Joint.RL_CALF]
}

ACTUATOR_DICT = {
    Joint.FR_HIP: 0,   Joint.FR_THIGH: 1,  Joint.FR_CALF: 2,
    Joint.FL_HIP: 3,    Joint.FL_THIGH: 4,   Joint.FL_CALF: 5,
    Joint.RR_HIP: 6,    Joint.RR_THIGH: 7,   Joint.RR_CALF: 8,
    Joint.RL_HIP: 9,     Joint.RL_THIGH: 10,   Joint.RL_CALF: 11,
}

SENSOR_POS_DICT = {
    Joint.FR_HIP: 0,   Joint.FR_THIGH: 1,  Joint.FR_CALF: 2,
    Joint.FL_HIP: 3,    Joint.FL_THIGH: 4,   Joint.FL_CALF: 5,
    Joint.RR_HIP: 6,    Joint.RR_THIGH: 7,   Joint.RR_CALF: 8,
    Joint.RL_HIP: 9,     Joint.RL_THIGH: 10,   Joint.RL_CALF: 11,
}

SENSOR_VEL_DICT = {
    Joint.FR_HIP: 12,  Joint.FR_THIGH: 13, Joint.FR_CALF: 14,
    Joint.FL_HIP: 15,   Joint.FL_THIGH: 16,  Joint.FL_CALF: 17,
    Joint.RR_HIP: 18,   Joint.RR_THIGH: 19,  Joint.RR_CALF: 20,
    Joint.RL_HIP: 21,    Joint.RL_THIGH: 22,   Joint.RL_CALF: 23,
}

OSCILLATOR_TO_THIGH = {
    0: Joint.FL_THIGH,
    1: Joint.FR_THIGH,
    2: Joint.RR_THIGH,
    3: Joint.RL_THIGH,
}

# TODO: Test and tune these values for energy efficiency
# ENERGY_EFFICIENT_FREQ = {
#     Gait.WALK: 0.5,
#     Gait.TROT: 0.7,
#     Gait.BOUND: 1.0,
#     Gait.PACE: 0.8,
#     Gait.GALLOP: 1.2,
#     Gait.CANTER: 1.0,
# }

NEURON_POSITION = {
    0: ''
}


LEG_LABELS = ['FL', 'FR', 'RR', 'RL']

# Go2 actuator torque limit (Nm) — from MuJoCo model ctrlrange
ACTUATOR_TORQUE_LIMIT = 23.7
