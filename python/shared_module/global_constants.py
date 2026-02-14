import numpy as np

# ── Joint ranges (radians) ──────────────────────────────────────
KNEE_POS_RANGE      = (-2.7227, -0.83776)
FRONT_HIP_POS_RANGE = (-1.5708, 3.4907)
BACK_HIP_POS_RANGE  = (-0.5236, 4.5379)
ABDUCTION_POS_RANGE = (-1.0472, 1.0472)

# ── Actuator / sensor index maps ────────────────────────────────
ACTUATOR_DICT = {
    'front_right_hip': 0,   'front_right_thigh': 1,  'front_right_calf': 2,
    'front_left_hip': 3,    'front_left_thigh': 4,   'front_left_calf': 5,
    'rear_right_hip': 6,    'rear_right_thigh': 7,   'rear_right_calf': 8,
    'rear_left_hip': 9,     'rear_left_thigh': 10,   'rear_left_calf': 11,
}

SENSOR_POS_DICT = {
    'front_right_hip': 0,   'front_right_thigh': 1,  'front_right_calf': 2,
    'front_left_hip': 3,    'front_left_thigh': 4,   'front_left_calf': 5,
    'rear_right_hip': 6,    'rear_right_thigh': 7,   'rear_right_calf': 8,
    'rear_left_hip': 9,     'rear_left_thigh': 10,   'rear_left_calf': 11,
}

SENSOR_VEL_DICT = {
    'front_right_hip': 12,  'front_right_thigh': 13, 'front_right_calf': 14,
    'front_left_hip': 15,   'front_left_thigh': 16,  'front_left_calf': 17,
    'rear_right_hip': 18,   'rear_right_thigh': 19,  'rear_right_calf': 20,
    'rear_left_hip': 21,    'rear_left_thigh': 22,   'rear_left_calf': 23,
}

OSCILLATOR_TO_THIGH = {
    0: 'front_left_thigh',
    1: 'front_right_thigh',
    2: 'rear_right_thigh',
    3: 'rear_left_thigh',
}

OSCILLATOR_TO_CALF = {
    0: 'front_left_calf',
    1: 'front_right_calf',
    2: 'rear_right_calf',
    3: 'rear_left_calf',
}

GAITS = {
    'trot':   np.array([0.0,       np.pi,     0.0,       np.pi]),
    'walk':   np.array([0.0,       np.pi/2,   np.pi,     3*np.pi/2]),
    'bound':  np.array([0.0,       0.0,       np.pi,     np.pi]),
    'pace':   np.array([0.0,       np.pi,     np.pi,     0.0]),
    'gallop': np.array([0.0,       0.0,       np.pi*0.8, np.pi*0.8]),
} 

LEG_LABELS = ['FL', 'FR', 'RR', 'RL']
