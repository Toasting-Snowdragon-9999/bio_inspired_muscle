from configparser import ConfigParser
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from shared_module.robot_state import Gait
from cpg.trajectory_builder import EllipsoidConfig


BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SETTINGS_PATH = os.path.join(
    BASE_PATH,
    "data",
    ".settings"
)


def load_settings_from_file(gait: Gait):

    parser = ConfigParser()

    gait_path = os.path.join(
        SETTINGS_PATH,
        f".{gait.name.upper()}.ini"
    )

    if not os.path.exists(gait_path):
        raise FileNotFoundError(
            f"Settings file for gait {gait.name} not found at {gait_path}"
        )

    parser.read(gait_path)

    section = "gait"

    duty_factor = parser.getfloat(section, "duty_factor")
    frequency = parser.getfloat(section, "frequency")

    config = EllipsoidConfig(

        # ── Front legs ─────────────────────────────
        front_x_fore=parser.getfloat(section, "front_x_fore"),
        front_x_hind=parser.getfloat(section, "front_x_hind"),
        front_z_top=parser.getfloat(section, "front_z_top"),
        front_z_bottom=parser.getfloat(section, "front_z_bottom"),
        front_rotation=parser.getfloat(section, "front_rotation"),
        front_skew=parser.getfloat(section, "front_skew"),

        # ── Rear legs ──────────────────────────────
        rear_x_fore=parser.getfloat(section, "rear_x_fore"),
        rear_x_hind=parser.getfloat(section, "rear_x_hind"),
        rear_z_top=parser.getfloat(section, "rear_z_top"),
        rear_z_bottom=parser.getfloat(section, "rear_z_bottom"),
        rear_rotation=parser.getfloat(section, "rear_rotation"),
        rear_skew=parser.getfloat(section, "rear_skew"),
    )

    return config, frequency, duty_factor