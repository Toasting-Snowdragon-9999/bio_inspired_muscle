import array
import os, sys
import numpy as np
import mujoco as mj
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_module.global_constants import *

class BalanceController:
    """
    Balance controller to control the hip abductor and adductor
    Should control them such that they are always normal to the ground
    """

    def __init__():
        pass
