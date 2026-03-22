import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from shared_module.robot_state import Gait

class FuzzyGaitSwitch:
    def __init__(self):
        # Define fuzzy variables
        self.speed = ctrl.Antecedent(np.arange(0, 5, 1), 'speed')  # in Hertz
        self.gait = ctrl.Consequent(np.arange(0, 3, 1), 'gait')

        # Define membership functions for speed
        self.speed['slow'] = fuzz.trimf(self.speed.universe, [0, 0, 5])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [0, 5, 10])
        self.speed['fast'] = fuzz.trimf(self.speed.universe, [5, 10, 10])

        # Define membership functions for gait
        self.gait[Gait.WALK.name] = fuzz.trimf(self.gait.universe, [0, 0, 1])
        self.gait[Gait.TROT.name] = fuzz.trimf(self.gait.universe, [0, 1, 2])
        self.gait[Gait.GALLOP.name] = fuzz.trimf(self.gait.universe, [1, 2, 2])

        rule1 = ctrl.Rule(self.speed['slow'], self.gait[Gait.WALK.name])
        rule2 = ctrl.Rule(self.speed['medium'], self.gait[Gait.TROT.name])
        rule3 = ctrl.Rule(self.speed['fast'], self.gait[Gait.GALLOP.name])
        
        # Create control system and simulation
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3])
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
