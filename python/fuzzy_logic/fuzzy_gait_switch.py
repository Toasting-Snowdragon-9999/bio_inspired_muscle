import os,sys
from typing import Any
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_module.robot_state import Gait

class FuzzyGaitSwitch:
    def __init__(self):
        # Define fuzzy variables
        # self.vel_cmd = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'velocity')
        self.stability = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'stability')
        self.speed_error = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'speed_error')

        # self.frequency = ctrl.Consequent(np.arange(0, 4.0, 0.01), 'frequency')
        self.blend_rate = ctrl.Consequent(np.arange(-1.0, 1.01, 0.01), 'blend_rate')
        self.blending_factor = 0.0  # Initialize blending factor

        # Define membership functions
        # self.vel_cmd['slow']   = fuzz.trapmf(self.vel_cmd.universe, [0.0, 0.0, 0.2, 0.4])
        # self.vel_cmd['medium'] = fuzz.trimf(self.vel_cmd.universe, [0.25, 0.5, 0.75])
        # self.vel_cmd['fast']   = fuzz.trapmf(self.vel_cmd.universe, [0.6, 0.8, 1.0, 1.0])

        # self.frequency['low']    = fuzz.trapmf(self.frequency.universe, [0.0, 0.0, 1.0, 2.0])
        # self.frequency['medium'] = fuzz.trimf(self.frequency.universe, [1.8, 2.5, 3.0])
        # self.frequency['high']   = fuzz.trapmf(self.frequency.universe, [2.8, 3.5, 4.0, 4.0])

        self.stability['very_unstable'] = fuzz.trapmf(self.stability.universe, [0.0, 0.0, 0.2, 0.4])
        self.stability['unstable']      = fuzz.trimf(self.stability.universe, [0.3, 0.5, 0.7])
        self.stability['stable']        = fuzz.trimf(self.stability.universe, [0.6, 0.75, 0.9])
        self.stability['very_stable']   = fuzz.trapmf(self.stability.universe, [0.8, 0.9, 1.0, 1.0])

        self.speed_error['low']    = fuzz.trapmf(self.speed_error.universe, [0.0, 0.0, 0.2, 0.4])
        self.speed_error['medium'] = fuzz.trimf(self.speed_error.universe, [0.3, 0.5, 0.7])
        self.speed_error['high']   = fuzz.trapmf(self.speed_error.universe, [0.6, 0.8, 1.0, 1.0])

        self.blend_rate['backward_fast'] = fuzz.trapmf(self.blend_rate.universe, [-1.00, -1.00, -0.70, -0.45])
        self.blend_rate['backward_slow'] = fuzz.trimf(self.blend_rate.universe, [-0.65, -0.32, -0.05])
        self.blend_rate['hold'] = fuzz.trimf(self.blend_rate.universe, [-0.10, 0.00, 0.10])
        self.blend_rate['forward_slow'] = fuzz.trimf(self.blend_rate.universe, [0.05, 0.32, 0.65])
        self.blend_rate['forward_fast'] = fuzz.trapmf(self.blend_rate.universe, [0.45, 0.70, 1.00, 1.00])

        # Define the fuzzy rules
        self.high_stability_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['very_stable'], self.blend_rate['hold'])
        self.high_stability_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['very_stable'], self.blend_rate['forward_slow'])
        self.high_stability_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['very_stable'], self.blend_rate['forward_fast'])

        self.stability_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['stable'], self.blend_rate['hold'])
        self.stability_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['stable'], self.blend_rate['forward_slow'])
        self.stability_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['stable'], self.blend_rate['forward_slow'])

        self.unstable_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['unstable'], self.blend_rate['backward_slow'])
        self.unstable_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['unstable'], self.blend_rate['backward_slow'])
        self.unstable_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['unstable'], self.blend_rate['backward_fast'])

        self.very_unstable_rule1 = ctrl.Rule(self.speed_error['low'] & self.stability['very_unstable'], self.blend_rate['backward_slow'])
        self.very_unstable_rule2 = ctrl.Rule(self.speed_error['medium'] & self.stability['very_unstable'], self.blend_rate['backward_fast'])
        self.very_unstable_rule3 = ctrl.Rule(self.speed_error['high'] & self.stability['very_unstable'], self.blend_rate['backward_fast'])
        
        # Define the system
        fuzzy_ctrl   = ctrl.ControlSystem([self.vel_stability_rule]) # Create a control system
        fuzzy_sim    = ctrl.ControlSystemSimulation(fuzzy_ctrl) # Create a simulation for your control system

    def show_membership_functions(self):
        self.vel_cmd.view()
        self.stability.view()
        self.speed_error.view()
        self.frequency.view()
        self.blend_rate.view()
        plt.show()

    def blending_factor(self, blend_rate_output: float, dt: float) -> float:
        """Update blending factor based on blend_rate output."""
        self.blending_factor = np.clip(self.blending_factor + blend_rate_output * dt, 0.0, 1.0)
        return self.blending_factor

    def interpolate(self, a: float, b: float, s: float) -> float:
        """Linear interpolation between a and b with blending factor s."""
        return (1 - s) * a + s * b

    def blend_gaits(self, old_gait, new_gait, blending_factor) -> Any:
        pass

    def blend_trajectories(self, old_traj, new_traj, blending_factor) -> Any:
        pass

    def update(self, ) -> Gait:
        pass

    def gait_picker(self, velocity: float) -> Gait:
        pass

        # self.fuzzy_sim.input['velocity'] = self.robot_interface.body_velocity
        # self.fuzzy_sim.input['stability'] = self.robot_interface.stability_metric

        # self.fuzzy_sim.compute()

        # # Read outputs
        # freq = self.fuzzy_sim.output['frequency']
        # duty = self.fuzzy_sim.output['duty_factor']

        # # Apply (optionally smooth)
        # self.robot_interface.frequency = freq
        # self.robot_interface.duty_factor = duty
    
if __name__ == "__main__":
    fuzzy_switch = FuzzyGaitSwitch()
    fuzzy_switch.show_membership_functions()
