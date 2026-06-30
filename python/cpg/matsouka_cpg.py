"""@brief Plain NumPy Matsuoka half-center oscillator CPG model (mutually inhibiting neurons)."""

from dataclasses import dataclass
import numpy as np

@dataclass
class Neuron:
    """@brief State and parameters of a single Matsuoka neuron (internal state, adaptation/fatigue, output)."""
    tau: float  # Time constant for the internal state
    T: float    # Time constant for adaptation
    b: float    # adaptation strength
    s: float    # tonic drive

    x: float = 0.0          # Internal state
    dx: float = 0.0         # Derivative of internal state

    x_adapt: float = 0.0    # Adaption state (fatigue)
    dx_adapt: float = 0.0   # Derivative of adaption state
    
    y: float = 0.0          # Output of the neuron (after nonlinearity)

class MatsuokaCPG:
    """@brief Four-neuron Matsuoka CPG with a mutual-inhibition matrix that shapes the quadruped gait pattern."""

    def __init__(self):
        """@brief Construct the CPG: four Matsuoka neurons, feed inputs, and the inhibitory coupling matrix."""
        self.neurons_cnt = 4
        self.feed = [0.0, 0.0, 0.02, 0.0] 
        self.neurons = [Neuron(tau=0.05, T=0.5, b=2.5, s=1.0) for _ in range(self.neurons_cnt)]
        
        self.inhibitory_connection = np.array([
            [0.0, 1.5, 2.0, 1.0],
            [1.5, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0]
        ])

    def update_inhibition(self):
        """@brief Recompute each neuron's total inhibitory input I_inh from the others' outputs."""
        # Matrix multiply: I_inh[i] = Σ_j a[i,j] * y[j]
        self.I_inh = np.dot(self.inhibitory_connection, [neuron.y for neuron in self.neurons])