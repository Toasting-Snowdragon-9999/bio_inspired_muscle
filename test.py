import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Simple Kuramoto oscillator demonstration
# -------------------------------------------------

N = 4                  # Number of oscillators
dt = 0.001
t_max = 10.0
steps = int(t_max / dt)

# Natural frequencies
omega = 1 * np.pi * np.array([1.0, 1.0, 1.0, 1.0])

# Coupling matrix
K = 1.5
a = np.ones((N, N)) * K
np.fill_diagonal(a, 0.0)

# Initial phases
theta = np.array([
    0.0,
    np.pi,
    0.2 * np.pi,
    1.2 * np.pi
])

# Store outputs
theta_history = np.zeros((steps, N))
time = np.zeros(steps)

# -------------------------------------------------
# Kuramoto derivatives
# -------------------------------------------------

def derivatives(theta):
    dtheta = np.zeros(N)

    for i in range(N):
        coupling = 0.0

        for j in range(N):
            if i != j:
                coupling += a[i, j] * np.sin(theta[j] - theta[i])

        dtheta[i] = omega[i] + coupling

    return dtheta

def get_oscillator_outputs(phases) -> tuple[np.ndarray, np.ndarray]:
    """
    Return normalised output signal per leg for oscillator graph overlay.
    Returns:
        leg_outputs:  shape (4,)  — -sin(θ_i)
        knee_outputs: shape (4,)  — clamped sin(θ_i + knee_offset)
    """
    outputs = []
    for i in range(4):
        phase = phases[i]
        outputs.append(max(np.cos(phase), 0.0)) # Clamp to [0, 1]

    return outputs

# -------------------------------------------------
# RK4 Integration
# -------------------------------------------------

for step in range(steps):

    k1 = derivatives(theta)
    k2 = derivatives(theta + 0.5 * dt * k1)
    k3 = derivatives(theta + 0.5 * dt * k2)
    k4 = derivatives(theta + dt * k3)

    theta += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Wrap phase to [0, 2π]
    outputs = get_oscillator_outputs(theta)
    theta_history[step] = outputs
    time[step] = step * dt

# -------------------------------------------------
# Plot phases
# -------------------------------------------------

base_size = 18
plt.rcParams.update({
    'font.size': base_size,        # Default text size
    'axes.titlesize': base_size + 5,   # Title size
    'axes.labelsize': base_size + 4,   # X and Y label size
    'xtick.labelsize': base_size - 2,  # X tick size
    'ytick.labelsize': base_size - 2,  # Y tick size
    'legend.fontsize': base_size + 2   # Legend size
})

plt.figure(figsize=(10, 5))

for i in range(N):
    plt.plot(time, theta_history[:, i], label=f"Oscillator {i}")

plt.xlabel("Time (seconds)")
plt.ylabel(r'$\cos(\theta)$')
plt.title("Kuramoto Oscillator Phase Evolution")
plt.legend(loc='center right')
plt.tight_layout()
plt.grid(True)

plt.show()