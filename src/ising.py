import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
N = 200  # Grid size
J = 1e2  # Interaction strength
T = 1  # Temperature
num_steps = 10000  # Number of Metropolis steps

# Initialize spins in the range [-pi/2, pi/2]
spins = np.random.uniform(-np.pi / 2, np.pi / 2, (N, N))

def energy_change(spins, i, j, new_theta):
    """
    Compute energy change if spin (i, j) is changed to new_theta.
    """
    neighbors = [
        (i-1, j), (i+1, j), (i, j-1), (i, j+1),  # Original 4 neighbors
    ]
    energy_old = 0
    energy_new = 0
    
    for ni, nj in neighbors:
        ni %= N  # Apply periodic boundary conditions
        nj %= N  # Apply periodic boundary conditions
        
        energy_old += -J * np.cos(spins[i, j] - spins[ni, nj])
        energy_new += -J * np.cos(new_theta - spins[ni, nj])
    
    return energy_new - energy_old

# Metropolis Algorithm
for step in range(num_steps):
    i, j = random.randint(0, N-1), random.randint(0, N-1)  # Pick a random spin
    new_theta = spins[i, j] + np.random.uniform(-0.1, 0.1)  # Small random change
    new_theta = ((new_theta + np.pi/2) % np.pi) - np.pi/2  # Ensure wrapping
    
    delta_E = energy_change(spins, i, j, new_theta)
    
    if delta_E < 0 or np.exp(-delta_E / T) > np.random.rand():
        spins[i, j] = new_theta  # Accept new state

# Plot final stable state
plt.figure(figsize=(8, 6))
plt.imshow(spins, cmap='hsv', interpolation='nearest')
plt.colorbar(label='Spin Angle (radians)')
plt.title('Stable State of the 2D Ising Model with Continuous Spins')
plt.show()

