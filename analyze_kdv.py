#!/usr/bin/env python3
"""Simple test to understand the KdV equation better."""

import numpy as np
import matplotlib.pyplot as plt

# Define the analytical soliton solution
def analytical_soliton(x, t, kappa):
    """Analytical single soliton solution for KdV."""
    c = 4 * kappa**2  # soliton speed
    x_center = c * t
    x_shifted = x - x_center
    return 2 * kappa**2 / np.cosh(kappa * x_shifted)**2

# Parameters
kappa = 0.5
L = 100.0
N = 256
dx = L / N
x = np.arange(N) * dx

# Test the analytical solution at t=0
u0 = analytical_soliton(x, 0.0, kappa)
print(f"Max amplitude at t=0: {np.max(u0)}")
print(f"Expected amplitude: {2 * kappa**2}")

# Test at t=0.5
u05 = analytical_soliton(x, 0.5, kappa)
print(f"Max amplitude at t=0.5: {np.max(u05)}")
print(f"Expected amplitude: {2 * kappa**2}")

# Plot to visualize
plt.figure(figsize=(10, 6))
plt.plot(x, u0, label='t=0')
plt.plot(x, u05, label='t=0.5')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Analytical KdV Soliton')
plt.legend()
plt.grid(True)
plt.savefig('/tmp/analytics_soliton.png')
print("Saved plot to /tmp/analytics_soliton.png")

# Now test the derivative
u_x = np.gradient(u0, dx)
print(f"Max |u_x|: {np.max(np.abs(u_x))}")