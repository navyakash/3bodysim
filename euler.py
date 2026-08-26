"""
Three-body gravity simulator - Euler method (Layer 3a).

This is the simplest possible integrator, on purpose. Run it, watch it work,
then watch it slowly drift. That drift is what motivates RK4 next.

Requires: numpy, matplotlib   ->   pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# We set G = 1 by choosing convenient units. This is standard in N-body work:
# it keeps the numbers sane and changes nothing about the physics.
G = 1.0

# A tiny "softening" length. When two bodies pass very close, dist**3 in the
# denominator blows up and the sim explodes. Softening quietly caps that.
# It very slightly alters the physics; set it to 0.0 to see the pure version.
EPS = 1e-3

# ---- Initial conditions -----------------------------------------------------
# masses of the three bodies
m = np.array([2.0, 1.0, 1.0])

# positions (x, y) of each body  ->  shape (3, 2)
pos = np.array([
    [ 0.0,  0.0],   # body 1 (the heavy one, near the middle)
    [ 1.5,  0.0],   # body 2
    [-1.5,  0.0],   # body 3
])

# velocities (vx, vy) of each body  ->  shape (3, 2)
vel = np.array([
    [ 0.0,  0.0],   # body 1 starts at rest
    [ 0.0,  0.81],   # body 2 moving "up"
    [ 0.0, -0.8],   # body 3 moving "down"
])


def accelerations(pos, m):
    """The one piece of real physics: acceleration on each body from gravity.

    This is f(Y) for the velocity part - given where everyone is, how is each
    body's velocity changing right now?
    """
    n = len(m)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pos[j] - pos[i]                 # arrow from body i to body j
            dist = np.sqrt(diff @ diff + EPS**2)   # softened distance
            acc[i] += G * m[j] * diff / dist**3    # Newton's law, vector form
    return acc


# ---- Euler integration loop -------------------------------------------------
dt = 0.001        # time step (smaller = more accurate, but more steps)
steps = 20000     # total steps -> simulates t = steps * dt = 20 time units

# somewhere to record every body's position at every step, for plotting
trajectory = np.zeros((steps, 3, 2))

for s in range(steps):
    trajectory[s] = pos                # record current positions
    acc = accelerations(pos, m)        # evaluate the physics at the current state
    pos = pos + vel * dt               # r_next = r + v*dt
    vel = vel + acc * dt               # v_next = v + a*dt


# ---- Plot the trajectories --------------------------------------------------
plt.figure(figsize=(7, 7))
colors = ["#185FA5", "#BA7517", "#D85A30"]
for i in range(3):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1],
             color=colors[i], lw=0.8, label=f"Body {i + 1}")
    plt.plot(trajectory[0, i, 0], trajectory[0, i, 1],
             "o", color=colors[i])   # mark each starting point
plt.axis("equal")
plt.legend()
plt.title("Three-body simulation (Euler method)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()