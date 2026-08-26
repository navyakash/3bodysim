"""
Three-body gravity simulator - RK4 method (Layer 3b).

RK4 samples the slope at four points across each step and averages them,
turning Euler's dt-sized error into a dt^4-sized error. Same physics as before,
far more accurate: the orbits stop drifting and total energy stays nearly flat.

Notice the structure. The simulation is now a function, simulate(...), that takes
ALL of its inputs as arguments. That is deliberate. Today those arguments come
from a dict or from typed prompts; later your web UI will hand this same function
the exact same arguments. The engine never needs to know where the numbers came
from - that separation is what makes the eventual UI easy.

Requires: numpy, matplotlib   ->   pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

G = 1.0
EPS = 1e-3   # softening length; set to 0.0 for pure Newtonian gravity


# ---- Physics: the one real function, f(Y) -----------------------------------
def accelerations(pos, m):
    """Acceleration on each body from gravity. Works for any number of bodies."""
    n = len(m)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            dist = np.sqrt(diff @ diff + EPS**2)
            acc[i] += G * m[j] * diff / dist**3
    return acc


def deriv(pos, vel, m):
    """f(Y): rate of change of the state = (d pos/dt, d vel/dt) = (vel, acc)."""
    return vel, accelerations(pos, m)


# ---- Numerical method: one RK4 step -----------------------------------------
def rk4_step(pos, vel, m, dt):
    k1p, k1v = deriv(pos, vel, m)                                   # slope at start
    k2p, k2v = deriv(pos + 0.5*dt*k1p, vel + 0.5*dt*k1v, m)         # midpoint, est 1
    k3p, k3v = deriv(pos + 0.5*dt*k2p, vel + 0.5*dt*k2v, m)         # midpoint, est 2
    k4p, k4v = deriv(pos + dt*k3p,     vel + dt*k3v,     m)         # slope at end
    pos_next = pos + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    vel_next = vel + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return pos_next, vel_next


# ---- Diagnostic: total energy (real physics keeps this constant) ------------
def total_energy(pos, vel, m):
    kinetic = 0.5 * np.sum(m * np.sum(vel**2, axis=1))
    potential = 0.0
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(np.sum((pos[j] - pos[i])**2) + EPS**2)
            potential -= G * m[i] * m[j] / r
    return kinetic + potential


# ---- The engine: pure function, inputs in -> trajectory out -----------------
def simulate(masses, positions, velocities, dt, steps):
    m = np.array(masses, dtype=float)
    pos = np.array(positions, dtype=float)
    vel = np.array(velocities, dtype=float)
    n = len(m)

    trajectory = np.zeros((steps, n, 2))
    energy = np.zeros(steps)

    for s in range(steps):
        trajectory[s] = pos
        energy[s] = total_energy(pos, vel, m)
        pos, vel = rk4_step(pos, vel, m, dt)

    return trajectory, energy


# ---- Input layer (today: a dict, or typed prompts; tomorrow: a web form) -----
def default_scenario():
    return {
        "masses":     [2.0, 1.0, 1.0],
        "positions":  [[0.0, 0.0], [1.5, 0.0], [-1.5, 0.0]],
        "velocities": [[0.0, 0.0], [0.0, 0.8], [0.0, -0.8]],
        "dt": 0.001,
        "steps": 20000,
    }


def ask_for_scenario():
    """Placeholder for the eventual UI: gather the same values from the user."""
    n = int(input("How many bodies? "))
    masses, positions, velocities = [], [], []
    for i in range(n):
        print(f"\nBody {i + 1}:")
        masses.append(float(input("  mass: ")))
        x  = float(input("  start x: "));    y  = float(input("  start y: "))
        vx = float(input("  velocity x: ")); vy = float(input("  velocity y: "))
        positions.append([x, y])
        velocities.append([vx, vy])
    dt = float(input("\ntime step dt (e.g. 0.001): "))
    steps = int(input("number of steps (e.g. 20000): "))
    return {"masses": masses, "positions": positions,
            "velocities": velocities, "dt": dt, "steps": steps}


# ---- Run --------------------------------------------------------------------
if __name__ == "__main__":
    # To type your own values, swap the next line for: cfg = ask_for_scenario()
    cfg = default_scenario()

    traj, energy = simulate(cfg["masses"], cfg["positions"],
                            cfg["velocities"], cfg["dt"], cfg["steps"])

    drift = abs((energy[-1] - energy[0]) / energy[0]) * 100
    print(f"Relative energy drift over the run: {drift:.5f}%")

    n = len(cfg["masses"])
    colors = ["#185FA5", "#BA7517", "#D85A30", "#1D9E75", "#7F77DD"]
    plt.figure(figsize=(7, 7))
    for i in range(n):
        c = colors[i % len(colors)]
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=c, lw=0.8, label=f"Body {i + 1}")
        plt.plot(traj[0, i, 0], traj[0, i, 1], "o", color=c)
    plt.axis("equal")
    plt.legend()
    plt.title("Three-body simulation (RK4 method)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()