"""
Three-body (n-body) gravity engine.

UI-agnostic. Supports Euler and RK4 integrators (via `method`), energy
diagnostics, and the two analysis helpers the chaos demo needs:
separation() and estimate_lyapunov().

The force calculation is vectorized with NumPy broadcasting so that running
several simulations per click (needed for the chaos comparison) stays fast.
"""

import numpy as np

G = 1.0   # gravitational constant, in convenient units


def accelerations(pos, m, eps=1e-3):
    """Acceleration on every body at once, via broadcasting. Works for any n.

    diff[i, j] is the vector from body i to body j. Newton's law says
    a[i] = sum_j  G * m[j] * diff[i, j] / |diff[i, j]|**3.
    """
    diff = pos[None, :, :] - pos[:, None, :]        # (n, n, 2)
    dist2 = np.sum(diff**2, axis=2) + eps**2        # (n, n), softened
    inv = dist2 ** -1.5                             # = 1 / distance**3
    np.fill_diagonal(inv, 0.0)                      # a body exerts no force on itself
    return G * np.sum(inv[:, :, None] * m[None, :, None] * diff, axis=1)


def deriv(pos, vel, m, eps):
    """f(Y): (d pos/dt, d vel/dt) = (vel, acc)."""
    return vel, accelerations(pos, m, eps)


def euler_step(pos, vel, m, dt, eps):
    """One explicit Euler step: first-order, cheap, leaks energy over time."""
    acc = accelerations(pos, m, eps)
    return pos + vel * dt, vel + acc * dt


def rk4_step(pos, vel, m, dt, eps):
    """One RK4 step: samples the slope at four points and averages them."""
    k1p, k1v = deriv(pos, vel, m, eps)
    k2p, k2v = deriv(pos + 0.5*dt*k1p, vel + 0.5*dt*k1v, m, eps)
    k3p, k3v = deriv(pos + 0.5*dt*k2p, vel + 0.5*dt*k2v, m, eps)
    k4p, k4v = deriv(pos + dt*k3p,     vel + dt*k3v,     m, eps)
    pos_next = pos + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    vel_next = vel + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return pos_next, vel_next


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def total_energy(pos, vel, m, eps=1e-3):
    """Kinetic + gravitational potential. Real physics keeps this constant."""
    kinetic = 0.5 * np.sum(m * np.sum(vel**2, axis=1))
    potential = 0.0
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(np.sum((pos[j] - pos[i])**2) + eps**2)
            potential -= G * m[i] * m[j] / r
    return kinetic + potential


def simulate(masses, positions, velocities, dt, steps, eps=1e-3, method="rk4"):
    """Run the simulation with the chosen integrator.

    Returns
    -------
    trajectory : ndarray, shape (steps, n_bodies, 2)
    energy     : ndarray, shape (steps,)
    """
    step = STEPPERS[method]
    m = np.asarray(masses, dtype=float)
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    n = len(m)

    trajectory = np.zeros((steps, n, 2))
    energy = np.zeros(steps)

    for s in range(steps):
        trajectory[s] = pos
        energy[s] = total_energy(pos, vel, m, eps)
        pos, vel = step(pos, vel, m, dt, eps)

    return trajectory, energy


# ---- Chaos analysis ---------------------------------------------------------
def separation(traj_a, traj_b):
    """Distance between two whole configurations at each time step.

    A single number per step: how far apart the two systems have grown.
    """
    diff = np.asarray(traj_a) - np.asarray(traj_b)
    return np.sqrt(np.sum(diff**2, axis=(1, 2)))


def estimate_lyapunov(d, dt):
    """Rough largest-Lyapunov-exponent estimate: the slope of ln(separation)
    vs time, fitted over the exponential-growth window (before it saturates).

    Returns the slope (per time unit), or None if there isn't a clean window.
    """
    d = np.asarray(d, dtype=float)
    positive = d[d > 0]
    if positive.size == 0:
        return None
    d0, dmax = positive[0], d.max()
    if dmax <= 0:
        return None
    lo, hi = 10 * d0, 0.1 * dmax          # skip the noisy start and the saturated end
    mask = (d > lo) & (d < hi)
    if mask.sum() < 10:
        return None
    idx = np.where(mask)[0]
    slope = np.polyfit(idx * dt, np.log(d[idx]), 1)[0]
    return float(slope)