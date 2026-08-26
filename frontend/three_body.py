"""
Three-body (n-body) gravity engine.

UI-agnostic on purpose. Now supports two integrators - Euler and RK4 - selected
with the `method` argument, so the interface can run the same system both ways
and compare their numerical behaviour.
"""

import numpy as np

G = 1.0   # gravitational constant, in convenient units


def accelerations(pos, m, eps=1e-3):
    """Acceleration on each body from Newtonian gravity. Works for any n."""
    n = len(m)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            dist = np.sqrt(diff @ diff + eps**2)
            acc[i] += G * m[j] * diff / dist**3
    return acc


def deriv(pos, vel, m, eps):
    """f(Y): (d pos/dt, d vel/dt) = (vel, acc)."""
    return vel, accelerations(pos, m, eps)


def euler_step(pos, vel, m, dt, eps):
    """One explicit Euler step: first-order, cheap, and leaks energy over time."""
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