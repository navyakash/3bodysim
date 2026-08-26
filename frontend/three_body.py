"""
Three-body (n-body) gravity engine.

UI-agnostic. Euler and RK4 integrators; the force calculation is vectorized with
NumPy so running several simulations per click stays fast.

simulate() now records BOTH positions and velocities over time, so any conserved
quantity (energy, linear momentum, angular momentum) can be reconstructed after
the fact by the *_series helpers below.
"""

import numpy as np

G = 1.0   # gravitational constant, in convenient units


def accelerations(pos, m, eps=1e-3):
    """Acceleration on every body at once, via broadcasting. Works for any n.

    diff[i, j] is the vector from body i to body j; Newton's law gives
    a[i] = sum_j  G * m[j] * diff[i, j] / |diff[i, j]|**3.
    """
    diff = pos[None, :, :] - pos[:, None, :]        # (n, n, 2)
    dist2 = np.sum(diff**2, axis=2) + eps**2        # (n, n), softened
    inv = dist2 ** -1.5                             # 1 / distance**3
    np.fill_diagonal(inv, 0.0)                      # a body exerts no force on itself
    return G * np.sum(inv[:, :, None] * m[None, :, None] * diff, axis=1)


def deriv(pos, vel, m, eps):
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


def simulate(masses, positions, velocities, dt, steps, eps=1e-3, method="rk4"):
    """Integrate the system, recording positions AND velocities each step.

    Returns
    -------
    pos_traj : ndarray, shape (steps, n_bodies, 2)
    vel_traj : ndarray, shape (steps, n_bodies, 2)
    """
    step = STEPPERS[method]
    m = np.asarray(masses, dtype=float)
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    n = len(m)

    pos_traj = np.zeros((steps, n, 2))
    vel_traj = np.zeros((steps, n, 2))
    for s in range(steps):
        pos_traj[s] = pos
        vel_traj[s] = vel
        pos, vel = step(pos, vel, m, dt, eps)

    return pos_traj, vel_traj


# ---- Conserved-quantity time series -----------------------------------------
def energy_series(pos_traj, vel_traj, masses, eps=1e-3):
    """Total energy (kinetic + gravitational potential) at every time step."""
    m = np.asarray(masses, dtype=float)
    n = len(m)
    kinetic = 0.5 * np.sum(m[None, :] * np.sum(vel_traj**2, axis=2), axis=1)
    dr = pos_traj[:, :, None, :] - pos_traj[:, None, :, :]     # (T, n, n, 2)
    dist = np.sqrt(np.sum(dr**2, axis=3) + eps**2)            # (T, n, n)
    iu = np.triu_indices(n, 1)
    mm = np.outer(m, m)[iu]                                    # mass products, upper pairs
    potential = -G * np.sum(mm[None, :] / dist[:, iu[0], iu[1]], axis=1)
    return kinetic + potential


def momentum_series(vel_traj, masses):
    """Total linear momentum vector at every time step. Shape (steps, 2)."""
    m = np.asarray(masses, dtype=float)
    return np.sum(m[None, :, None] * vel_traj, axis=1)


def angular_momentum_series(pos_traj, vel_traj, masses):
    """Total angular momentum (z-component in 2D) at every time step. Shape (steps,)."""
    m = np.asarray(masses, dtype=float)
    cross = pos_traj[:, :, 0] * vel_traj[:, :, 1] - pos_traj[:, :, 1] * vel_traj[:, :, 0]
    return np.sum(m[None, :] * cross, axis=1)


# ---- Chaos analysis ---------------------------------------------------------
def separation(traj_a, traj_b):
    """Distance between two whole configurations at each time step."""
    diff = np.asarray(traj_a) - np.asarray(traj_b)
    return np.sqrt(np.sum(diff**2, axis=(1, 2)))


def estimate_lyapunov(d, dt):
    """Rough largest-Lyapunov estimate: slope of ln(separation) vs time over the
    exponential-growth window. Returns the slope, or None if no clean window."""
    d = np.asarray(d, dtype=float)
    positive = d[d > 0]
    if positive.size == 0:
        return None
    d0, dmax = positive[0], d.max()
    if dmax <= 0:
        return None
    lo, hi = 10 * d0, 0.1 * dmax
    mask = (d > lo) & (d < hi)
    if mask.sum() < 10:
        return None
    idx = np.where(mask)[0]
    return float(np.polyfit(idx * dt, np.log(d[idx]), 1)[0])