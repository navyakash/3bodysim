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


# ---- Event detection (analysis of a single completed run) -------------------
def detect_events(pos_traj, vel_traj, masses, dt, eps=1e-3,
                  r_collide=0.15, r_escape=None, slingshot_gain=1.4):
    """Classify what physically happened during one run. Heuristic but tunable.

    Returns a dict describing collision, escape, slingshot and capture.
    """
    m = np.asarray(masses, float)
    n = len(m); M = m.sum()
    T = pos_traj.shape[0]
    if r_escape is None:
        r_escape = 8.0 * np.sqrt(np.mean(np.sum((pos_traj[0] - pos_traj[0].mean(0))**2, axis=1)) + 1e-9)
        r_escape = max(r_escape, 8.0)

    # pairwise distances over time
    diff = pos_traj[:, :, None, :] - pos_traj[:, None, :, :]      # (T,n,n,2)
    dist = np.sqrt(np.sum(diff**2, axis=3) + eps**2)             # (T,n,n)
    iu = np.triu_indices(n, 1)
    pair_d = dist[:, iu[0], iu[1]]                               # (T, pairs)

    min_pair = pair_d.min()
    k_min = np.unravel_index(pair_d.argmin(), pair_d.shape)      # (time, pair)
    pair_ij = (iu[0][k_min[1]], iu[1][k_min[1]])

    collision = bool(min_pair < r_collide)

    # distance from centre of mass
    com = (m[None, :, None] * pos_traj).sum(axis=1) / M          # (T,2)
    rcom = np.sqrt(((pos_traj - com[:, None, :])**2).sum(axis=2))  # (T,n)
    max_com_body = rcom.max(axis=0)
    escaper = int(max_com_body.argmax())
    escape = bool(max_com_body[escaper] > r_escape)
    escape_time = float(np.argmax(rcom[:, escaper] > r_escape) * dt) if escape else None

    # slingshot: a body's speed jumps across the closest encounter
    speed = np.sqrt(np.sum(vel_traj**2, axis=2))                 # (T,n)
    t_enc = k_min[0]
    slingshot = False; sling_body = None
    if 0 < t_enc < T - 1:
        w = max(1, T // 50)
        before = speed[max(0, t_enc - w):t_enc].mean(axis=0)
        after = speed[t_enc:min(T, t_enc + w)].mean(axis=0)
        gains = after / np.maximum(before, 1e-9)
        if gains.max() > slingshot_gain:
            slingshot = True; sling_body = int(gains.argmax())

    # capture: at the end, two bodies form a tight bound pair while a third is far
    end_pair = dist[-1][iu[0], iu[1]]
    tight = end_pair.argmin()
    i, j = iu[0][tight], iu[1][tight]
    third = [b for b in range(n) if b not in (i, j)]
    capture = False; cap_pair = None
    if n == 3 and rcom[-1, third[0]] > r_escape * 0.5:
        rel_v = np.sum((vel_traj[-1, i] - vel_traj[-1, j])**2)
        rel_r = np.sqrt(np.sum((pos_traj[-1, i] - pos_traj[-1, j])**2) + eps**2)
        mu = m[i] * m[j] / (m[i] + m[j])
        two_body_E = 0.5 * mu * rel_v - G * m[i] * m[j] / rel_r
        if two_body_E < 0 and end_pair[tight] < r_escape * 0.5:
            capture = True; cap_pair = (int(i), int(j))

    return {
        "collision": collision, "collision_pair": pair_ij if collision else None,
        "collision_time": float(k_min[0] * dt) if collision else None,
        "min_pair_distance": float(min_pair),
        "escape": escape, "escape_body": escaper if escape else None,
        "escape_time": escape_time, "max_com_distance": float(max_com_body[escaper]),
        "slingshot": slingshot, "slingshot_body": sling_body,
        "capture": capture, "capture_pair": cap_pair,
    }


# ---- Gravitational field ----------------------------------------------------
def potential_grid(masses, positions, xlim, ylim, res=220, eps=0.05):
    """Gravitational potential Phi(x, y) = -sum_i G m_i / |r - r_i| on a grid.

    Returns (X, Y, Phi) suitable for a contour / heat map.
    """
    m = np.asarray(masses, float)
    pos = np.asarray(positions, float)
    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(xs, ys)
    Phi = np.zeros_like(X)
    for mi, (px, py) in zip(m, pos):
        Phi -= G * mi / np.sqrt((X - px)**2 + (Y - py)**2 + eps**2)
    return X, Y, Phi