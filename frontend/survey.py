"""
Batched survey engine for the stability map.

All simulations (and a perturbed twin of each) are stepped forward together as
one big (N, n_bodies, 2) array, so a full 1,000-run scan takes seconds. Each run
is reduced on the fly to a few numbers and then classified:

  collision : two bodies came within r_collide
  escape    : a body reached r_escape from the centre of mass
  chaotic   : bounded, but a 1e-6 twin diverged strongly (sensitive)
  periodic  : bounded, and its twin stayed close (regular / stable orbit)
"""

import numpy as np
import pandas as pd

from three_body import G

PERT = 1e-6   # size of the twin perturbation used to test for chaos


def _accel(pos, m, eps):
    diff = pos[:, None, :, :] - pos[:, :, None, :]      # (N,n,n,2): diff[k,i,j]=pos_j-pos_i
    dist2 = np.sum(diff**2, axis=3) + eps**2
    inv = dist2 ** -1.5
    d = np.arange(pos.shape[1])
    inv[:, d, d] = 0.0
    return G * np.sum(inv[:, :, :, None] * m[None, None, :, None] * diff, axis=2)


def _rk4(pos, vel, m, dt, eps):
    a1 = _accel(pos, m, eps)
    p2, v2 = pos + 0.5*dt*vel, vel + 0.5*dt*a1; a2 = _accel(p2, m, eps)
    p3, v3 = pos + 0.5*dt*v2,  vel + 0.5*dt*a2; a3 = _accel(p3, m, eps)
    p4, v4 = pos + dt*v3,      vel + dt*a3;     a4 = _accel(p4, m, eps)
    return (pos + (dt/6.0)*(vel + 2*v2 + 2*v3 + v4),
            vel + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4))


def _energy(pos, vel, m, eps):
    ke = 0.5 * np.sum(m[None, :] * np.sum(vel**2, axis=2), axis=1)
    diff = pos[:, None, :, :] - pos[:, :, None, :]
    dist = np.sqrt(np.sum(diff**2, axis=3) + eps**2)
    n = pos.shape[1]; iu = np.triu_indices(n, 1)
    mm = np.outer(m, m)[iu]
    pe = -G * np.sum(mm[None, :] / dist[:, iu[0], iu[1]], axis=1)
    return ke + pe


def run_stability_map(masses, base_positions, base_velocities, body_index,
                      pos_axis, vel_axis, dt, steps, eps,
                      r_collide=0.06, r_escape=15.0, growth_thresh=1e3):
    """Scan a grid: Y axis = body_index's initial x-position, X axis = its vy.

    Every grid cell is one simulation (plus a perturbed twin, used only to test
    for chaos). Returns (dataframe, class_grid, pos_axis, vel_axis).
    """
    m = np.asarray(masses, float)
    n = len(m); M = m.sum()
    ny, nx = len(pos_axis), len(vel_axis)
    S = ny * nx

    pos0 = np.tile(np.asarray(base_positions, float), (S, 1, 1))
    vel0 = np.tile(np.asarray(base_velocities, float), (S, 1, 1))
    PY, VX = np.meshgrid(pos_axis, vel_axis, indexing="ij")
    pos0[:, body_index, 0] = PY.ravel()
    vel0[:, body_index, 1] = VX.ravel()

    # stack the real runs and their perturbed twins into one batch of 2*S
    pos_p = pos0.copy(); pos_p[:, body_index, 0] += PERT
    pos = np.concatenate([pos0, pos_p], axis=0)
    vel = np.concatenate([vel0, vel0], axis=0)

    E0 = _energy(pos, vel, m, eps)[:S]
    min_pair = np.full(S, np.inf)
    max_sep = np.zeros(S)
    max_com = np.zeros(S)
    life = np.full(S, steps, dtype=float)
    done = np.zeros(S, bool)
    d = np.arange(n)

    for s in range(steps):
        p = pos[:S]                                       # diagnostics on the real runs only
        diff = p[:, None, :, :] - p[:, :, None, :]
        dist2 = np.sum(diff**2, axis=3)
        max_sep = np.maximum(max_sep, np.sqrt(dist2.max(axis=(1, 2))))
        dd = dist2.copy(); dd[:, d, d] = np.inf
        mp = np.sqrt(dd.min(axis=(1, 2)))
        min_pair = np.minimum(min_pair, mp)

        com = (m[None, :, None] * p).sum(axis=1) / M
        mc = np.sqrt(((p - com[:, None, :])**2).sum(axis=2)).max(axis=1)
        max_com = np.maximum(max_com, mc)

        newly = (~done) & ((mp < r_collide) | (mc > r_escape))
        life[newly] = s; done |= newly

        pos, vel = _rk4(pos, vel, m, dt, eps)

    Ef = _energy(pos, vel, m, eps)[:S]
    e_err = np.abs((Ef - E0) / E0)
    sep_final = np.sqrt(((pos[:S] - pos[S:])**2).sum(axis=(1, 2)))
    growth = sep_final / PERT

    collision = min_pair < r_collide
    escape = (~collision) & (max_com > r_escape)
    bounded = ~(collision | escape)
    chaotic = bounded & (growth > growth_thresh)
    periodic = bounded & ~chaotic

    label = np.empty(S, dtype=object)
    label[collision] = "collision"
    label[escape] = "escape"
    label[chaotic] = "chaotic"
    label[periodic] = "periodic"

    df = pd.DataFrame({
        "init_position": PY.ravel(),
        "init_velocity": VX.ravel(),
        "class": label,
        "stable": bounded,
        "escape": escape,
        "collision": collision,
        "energy_error": e_err,
        "max_separation": max_sep,
        "lifetime": life * dt,
        "twin_growth": growth,
    })
    return df, label.reshape(ny, nx), pos_axis, vel_axis