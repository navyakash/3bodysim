"""
Batched survey engine.

All simulations (and a perturbed twin of each) are stepped forward together as
one big (N, n_bodies, 2) array, so scans of hundreds-to-thousands of runs take
seconds. Each run is reduced to a few numbers and classified:

  collision : two bodies came within r_collide
  escape    : a body reached r_escape from the centre of mass
  chaotic   : bounded, but a 1e-6 twin diverged strongly (sensitive)
  periodic  : bounded, and its twin stayed close (regular / stable orbit)

It also exposes initial_features(): quantities known at t=0 (energy, angular
momentum, ...) used as inputs to the fate-prediction model. Those are computed
WITHOUT running a simulation - that separation is what makes the ML honest.
"""

import numpy as np
import pandas as pd

from three_body import G

PERT = 1e-6   # size of the twin perturbation used to test for chaos

# One shared scenario so the stability map and the ML model describe the same
# family of systems: a heavy central body, a distant perturber, and a test body
# launched from a range of positions (x) and speeds (vy).
DEFAULT_SCENARIO = dict(
    masses=[3.0, 1.0, 1.0],
    base_pos=[[0.0, 0.0], [1.5, 0.0], [5.0, 0.0]],
    base_vel=[[0.0, 0.0], [0.0, 1.0], [0.0, 0.6]],
    body_index=1,
    pos_range=(1.0, 3.0),
    vel_range=(0.5, 3.0),
    dt=0.004,
    eps=0.02,
)


def _accel(pos, m, eps):
    diff = pos[:, None, :, :] - pos[:, :, None, :]
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


def _min_pair_dist(pos, eps=0.0):
    diff = pos[:, None, :, :] - pos[:, :, None, :]
    dist2 = np.sum(diff**2, axis=3)
    d = np.arange(pos.shape[1])
    dist2[:, d, d] = np.inf
    return np.sqrt(dist2.min(axis=(1, 2)) + eps)


def _simulate_and_classify(m, pos0, vel0, dt, steps, eps,
                           r_collide, r_escape, growth_thresh, pert_index):
    """Run a batch of initial states (+ twins) and classify each. Returns a dict
    of per-simulation arrays."""
    S = pos0.shape[0]; n = pos0.shape[1]; M = m.sum()
    pos_p = pos0.copy(); pos_p[:, pert_index, 0] += PERT
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
        p = pos[:S]
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
    growth = np.sqrt(((pos[:S] - pos[S:])**2).sum(axis=(1, 2))) / PERT

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

    return dict(label=label, stable=bounded, escape=escape, collision=collision,
                energy_error=e_err, max_separation=max_sep, lifetime=life * dt,
                twin_growth=growth)


def _build_states(scenario, pos_vals, vel_vals):
    """Construct (S, n, 2) initial position/velocity arrays from paired samples."""
    m = np.asarray(scenario["masses"], float)
    bi = scenario["body_index"]
    S = len(pos_vals)
    pos0 = np.tile(np.asarray(scenario["base_pos"], float), (S, 1, 1))
    vel0 = np.tile(np.asarray(scenario["base_vel"], float), (S, 1, 1))
    pos0[:, bi, 0] = pos_vals
    vel0[:, bi, 1] = vel_vals
    return m, pos0, vel0


def initial_features(scenario, pos_vals, vel_vals):
    """Quantities known at t=0 (no simulation). Returns a DataFrame of features."""
    m, pos0, vel0 = _build_states(scenario, np.asarray(pos_vals, float),
                                  np.asarray(vel_vals, float))
    eps = scenario["eps"]
    e0 = _energy(pos0, vel0, m, eps)
    lz = np.sum(m[None, :] * (pos0[:, :, 0] * vel0[:, :, 1]
                              - pos0[:, :, 1] * vel0[:, :, 0]), axis=1)
    mind = _min_pair_dist(pos0)
    return pd.DataFrame({
        "init_position": np.asarray(pos_vals, float),
        "init_velocity": np.asarray(vel_vals, float),
        "init_energy": e0,
        "init_ang_mom": lz,
        "init_min_dist": mind,
    })


def run_stability_map(scenario, n_grid, steps,
                      r_collide=0.06, r_escape=10.0, growth_thresh=1e3):
    """Grid scan -> (dataframe, class_grid, pos_axis, vel_axis)."""
    pa = np.linspace(*scenario["pos_range"], n_grid)
    va = np.linspace(*scenario["vel_range"], n_grid)
    PY, VX = np.meshgrid(pa, va, indexing="ij")
    m, pos0, vel0 = _build_states(scenario, PY.ravel(), VX.ravel())
    out = _simulate_and_classify(m, pos0, vel0, scenario["dt"], steps, scenario["eps"],
                                 r_collide, r_escape, growth_thresh, scenario["body_index"])
    df = pd.DataFrame({
        "init_position": PY.ravel(), "init_velocity": VX.ravel(),
        "class": out["label"], "stable": out["stable"],
        "escape": out["escape"], "collision": out["collision"],
        "energy_error": out["energy_error"], "max_separation": out["max_separation"],
        "lifetime": out["lifetime"],
    })
    return df, out["label"].reshape(n_grid, n_grid), pa, va


def run_samples(scenario, n_samples, steps, seed=0,
                r_collide=0.06, r_escape=10.0, growth_thresh=1e3):
    """Randomly sampled scan -> a dataset with t=0 features + labels + diagnostics."""
    rng = np.random.default_rng(seed)
    pos_vals = rng.uniform(*scenario["pos_range"], n_samples)
    vel_vals = rng.uniform(*scenario["vel_range"], n_samples)
    m, pos0, vel0 = _build_states(scenario, pos_vals, vel_vals)
    out = _simulate_and_classify(m, pos0, vel0, scenario["dt"], steps, scenario["eps"],
                                 r_collide, r_escape, growth_thresh, scenario["body_index"])
    feats = initial_features(scenario, pos_vals, vel_vals)
    feats["class"] = out["label"]
    feats["stable"] = out["stable"]
    feats["energy_error"] = out["energy_error"]
    feats["max_separation"] = out["max_separation"]
    feats["lifetime"] = out["lifetime"]
    return feats