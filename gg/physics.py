"""Softened Newtonian gravity + symplectic integration (plan.md 2.4, A1, A2).

Dimension-generic: every function works on arrays of shape (d,) / (n, d), so the
3D lift (Stage E) reuses this file unchanged - only the vector length changes.
"""
from __future__ import annotations

import numpy as np

G = 1.0
SOFTENING = 0.05                 # epsilon in A1 / A2, removes the 1/r^2 singularity
VEL_CLAMP_ESCAPE_MULT = 4.0      # clamp |v| to this * local escape speed so NaNs never enter a gradient

# Discrete action set (plan.md 2.3): 0 = coast, 1..8 = fixed burn on a compass rose.
_COMPASS_2D = np.array(
    [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=float
)
COMPASS_2D = _COMPASS_2D / np.linalg.norm(_COMPASS_2D, axis=1, keepdims=True)
N_ACTIONS_2D = 9


def action_to_thrust(a: int, a_mag: float, dims: int = 2) -> np.ndarray:
    if dims != 2:
        raise NotImplementedError("3D action head is Stage E")
    if a == 0:
        return np.zeros(dims)
    return a_mag * COMPASS_2D[a - 1]


def accel_gravity(r, body_pos, body_mass, eps: float = SOFTENING) -> np.ndarray:
    """a_grav at position r from bodies (n, d) / (n,). Formula A1 (thrust added by caller)."""
    d = body_pos - r                              # (n, d): craft -> body
    dist2 = np.sum(d * d, axis=-1) + eps * eps
    inv = dist2 ** -1.5
    return G * np.sum((body_mass * inv)[:, None] * d, axis=0)


def potential(r, body_pos, body_mass, eps: float = SOFTENING) -> float:
    """Softened specific potential Phi(r) = -sum G M_i / sqrt(|d|^2 + eps^2).

    Its gradient is exactly -accel_gravity, so E = 1/2 |v|^2 + Phi is the invariant
    to watch for integration drift (the Stage-A gate).
    """
    d = body_pos - r
    dist = np.sqrt(np.sum(d * d, axis=-1) + eps * eps)
    return float(-G * np.sum(body_mass / dist))


def dpotential_dt(r, body_pos, body_vel, body_mass, eps: float = SOFTENING) -> float:
    """dPhi/dt at FIXED r, when the bodies are moving (Stage C observation group).

    Phi = -sum G M_i / d_i with d_i = sqrt(|r - r_i|^2 + eps^2), so
        dPhi/dt = sum_i G M_i (r_i - r) . v_i / d_i^3
    which is just each body's own acceleration contribution dotted with its velocity.
    Zero whenever the bodies are static, which is why it stays zero-filled before C.
    """
    d = np.asarray(r, dtype=float) - np.asarray(body_pos, dtype=float)   # (n, dims)
    dist = np.sqrt((d * d).sum(axis=1) + eps * eps)
    contrib = (G * np.asarray(body_mass) / dist ** 3)                    # (n,)
    return float(-(contrib * (d * np.asarray(body_vel, dtype=float)).sum(axis=1)).sum())


def specific_energy(r, v, body_pos, body_mass, eps: float = SOFTENING) -> float:
    return 0.5 * float(np.dot(v, v)) + potential(r, body_pos, body_mass, eps)


def angular_momentum_2d(r, v) -> float:
    return float(r[0] * v[1] - r[1] * v[0])


def eccentricity_vec_2d(r, v, mu: float) -> np.ndarray:
    """e = (v x L)/mu - r_hat, with L scalar in 2D so v x L = (v_y L, -v_x L)."""
    lz = angular_momentum_2d(r, v)
    rmag = float(np.linalg.norm(r)) + 1e-12
    return np.array([v[1] * lz, -v[0] * lz]) / mu - r / rmag


def verlet_step(r, v, dt, accel_fn, a_thrust, process_noise=None):
    """One velocity-Verlet / leapfrog step (A2). `accel_fn(r)` returns gravity only;
    `a_thrust` is held constant across the step."""
    a0 = accel_fn(r) + a_thrust
    v_half = v + 0.5 * dt * a0
    r_new = r + dt * v_half
    a1 = accel_fn(r_new) + a_thrust
    v_new = v_half + 0.5 * dt * a1
    if process_noise is not None:
        v_new = v_new + process_noise
    return r_new, v_new
