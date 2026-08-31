"""Frozen observation superset + reward config, shared by controller and worker.

The observation layout is FROZEN from Stage A (plan.md 5, 5.1): later stages only
flip a group from zero-filled to live via `FIRST_ACTIVE_STAGE`, never resize the
vector, so ablation / necessity scores stay directly comparable across the ladder.
"""
from __future__ import annotations

import hashlib
import json

STAGES = ("A", "B", "C", "D", "E", "F")

# The order of this list IS the observation-vector order.
#   (name, size in 2D, first stage at which the group carries real signal)
OBS_GROUPS_2D = [
    ("portal_rel_pos",       2, "A"),   # Delta r_p = r_p - r
    ("craft_vel",            2, "A"),   # v
    ("speed",                1, "A"),   # |v|
    ("closing_speed_portal", 1, "A"),   # -d|Delta r_p|/dt   (+ = approaching)
    ("bearing_error",        1, "A"),   # signed angle(v, Delta r_p) / pi
    ("a_grav",               2, "A"),   # net gravitational acceleration vector
    ("a_grav_magdir",        2, "A"),   # |a_grav| / g0 , phi_g / pi
    ("potential",            1, "A"),   # Phi = -sum G M_i / d_i
    ("nearest_body",         4, "B"),   # (r_n - r), d_n, M_n
    ("tca_nearest",          1, "C"),   # time to closest approach to nearest body
    ("nearest_rel_vel",      2, "C"),   # v_n - v
    ("dphi_dt",              1, "C"),   # partial Phi / partial t  (bodies moving)
    ("body_count",           1, "B"),   # N
    ("perbody_topk",         9, "B"),   # {(r_i - r), M_i} for the k nearest, k=3
    ("fuel_fraction",        1, "A"),   # f = m_fuel / m_fuel,0
    ("craft_mass",           1, "A"),   # m / m_0
    ("control_authority",    1, "A"),   # Pi_1 = a_thrust_max / g0
    ("time_fraction",        1, "A"),   # t / T_budget
    ("orbital_energy",       1, "A"),   # E = 1/2 |v|^2 + Phi
    ("angular_momentum",     1, "A"),   # L = x v_y - y v_x   (scalar in 2D)
    ("eccentricity_vec",     2, "A"),   # e  (already dimensionless)
]

TOPK = 3


def obs_layout(dims: int = 2):
    """Return ({name: (start, stop, first_stage)}, total_dim)."""
    if dims != 2:
        raise NotImplementedError("step 1 is 2D only; the 3D lift is Stage E")
    layout, i = {}, 0
    for name, size, first in OBS_GROUPS_2D:
        layout[name] = (i, i + size, first)
        i += size
    return layout, i


OBS_LAYOUT_2D, OBS_DIM_2D = obs_layout(2)


def stage_active(first_stage: str, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(first_stage)


def obs_spec_hash(dims: int = 2) -> str:
    """Short hash of the layout; the worker rejects a mismatched controller hash
    so the two can never silently drift (plan.md 4)."""
    payload = json.dumps({"groups": OBS_GROUPS_2D, "topk": TOPK, "dims": dims}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


# Reward config (plan.md 2.5). Shaping is potential-based and per-step clipped so
# it cannot change the optimal policy or be farmed by oscillating.
REWARD = dict(
    k_progress=1.0, progress_clip=0.10,
    k_proximity=0.3, proximity_rho_body_radii=3.0,
    k_fuel=0.05,
    k_time=0.01,
    # A7 capture has TWO conditions; until Stage C only one of them was shaped.
    # `k_progress` pays for closing distance and is indifferent to arrival speed, so
    # diving at the portal is optimal under the reward and is then refused by the speed
    # gate - measured at Stage C as 41/48 held-out episodes inside r_capture at 0.76-0.96
    # against v_capture 0.6. `k_brake` is the matching potential for the speed condition.
    k_brake=100.0, brake_clip=1.0, brake_gate_radii=3.0,
    # k_brake is LARGE on purpose. Measured at Stage C: 1 and 10 leave the
    # policy diving at 1.1 (v_capture is 0.6) and never capturing; 40 is where
    # braking appears and held-out success leaves zero. brake_clip must stay
    # well above k_brake * max per-step dPhi (~0.007) or the clip binds on ~7%
    # of steps at k_brake=40 and the shaping stops telescoping.
    r_goal=100.0, r_crash=100.0, r_timeout=30.0, r_escape=50.0,
)
