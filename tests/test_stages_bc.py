"""Stage B and Stage C: the two new universe branches. Local, no network.

The property the whole necessity study rests on is that the observation vector is a
FROZEN superset (plan.md 5.1) - later rungs only flip groups from zero-filled to
live, never resize. Several tests here exist purely to defend that.
"""
from __future__ import annotations

import numpy as np
import pytest

from gg import spec
from gg.physics import SOFTENING, dpotential_dt, potential
from gg.rollout_worker import run_episode
from gg.universe import FIXED_LAYOUTS, build_universe

BASE = dict(dims=2, deterministic=True, max_steps=500, gamma=0.99, dt=0.02,
            obs_spec_hash=spec.obs_spec_hash(2))


def run(**over):
    return run_episode(dict(BASE, **over))


# ------------------------------------------------------------------- stage B

def test_twelve_layouts_exist_and_two_are_held_out():
    assert sorted(FIXED_LAYOUTS) == list(range(12))
    import json, pathlib
    keys = json.loads(pathlib.Path("stages/stage_b_fixed_planets/keys.json").read_text())
    assert [k["layout_id"] for k in keys["eval_held_out"]] == [10, 11]
    assert len(keys["train"]) == 10


def test_every_layout_keeps_the_star_and_adds_planets():
    for lid in FIXED_LAYOUTS:
        u = build_universe("B", {"layout_id": lid})
        assert np.allclose(u.body_pos[0], [0.0, 0.0]), "body 0 must remain the star"
        assert u.body_mass[0] == 1.0
        assert len(u.body_mass) == 1 + len(FIXED_LAYOUTS[lid])
        assert (u.body_mass[1:] < u.body_mass[0]).all(), "planets must not outweigh the star"
        assert u.body_vel.shape == u.body_pos.shape


def test_layouts_differ_in_geometry_not_just_index():
    """Eight near-identical fields teach the policy nothing and read as a render bug."""
    sigs = {lid: tuple(np.round(np.sort(np.linalg.norm(
        build_universe("B", {"layout_id": lid}).body_pos, axis=1)), 3))
        for lid in FIXED_LAYOUTS}
    assert len(set(sigs.values())) == len(sigs), "two layouts are geometrically identical"


def test_no_planet_sits_on_the_craft_start_or_the_portal():
    for lid in FIXED_LAYOUTS:
        u = build_universe("B", {"layout_id": lid})
        d_start = np.linalg.norm(u.body_pos[1:] - u.craft_r0, axis=1)
        d_portal = np.linalg.norm(u.body_pos[1:] - u.portal_pos, axis=1)
        assert (d_start > u.body_radius[1:]).all(), f"layout {lid} spawns the craft inside a planet"
        assert (d_portal > u.body_radius[1:]).all(), f"layout {lid} buries the portal in a planet"


def test_unknown_layout_is_rejected_loudly():
    with pytest.raises(SystemExit):
        build_universe("B", {"layout_id": 99})


def test_planet_contact_terminates_the_episode():
    """Collision must fire on a PLANET, not only on the star (Stage B step 3)."""
    u = build_universe("B", {"layout_id": 0})
    planet = u.body_pos[1]
    start = planet + np.array([0.0, -0.6])          # just below it, aimed straight in
    ep = run(stage="B", layout_id=0, policy_kind="no_thrust",
             overrides={"craft_r0": start.tolist(), "craft_v0": [0.0, 0.55]})
    assert ep["termination"] == "collision"


def test_stage_b_groups_are_live_at_b_and_zero_at_a():
    """The flip is the whole mechanism of the ladder."""
    a = np.array(run(stage="A", policy_kind="scripted_pursuit")["observations"])
    b = np.array(run(stage="B", layout_id=2, policy_kind="scripted_pursuit")["observations"])
    assert a.shape[1] == b.shape[1] == spec.OBS_DIM_2D, "the superset must never resize"
    for g in ("nearest_body", "body_count", "perbody_topk"):
        lo, hi, _ = spec.OBS_LAYOUT_2D[g]
        assert np.all(a[:, lo:hi] == 0.0), f"{g} must be zero-filled at Stage A"
        assert np.any(b[:, lo:hi] != 0.0), f"{g} must be live at Stage B"


def test_stage_c_only_groups_stay_zero_at_b():
    b = np.array(run(stage="B", layout_id=2, policy_kind="scripted_pursuit")["observations"])
    for g in ("tca_nearest", "nearest_rel_vel", "dphi_dt"):
        lo, hi, _ = spec.OBS_LAYOUT_2D[g]
        assert np.all(b[:, lo:hi] == 0.0), f"{g} is Stage C's, it must be zero at B"


# ------------------------------------------------------------------- stage C

def test_planets_orbit_but_the_star_stays_put():
    u = build_universe("C", {"layout_id": 2, "orbital_phase": 0.0})
    p0, _ = u.bodies_at(0.0)
    p1, v1 = u.bodies_at(4.0)
    assert np.allclose(p0[0], [0.0, 0.0]) and np.allclose(p1[0], [0.0, 0.0])
    assert not np.allclose(p0[1:], p1[1:]), "planets must move"
    assert np.allclose(np.linalg.norm(p0[1:], axis=1), np.linalg.norm(p1[1:], axis=1)), \
        "circular orbits must preserve radius"
    assert np.allclose((p1[1:] * v1[1:]).sum(axis=1), 0.0, atol=1e-9), \
        "circular orbit velocity must be perpendicular to the radius"


def test_orbital_phase_changes_the_world():
    a = build_universe("C", {"layout_id": 2, "orbital_phase": 0.0}).body_pos
    b = build_universe("C", {"layout_id": 2, "orbital_phase": 0.5}).body_pos
    assert not np.allclose(a, b)


def test_same_key_replays_exactly():
    """A universe must be a pure function of its key (plan.md 3.4) or replay lies."""
    k = {"layout_id": 6, "orbital_phase": 0.31}
    assert np.allclose(build_universe("C", k).body_pos, build_universe("C", k).body_pos)
    one = run(stage="C", policy_kind="scripted_pursuit", **k)
    two = run(stage="C", policy_kind="scripted_pursuit", **k)
    assert one["physics_checksum"] == two["physics_checksum"]


def test_bodies_trajectory_actually_varies_at_c():
    """The renderer reads bodies_trajectory; at C it must not be a constant."""
    ep = run(stage="C", layout_id=2, orbital_phase=0.0, policy_kind="scripted_pursuit")
    bt = np.array(ep["bodies_trajectory"])
    assert bt.shape[0] == ep["steps"] + 1
    assert np.allclose(bt[:, 0, :2], 0.0), "star column must stay at the origin"
    assert not np.allclose(bt[0, 1:, :2], bt[-1, 1:, :2]), "planet columns must move"


def test_stage_c_groups_go_live():
    obs = np.array(run(stage="C", layout_id=2, orbital_phase=0.0,
                       policy_kind="scripted_pursuit")["observations"])
    assert obs.shape[1] == spec.OBS_DIM_2D
    for g in ("tca_nearest", "nearest_rel_vel", "dphi_dt"):
        lo, hi, _ = spec.OBS_LAYOUT_2D[g]
        assert np.any(obs[:, lo:hi] != 0.0), f"{g} must be live at Stage C"


def test_dpotential_dt_matches_finite_differences():
    """dPhi/dt is a new hand-derived formula, so it gets the same treatment as the
    policy gradient: checked against a central difference of the actual potential."""
    rng = np.random.default_rng(0)
    r = np.array([2.0, 1.0])
    bpos = np.array([[0.0, 0.0], [1.2, 0.7], [-0.9, 1.8]])
    bvel = rng.normal(size=(3, 2)) * 0.3
    bmass = np.array([1.0, 0.05, 0.08])
    analytic = dpotential_dt(r, bpos, bvel, bmass)
    h = 1e-7
    plus = potential(r, bpos + h * bvel, bmass, SOFTENING)
    minus = potential(r, bpos - h * bvel, bmass, SOFTENING)
    numeric = (plus - minus) / (2 * h)
    assert abs(analytic - numeric) / max(abs(analytic) + abs(numeric), 1e-9) < 1e-5


def test_dpotential_dt_is_zero_for_static_bodies():
    bpos = np.array([[0.0, 0.0], [1.2, 0.7]])
    assert dpotential_dt(np.array([2.0, 1.0]), bpos, np.zeros_like(bpos),
                         np.array([1.0, 0.05])) == 0.0
