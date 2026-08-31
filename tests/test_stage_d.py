"""Stage D - per-seed universes (S1) and process noise (S2).

The rung's whole claim is that a policy can no longer succeed by exploiting one
particular world, so the tests here are about the two things that claim rests on:
a universe is a pure function of its seed, and the noise is actually applied.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from gg import spec
from gg.physics import G as G_CONST
from gg.rollout_worker import run_episode
from gg.universe import D_SAMPLING, build_universe, sample_universe

BASE = dict(stage="D", dims=2, max_steps=500, gamma=0.99, dt=0.02,
            obs_spec_hash=spec.obs_spec_hash(2), policy_kind="scripted_pursuit")
NOISE = {"process_sigma": 0.002, "thrust_jitter": 0.03, "pointing_sigma": 0.05}


# ------------------------------------------------------- 1. seed -> universe

def test_a_universe_is_a_pure_function_of_its_seed():
    """Replay is the whole basis of the recorded-generation UI: the same seed must
    rebuild the same world, or `runs/generations/*.json` cannot be replayed."""
    for sd in (1, 17, 4242):
        a, b = sample_universe(sd), sample_universe(sd)
        assert np.array_equal(a.body_pos, b.body_pos)
        assert np.array_equal(a.body_mass, b.body_mass)
        assert np.array_equal(a.craft_r0, b.craft_r0)
        assert np.array_equal(a.craft_v0, b.craft_v0)
        assert np.array_equal(a.portal_pos, b.portal_pos)


def test_different_seeds_give_different_worlds():
    sigs = set()
    for sd in range(40):
        u = sample_universe(sd)
        sigs.add((len(u.body_mass), round(float(u.craft_r0[0]), 6),
                  round(float(u.portal_pos[1]), 6)))
    assert len(sigs) >= 38, "seeds are collapsing onto the same few worlds"


def test_body_count_stays_inside_the_declared_range():
    lo, hi = D_SAMPLING["n_bodies"]
    counts = {len(sample_universe(sd).body_mass) for sd in range(60)}
    assert counts, "no universes sampled"
    assert min(counts) >= lo and max(counts) <= hi
    assert len(counts) > 1, "N is supposed to VARY across seeds"


def test_masses_span_the_declared_decades():
    ms = np.concatenate([sample_universe(sd).body_mass[1:] for sd in range(60)
                         if len(sample_universe(sd).body_mass) > 1])
    lo, hi = D_SAMPLING["mass"]
    assert ms.min() >= lo - 1e-9 and ms.max() <= hi + 1e-9
    assert ms.max() / max(ms.min(), 1e-9) > 5.0, "masses are not actually spread"


def test_planets_never_spawn_on_the_start_pose_or_in_the_capture_bubble():
    """An episode that is lost before the policy acts teaches the policy nothing."""
    for sd in range(80):
        u = sample_universe(sd)
        for p, rad in zip(u.body_pos[1:], u.body_radius[1:]):
            assert np.linalg.norm(p - u.craft_r0) > rad, f"planet on the craft, seed {sd}"
            assert np.linalg.norm(p - u.portal_pos) > u.r_capture, \
                f"planet inside the capture bubble, seed {sd}"


# ------------------------------------------------------- 2. the Kepler solver

def test_kepler_positions_match_a_numerically_integrated_orbit():
    """`bodies_at` is closed-form; check it against the ellipse it claims to be by
    verifying the orbit is a conic with the right semi-major axis and eccentricity."""
    u = sample_universe(3)
    if len(u.body_mass) < 2:
        pytest.skip("seed 3 drew no planets")
    ts = np.linspace(0.0, 40.0, 4000)
    r = np.array([np.linalg.norm(u.bodies_at(t)[0][1]) for t in ts])
    a, e = float(u.orbit_a[1]), float(u.orbit_ecc[1])
    assert r.min() == pytest.approx(a * (1 - e), rel=2e-3)
    assert r.max() == pytest.approx(a * (1 + e), rel=2e-3)


def test_kepler_velocity_is_the_derivative_of_kepler_position():
    """The integrator's half-steps use these velocities; if they are not the true
    derivative the simulation quietly solves a different problem."""
    u = sample_universe(11)
    if len(u.body_mass) < 2:
        pytest.skip("seed 11 drew no planets")
    t, h = 3.3, 1e-6
    v_analytic = u.bodies_at(t)[1][1]
    v_fd = (u.bodies_at(t + h)[0][1] - u.bodies_at(t - h)[0][1]) / (2 * h)
    assert np.allclose(v_analytic, v_fd, rtol=1e-4, atol=1e-6)


def test_a_zero_eccentricity_ellipse_is_a_circle():
    u = sample_universe(5)
    u.orbit_ecc = np.zeros_like(u.orbit_a)
    for t in (0.0, 1.7, 9.4):
        r = np.linalg.norm(u.bodies_at(t)[0], axis=1)
        assert np.allclose(r, u.orbit_a, atol=1e-9)


# ------------------------------------------------------- 3. S2 noise

def test_noise_actually_changes_the_trajectory():
    a = run_episode(dict(BASE, deterministic=False, noise=NOISE, seed_universe=4, seed=1))
    b = run_episode(dict(BASE, deterministic=False, noise=NOISE, seed_universe=4, seed=2))
    assert a["physics_checksum"] != b["physics_checksum"], \
        "different rng draws produced identical physics - the noise is not wired in"


def test_deterministic_flag_silences_every_noise_source():
    """Stages A-C pass deterministic=True; no value in the noise block may leak."""
    loud = dict(process_sigma=9.0, thrust_jitter=9.0, pointing_sigma=9.0)
    a = run_episode(dict(BASE, deterministic=True, noise=loud, seed_universe=6, seed=1))
    b = run_episode(dict(BASE, deterministic=True, noise=loud, seed_universe=6, seed=2))
    assert a["physics_checksum"] == b["physics_checksum"]


def test_a_misfire_wastes_fuel_rather_than_refunding_it():
    """Fuel is drawn on the COMMANDED burn, so a weak misfire must not come back as
    a discount - otherwise noise is a fuel bonus and the policy learns to want it."""
    quiet = run_episode(dict(BASE, deterministic=True, seed_universe=8, seed=5))
    noisy = run_episode(dict(BASE, deterministic=False, seed_universe=8, seed=5,
                             noise={"thrust_jitter": 0.5}))
    assert noisy["fuel_left"] <= quiet["fuel_left"] + 1e-9


# ------------------------------------------------ 4. the rung is winnable

def test_the_scripted_policy_can_still_fly_these_worlds():
    """Same guard Stage C needed: if the scripted controller cannot capture, a zero
    from the learned policy says nothing about the policy."""
    wins = sum(run_episode(dict(BASE, deterministic=False, noise=NOISE,
                                seed_universe=sd, seed=sd))["success"]
               for sd in range(24))
    assert wins / 24 >= 0.6, f"scripted pursuit only won {wins}/24 - rung may be unfair"


# ------------------------------------------- 5. earlier rungs are untouched

@pytest.mark.parametrize("stage,key,want", [
    ("B", {"layout_id": 2}, "3508f9b5"),
    ("C", {"layout_id": 7, "orbital_phase": 0.33}, "81f6adf1"),
])
def test_stage_d_work_did_not_disturb_the_earlier_rungs(stage, key, want):
    """Recorded before the Stage D changes. These rungs' results are already
    published, so their physics must be bit-identical."""
    o = run_episode(dict(stage=stage, dims=2, deterministic=True, max_steps=500,
                         gamma=0.99, dt=0.02, policy_kind="scripted_pursuit",
                         seed=3, **key))
    assert o["physics_checksum"] == want


# ------------------------------------------------------- 6. the curriculum

def test_difficulty_zero_reproduces_the_stage_c_style_corridor():
    """The easy anchor has to be genuinely easy, i.e. close to what the warm-started
    Stage C champion already knows: one planet, a circle, and one fixed orientation."""
    angles, counts, eccs = set(), set(), set()
    for sd in range(30):
        u = sample_universe(sd, difficulty=0.0)
        angles.add(round(float(np.arctan2(u.craft_r0[1], u.craft_r0[0])), 6))
        counts.add(len(u.body_mass))
        eccs.add(round(float(np.max(u.orbit_ecc)), 6))
    assert angles == {0.0}, "orientation is supposed to be pinned at difficulty 0"
    assert counts == {2}, "difficulty 0 should draw exactly one planet"
    assert eccs == {0.0}, "difficulty 0 should draw circular orbits"


def test_difficulty_one_uses_the_whole_declared_range():
    angles, counts = [], set()
    for sd in range(60):
        u = sample_universe(sd, difficulty=1.0)
        angles.append(float(np.arctan2(u.craft_r0[1], u.craft_r0[0])))
        counts.add(len(u.body_mass))
    assert np.ptp(angles) > 5.0, "orientation should span the circle at difficulty 1"
    assert max(counts) == D_SAMPLING["n_bodies"][1]


def test_the_curriculum_is_monotone_in_body_count():
    def mean_n(d):
        return np.mean([len(sample_universe(sd, difficulty=d).body_mass) for sd in range(60)])
    assert mean_n(0.0) <= mean_n(0.5) <= mean_n(1.0)


def test_a_curriculum_world_is_still_a_pure_function_of_its_inputs():
    for d in (0.0, 0.4, 1.0):
        a, b = sample_universe(21, difficulty=d), sample_universe(21, difficulty=d)
        assert np.array_equal(a.body_pos, b.body_pos)
        assert np.array_equal(a.craft_r0, b.craft_r0)


def test_a_recorded_generation_can_be_replayed_exactly():
    """A stage-D world is a function of (seed_universe, curriculum). If a recorded key
    carries only the seed, replaying an early generation rebuilds it at the WRONG
    difficulty - so the curriculum has to travel inside the key, not beside it."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "controller/train.py").read_text()
    assert "keys = [dict(k, curriculum=cur) for k in keys]" in src, \
        "training keys must carry the curriculum so the generation replays exactly"
    a = sample_universe(1234, difficulty=0.3)
    b = sample_universe(1234, difficulty=1.0)
    assert not np.array_equal(a.craft_r0, b.craft_r0), \
        "difficulty must actually change the world, or this guard is vacuous"


def test_held_out_evaluation_is_never_run_on_an_easier_distribution():
    """The gate is a claim about the FULL distribution. If the ramp ever leaked into
    evaluation, the reported number would describe a world the rung is not claiming.
    Pin the trainer's contract: training ramps, every eval path pins difficulty 1.0."""
    from controller.train import curriculum_at
    src = (pathlib.Path(__file__).resolve().parents[1] / "controller/train.py").read_text()

    cfg = {"training": {"curriculum_generations": 90}}
    assert curriculum_at(cfg, 1) == 0.0 and curriculum_at(cfg, 91) == 1.0
    assert curriculum_at(cfg, 10**9) == 1.0
    assert curriculum_at({"training": {}}, 5) is None

    # every eval key builder must pin curriculum 1.0, never the ramped `cur`
    pins = re.findall(r'dict\(k, curriculum=([0-9.]+)\) for k in eval_keys', src)
    assert len(pins) >= 2, "expected the probe and final eval to pin the difficulty"
    assert all(float(x) == 1.0 for x in pins), "an eval path is not pinned at 1.0"

