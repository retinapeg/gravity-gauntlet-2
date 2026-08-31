"""The Stage A gate, as pytest (tests/README.md). No network, no sandboxes.

`stage_a_selftest.py` prints the same gate as a human-readable table; this file is
the machine-checkable version that `make test` runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from gg import spec
from gg.physics import accel_gravity, action_to_thrust, potential, specific_energy
from gg.rollout_worker import run_episode
from gg.universe import build_universe

BASE = dict(stage="A", dims=2, deterministic=True, max_steps=500, gamma=0.99,
            dt=0.02, obs_spec_hash=spec.obs_spec_hash(2))


def run(**over):
    return run_episode(dict(BASE, **over))


# --------------------------------------------------------------- 1. integrator

def test_energy_drift_under_one_percent_coasting():
    """Coasting, E = 1/2|v|^2 + Phi is a true invariant of the *simulated* system,
    so drift here is integrator error alone."""
    ep = run(policy_kind="no_thrust")
    assert ep["steps"] == 500
    assert ep["energy_drift"] < 0.01


def test_softened_potential_is_the_antigradient_of_the_softened_accel():
    """The invariant above is only meaningful if Phi and a_grav match. Finite-
    difference the potential and compare with accel_gravity."""
    u = build_universe("A")
    r = np.array([1.7, -0.9])
    h = 1e-6
    grad = np.array([
        (potential(r + h * e, u.body_pos, u.body_mass)
         - potential(r - h * e, u.body_pos, u.body_mass)) / (2 * h)
        for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    ])
    assert np.allclose(-grad, accel_gravity(r, u.body_pos, u.body_mass), rtol=1e-5, atol=1e-8)


# ------------------------------------------------------------- 2. terminations

@pytest.mark.parametrize("expected, over", [
    ("timeout",   dict(policy_kind="no_thrust")),
    ("collision", dict(policy_kind="no_thrust", overrides={"craft_v0": [-0.55, 0.0]})),
    ("escape",    dict(policy_kind="no_thrust", overrides={"craft_v0": [2.0, 0.0]})),
    ("portal",    dict(policy_kind="scripted_pursuit")),
])
def test_all_four_terminations_fire(expected, over):
    """Each terminal is forced from a hand-built state rather than hoped for."""
    ep = run(**over)
    assert ep["termination"] == expected
    assert ep["success"] is (expected == "portal")


# ------------------------------------------------------- 3. the A7 speed gate

def test_fast_pass_through_the_portal_is_not_a_capture():
    """The most likely reward hack: dive through the portal at speed and claim the
    goal bonus. A7 gates capture on |v - v_p| < v_capture, so this must fail."""
    u = build_universe("A")
    ep = run(policy_kind="no_thrust",
             overrides={"craft_r0": [2.6, 1.7], "craft_v0": [0.0, 3.0]})

    traj = np.array([[row[1], row[2]] for row in ep["trajectory"]])
    closest = float(np.linalg.norm(traj - u.portal_pos, axis=1).min())
    assert closest < u.r_capture, "test is void unless the craft really passes inside r_capture"

    assert ep["success"] is False
    assert ep["termination"] != "portal"


# ------------------------------------------------------------ 4. the obs spec

def test_layout_width_equals_the_sum_of_group_sizes():
    layout, total = spec.obs_layout(2)
    assert total == sum(size for _, size, _ in spec.OBS_GROUPS_2D) == spec.OBS_DIM_2D
    assert len(layout) == len(spec.OBS_GROUPS_2D)


def test_groups_tile_the_vector_without_gaps_or_overlap():
    layout, total = spec.obs_layout(2)
    covered = sorted((start, stop) for start, stop, _ in layout.values())
    assert covered[0][0] == 0 and covered[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_obs_spec_hash_changes_when_a_group_is_added_or_reordered(monkeypatch):
    """That hash is the only thing standing between a misaligned observation
    vector and a week of confusing results."""
    original = list(spec.OBS_GROUPS_2D)
    before = spec.obs_spec_hash(2)

    monkeypatch.setattr(spec, "OBS_GROUPS_2D", list(reversed(original)), raising=True)
    assert spec.obs_spec_hash(2) != before

    monkeypatch.setattr(spec, "OBS_GROUPS_2D", original + [("bogus", 1, "A")], raising=True)
    assert spec.obs_spec_hash(2) != before


def test_stage_active_is_live_from_its_own_rung_onward():
    assert spec.stage_active("A", "A") is True
    assert spec.stage_active("A", "F") is True
    assert spec.stage_active("C", "B") is False
    assert spec.stage_active("C", "C") is True


def test_stage_a_zero_fills_every_later_group():
    """B/C-only groups must be exactly zero at Stage A (the A1 gate)."""
    ep = run(policy_kind="scripted_pursuit")
    obs = np.array(ep["observations"])
    assert obs.shape[1] == spec.OBS_DIM_2D
    assert np.all(np.isfinite(obs))
    for name, (start, stop, first) in spec.OBS_LAYOUT_2D.items():
        if not spec.stage_active(first, "A"):
            assert np.all(obs[:, start:stop] == 0.0), f"{name} must be zero-filled at Stage A"


# -------------------------------------------------------- 5. action -> thrust

def test_action_zero_is_exactly_no_thrust():
    assert np.array_equal(action_to_thrust(0, 0.3), np.zeros(2))


def test_the_eight_compass_actions_are_unit_directions():
    dirs = [action_to_thrust(a, 1.0) for a in range(1, 9)]
    assert len({tuple(np.round(d, 9)) for d in dirs}) == 8
    for d in dirs:
        assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.allclose(action_to_thrust(5, 0.3), 0.3 * action_to_thrust(5, 1.0))


def test_three_d_actions_raise_rather_than_returning_a_wrong_shape():
    with pytest.raises(NotImplementedError):
        action_to_thrust(1, 0.3, dims=3)


# ----------------------------------------------------------- 6. worker output

def test_worker_output_shapes_line_up():
    """plan.md 4.2: trajectory has one extra row (t=0); the learning arrays are
    one per step. The fan-out and the renderer both rely on this."""
    ep = run(policy_kind="scripted_pursuit")
    n = ep["steps"]
    assert len(ep["trajectory"]) == n + 1
    assert len(ep["bodies_trajectory"]) == n + 1
    for key in ("observations", "actions", "logprobs", "rewards"):
        assert len(ep[key]) == n, key
    assert ep["obs_spec_hash"] == spec.obs_spec_hash(2)
    assert 0.0 <= ep["fuel_left"] <= 1.0
