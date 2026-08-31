"""The necessity study's pure functions (plan.md 6). No sandboxes, no network.

The fleet-dependent arms (M2 ablation, M4 nested training) are exercised by the A9
gate run itself; what is testable offline is the scoring, the classification, and
the rank-stability statistic the gate turns on.
"""
from __future__ import annotations

import numpy as np
import pytest

from controller import necessity as nec
from gg import net, spec


def test_active_groups_excludes_later_stage_groups():
    a = nec.active_groups("A")
    assert "portal_rel_pos" in a and "a_grav" in a
    assert "nearest_body" not in a, "Stage B group must not be scored at Stage A"
    assert "tca_nearest" not in a, "Stage C group must not be scored at Stage A"
    assert set(nec.active_groups("A")) <= set(nec.active_groups("C"))


def test_nested_sets_are_actually_nested():
    """M4 only means anything if each set contains the previous one."""
    names = ["A", "B", "C", "D", "E"]
    for lo, hi in zip(names, names[1:]):
        assert set(nec.NESTED_SETS[lo]) < set(nec.NESTED_SETS[hi]), f"{lo} not inside {hi}"
    assert set(nec.NESTED_SETS["E"]) == set(spec.OBS_LAYOUT_2D), "E must be the full set"


def test_nested_set_e_leaves_nothing_ablated_at_stage_a():
    assert nec.groups_outside(nec.NESTED_SETS["E"], "A") == []
    dropped = nec.groups_outside(nec.NESTED_SETS["A"], "A")
    assert "a_grav" in dropped and "orbital_energy" in dropped


def test_m1_gate_reports_one_number_per_active_group_in_range():
    p = net.init_params(spec.OBS_DIM_2D, seed=0)
    g = nec.m1_gate(p, "A")
    assert set(g) == set(nec.active_groups("A"))
    assert all(0.0 <= v <= 1.0 for v in g.values())


def test_m1_gate_tracks_the_underlying_logits():
    """A closed gate must read near 0 and an open one near 1."""
    p = net.init_params(spec.OBS_DIM_2D, seed=0)
    lo, hi, _ = spec.OBS_LAYOUT_2D["speed"]
    p["g"][lo:hi] = -10.0
    lo2, hi2, _ = spec.OBS_LAYOUT_2D["craft_vel"]
    p["g"][lo2:hi2] = +10.0
    g = nec.m1_gate(p, "A")
    assert g["speed"] < 0.01
    assert g["craft_vel"] > 0.99


def test_m3_saliency_is_normalised_to_a_peak_of_one():
    p = net.init_params(spec.OBS_DIM_2D, seed=1)
    p["W3"] = (np.random.default_rng(2).normal(size=p["W3"].shape)).astype(np.float32)
    obs = np.random.default_rng(3).normal(size=(50, spec.OBS_DIM_2D)).astype(np.float32)
    s = nec.m3_saliency(p, obs, "A")
    assert set(s) == set(nec.active_groups("A"))
    assert np.isclose(max(s.values()), 1.0)
    assert all(v >= 0.0 for v in s.values())


def test_m3_saliency_is_zero_for_a_closed_gate():
    """Saliency is chained through sigma(g), so a shut gate must read ~0 - this is
    what keeps M1 and M3 telling a consistent story."""
    p = net.init_params(spec.OBS_DIM_2D, seed=1)
    p["W3"] = (np.random.default_rng(2).normal(size=p["W3"].shape)).astype(np.float32)
    lo, hi, _ = spec.OBS_LAYOUT_2D["eccentricity_vec"]
    p["g"][lo:hi] = -20.0
    obs = np.random.default_rng(3).normal(size=(50, spec.OBS_DIM_2D)).astype(np.float32)
    s = nec.m3_saliency(p, obs, "A")
    assert s["eccentricity_vec"] < 1e-4


@pytest.mark.parametrize("dret,dsucc,expected", [
    (10.0, 0.50, "necessary"),
    (0.0, 0.30, "necessary"),     # success collapse alone is enough
    (2.0, 0.00, "marginal"),
    (0.0, 0.06, "marginal"),
    (0.1, 0.00, "unnecessary"),
    (-5.0, -0.1, "unnecessary"),  # ablation HELPED: certainly not necessary
])
def test_classification_thresholds(dret, dsucc, expected):
    assert nec.classify(dret, dsucc, span=10.0) == expected


def test_rank_stability_is_perfect_for_identical_rankings():
    r = ["a", "b", "c", "d"]
    st = nec.rank_stability([r, list(r), list(r)])
    assert st["pairs"] == 3
    assert st["min_spearman"] == 1.0
    assert st["stable"] is True


def test_rank_stability_detects_a_reversed_ranking():
    r = ["a", "b", "c", "d", "e"]
    st = nec.rank_stability([r, list(reversed(r))])
    assert st["min_spearman"] < 0
    assert st["stable"] is False


def test_rank_stability_needs_at_least_two_runs():
    st = nec.rank_stability([["a", "b"]])
    assert st["stable"] is False and st["pairs"] == 0


def test_m2_scores_shape_and_sign():
    """Zeroing a group that the policy needs should show a POSITIVE delta return."""
    groups = ["g1", "g2"]
    raw = {
        "baseline": [(10.0, 1.0, 0.2), (10.0, 1.0, 0.2)],
        "zero": {"g1": [(2.0, 0.0, 1.5), (2.0, 0.0, 1.5)],
                 "g2": [(9.9, 1.0, 0.21), (9.9, 1.0, 0.21)]},
        "resample": {"g1": [(3.0, 0.1, 1.4), (3.0, 0.1, 1.4)],
                     "g2": [(10.0, 1.0, 0.2), (10.0, 1.0, 0.2)]},
    }
    s = nec.m2_scores(raw, groups)
    assert s["baseline"]["return"] == 10.0
    assert s["per_group"]["g1"]["delta_return_zeroed"] == 8.0
    assert s["per_group"]["g1"]["delta_success_zeroed"] == 1.0
    assert abs(s["per_group"]["g2"]["delta_return_zeroed"]) < 0.2
    assert s["per_group"]["g1"]["delta_portal_dist_zeroed"] > 0
