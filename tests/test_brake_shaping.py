"""The A7 speed gate has a matching shaping term (spec.REWARD k_brake).

Stage C measured a champion that reached inside r_capture on 41 of 48 held-out
episodes and was refused by the speed gate on every one of them (0.76-0.96 against
v_capture 0.6): it had learned to fly to the portal and not to brake, because
`k_progress` pays for closing distance and is indifferent to arrival speed.

These tests pin the two properties the fix has to have - it must reward braking, and
it must not be farmable by oscillating.
"""
from __future__ import annotations

import numpy as np

from gg import spec
from gg.universe import build_universe

R = spec.REWARD


def phi(u, r, v):
    """The potential as gg/rollout_worker.py builds it."""
    sigma = R["brake_gate_radii"] * float(u.r_capture)
    d = float(np.linalg.norm(u.portal_pos - np.asarray(r, dtype=float)))
    vrel = float(np.linalg.norm(np.asarray(v, dtype=float) - u.portal_vel))
    return (float(np.exp(-(d / sigma) ** 2))
            * float(np.exp(-(vrel / u.v_capture) ** 2)))


def shaping(u, states):
    """Total un-clipped brake shaping along a state sequence [(r, v), ...]."""
    return R["k_brake"] * sum(phi(u, *b) - phi(u, *a)
                              for a, b in zip(states, states[1:]))


def test_braking_at_the_portal_pays():
    u = build_universe("A")
    at_portal = u.portal_pos + np.array([0.05, 0.0])
    fast, slow = [0.0, 0.9], [0.0, 0.4]
    assert shaping(u, [(at_portal, fast), (at_portal, slow)]) > 0.0


def test_accelerating_at_the_portal_costs():
    u = build_universe("A")
    at_portal = u.portal_pos + np.array([0.05, 0.0])
    assert shaping(u, [(at_portal, [0.0, 0.4]), (at_portal, [0.0, 0.9])]) < 0.0


def test_speed_far_from_the_portal_is_not_shaped():
    """The term must not become a global speed limit - it is about ARRIVAL speed."""
    u = build_universe("A")
    far = u.portal_pos + np.array([6.0, 0.0])
    assert abs(shaping(u, [(far, [0.0, 0.9]), (far, [0.0, 0.2])])) < 1e-3


def test_a_closed_loop_earns_nothing():
    """The anti-farm property claimed in spec.REWARD: approach fast, brake, retreat,
    speed up, come back - a cycle through the same states must telescope to zero,
    or the policy can pump the shaping instead of flying to the portal."""
    u = build_universe("A")
    near = u.portal_pos + np.array([0.10, 0.0])
    mid = u.portal_pos + np.array([1.20, 0.0])
    loop = [(mid, [0.0, 0.9]), (near, [0.0, 0.9]), (near, [0.0, 0.3]),
            (mid, [0.0, 0.3]), (mid, [0.0, 0.9])]
    assert abs(shaping(u, loop)) < 1e-9


def test_the_well_is_wide_enough_to_brake_in():
    """Sizing check. Shedding the measured Stage C excess (~0.35) at a_thrust takes
    ~1.2 time units, i.e. roughly a length unit of travel - so the potential must
    still have real gradient a length unit out, not just inside r_capture."""
    u = build_universe("A")
    excess = 0.35
    t_brake = excess / u.a_thrust_mag
    lead = t_brake * u.v_capture           # distance covered while braking
    near = u.portal_pos + np.array([lead, 0.0])
    gain = shaping(u, [(near, [0.0, 0.95]), (near, [0.0, 0.60])])
    assert gain > 0.1, f"braking a lead-distance out barely pays: {gain:.4f}"


def test_the_potential_is_never_negative():
    """The regression that cost a 60-generation run: a potential that is 0 far away
    and negative near the portal makes NOT approaching optimal, and the policy duly
    stopped approaching (closest approach drifted 0.30 -> 1.78). Every reachable
    state must score >= 0 so that closing distance is never punished on its own."""
    u = build_universe("A")
    rng = np.random.default_rng(0)
    for _ in range(2000):
        r = u.portal_pos + rng.normal(0, 3.0, 2)
        v = rng.normal(0, 2.0, 2)
        assert phi(u, r, v) >= 0.0


def test_approaching_at_speed_still_beats_staying_away():
    """The direct statement of that failure: flying in fast must not be worse than
    loitering out of range, or `prog` and the brake term fight each other."""
    u = build_universe("A")
    far = u.portal_pos + np.array([5.0, 0.0])
    near_fast = u.portal_pos + np.array([0.10, 0.0])
    fast = [0.0, 0.90]
    assert shaping(u, [(far, fast), (near_fast, fast)]) > 0.0


def test_the_capture_state_is_the_global_maximum():
    u = build_universe("A")
    best = phi(u, u.portal_pos, u.portal_vel)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        r = u.portal_pos + rng.normal(0, 3.0, 2)
        v = rng.normal(0, 2.0, 2)
        assert phi(u, r, v) <= best + 1e-12


def test_the_clip_does_not_bind_at_the_configured_strength():
    """`k_brake` and `brake_clip` have to be chosen together. Measured over 4000 steps
    of real Stage C trajectories the largest per-step |dPhi| is ~0.007, so a clip below
    k_brake * that binds on a large fraction of steps - 6.9% at k_brake=40 with the old
    0.10 clip - and a shaping term that is clipped most of the time no longer telescopes,
    which is the whole basis of the anti-farm property above."""
    max_step_dphi = 0.007          # measured, gg Stage C champion trajectories
    assert R["brake_clip"] > R["k_brake"] * max_step_dphi, (
        f"brake_clip {R['brake_clip']} binds at k_brake {R['k_brake']}: raise the clip")
