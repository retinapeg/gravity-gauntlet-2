"""Policies. Everything implements the same `act(ctx) -> (action, logprob)` interface,
so the scripted controllers and the learned MLP (plan.md 3.1, `gg/net.py`) are
interchangeable everywhere - gates, fan-out, training and the necessity study.

`ctx` carries both the nondimensionalised observation vector (`ctx["obs"]`, what
the MLP will consume) and a few raw world-unit quantities the scripted controllers
find convenient.
"""
from __future__ import annotations

import numpy as np

from . import net
from .physics import COMPASS_2D


class NoThrust:
    """Coast forever - used for the energy-drift gate and the timeout terminal."""

    kind = "no_thrust"

    def act(self, ctx):
        return 0, 0.0


class PursuitPolicy:
    """Proportional pursuit: steer the velocity toward the portal by choosing the
    compass burn whose direction best matches the wanted velocity change, tapering
    the target speed near the portal and coasting once slow enough for the A7 gate
    to trigger. Deterministic, so logprob is reported as 0.
    """

    kind = "scripted_pursuit"

    def __init__(self, approach_gain: float = 1.5, coast_speed_frac: float = 0.85,
                 avoid_gain: float = 2.2, avoid_range: float = 1.1):
        self.approach_gain = approach_gain
        self.coast_speed_frac = coast_speed_frac
        self.avoid_gain = avoid_gain
        self.avoid_range = avoid_range

    def act(self, ctx):
        dr = ctx["portal_rel_pos"]                 # r_p - r, world units
        v = ctx["craft_vel"]
        dist = float(np.linalg.norm(dr)) + 1e-12
        to_portal = dr / dist

        # wanted velocity: aim at the portal, magnitude tapering as we close in,
        # capped at the episode's characteristic speed.
        v_des_mag = min(ctx["v_ref"], self.approach_gain * dist)
        dv = v_des_mag * to_portal - v

        # inside the capture bubble and already slow enough: coast and let A7 fire.
        closing = float(np.dot(v, to_portal))
        if dist < 1.3 * ctx["r_capture"] and abs(closing) < self.coast_speed_frac * ctx["v_capture"]:
            return 0, 0.0
        if float(np.linalg.norm(dv)) < 1e-3 * ctx["v_ref"]:
            return 0, 0.0

        want = dv / (float(np.linalg.norm(dv)) + 1e-12)

        # steer away from the nearest body, weighted by how close it is - keeps the
        # straight-line pursuit path from cutting through a gravity well.
        away = ctx.get("nearest_body_rel")         # r - r_body (world units)
        if away is not None:
            d_n = float(np.linalg.norm(away)) + 1e-12
            gate = float(np.exp(-(d_n / self.avoid_range) ** 2))
            want = want + self.avoid_gain * gate * (away / d_n)
            want = want / (float(np.linalg.norm(want)) + 1e-12)

        return int(np.argmax(COMPASS_2D @ want) + 1), 0.0


class MLPPolicy:
    """The learned policy (plan.md 3.1). Samples from Categorical(softmax(logits)) -
    that sampling IS stochasticity source S4, which is on from Stage A and is what
    makes M identical sandboxes produce M different trajectories (plan.md 2.6, 5.3).

    `greedy=True` takes the argmax instead, for deterministic champion evaluation.
    """

    kind = "mlp"

    def __init__(self, params: dict, seed: int = 0, greedy: bool = False):
        self.p = params
        self.rng = np.random.default_rng(int(seed))
        self.greedy = greedy

    def act(self, ctx):
        if self.greedy:
            return net.act_greedy(self.p, ctx["obs"])
        return net.act_sample(self.p, ctx["obs"], self.rng)


def make_policy(kind: str, weights_b64: str | None = None, seed: int = 0,
                greedy: bool = False):
    if kind == "no_thrust":
        return NoThrust()
    if kind == "mlp" or weights_b64:
        if not weights_b64:
            raise SystemExit('{"error": "policy_kind mlp requires policy_weights_b64"}')
        return MLPPolicy(net.unpack(weights_b64), seed=seed, greedy=greedy)
    if kind in ("scripted_pursuit", "scripted", None):
        return PursuitPolicy()
    raise SystemExit(f'{{"error": "unknown policy_kind {kind!r}"}}')
