"""One episode, one process (plan.md 4.2).

INPUT  : JSON on env var GG_INPUT, or on stdin.
OUTPUT : exactly one JSON line on stdout.

The browser renders from `trajectory` + `bodies_trajectory`; the controller learns
from `observations` / `actions` / `logprobs` / `rewards`. Daytona only calculates.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from . import spec
from .physics import (
    G,
    dpotential_dt,
    VEL_CLAMP_ESCAPE_MULT,
    accel_gravity,
    action_to_thrust,
    angular_momentum_2d,
    eccentricity_vec_2d,
    potential,
    specific_energy,
)
from .policy import make_policy
from .universe import build_universe
from .version import source_hash


# --------------------------------------------------------------------------- I/O

def read_input() -> dict:
    raw = os.environ.get("GG_INPUT") or sys.stdin.read()
    if not raw.strip():
        raise SystemExit('{"error": "no input on GG_INPUT or stdin"}')
    return json.loads(raw)


def episode_scales(u, r0, v0) -> dict:
    """Per-episode characteristic scales for nondimensionalisation (plan.md 2.2)."""
    L0 = float(np.linalg.norm(u.portal_pos - r0))
    g0 = float(np.linalg.norm(accel_gravity(r0, u.body_pos, u.body_mass)))
    if g0 < 1e-9:                                  # fallback: G M_tot / L0^2
        g0 = G * float(u.body_mass.sum()) / (L0 * L0)
    T0 = float(np.sqrt(L0 / g0))
    return dict(L0=L0, g0=g0, T0=T0, V0=L0 / T0, M0=float(u.body_mass.sum()))


# ----------------------------------------------------------------- observation

def build_observation(r, v, m, fuel, u, s, stage, t_frac, dphi_dt_world=0.0) -> np.ndarray:
    """Assemble the FROZEN 2D observation superset, nondimensionalised, zero-filling
    any group whose first-active stage is later than `stage` (plan.md 5.1)."""
    L0, g0, T0, V0, M0 = s["L0"], s["g0"], s["T0"], s["V0"], s["M0"]
    mu = G * M0
    obs = np.zeros(spec.OBS_DIM_2D)
    lay = spec.OBS_LAYOUT_2D

    dr_p = u.portal_pos - r
    dist_p = float(np.linalg.norm(dr_p)) + 1e-12
    to_p = dr_p / dist_p
    a_grav = accel_gravity(r, u.body_pos, u.body_mass)
    phi = potential(r, u.body_pos, u.body_mass)
    speed = float(np.linalg.norm(v))
    closing = float(np.dot(v, to_p))

    dbody = u.body_pos - r
    dbody_dist = np.linalg.norm(dbody, axis=1)
    order = np.argsort(dbody_dist)
    ni = int(order[0])
    rel_n, d_n = dbody[ni], float(dbody_dist[ni])
    rel_v_n = u.body_vel[ni] - v
    rr, rv = r - u.body_pos[ni], v - u.body_vel[ni]
    tca = max(0.0, float(-np.dot(rr, rv) / (np.dot(rv, rv) + 1e-12)))

    E = specific_energy(r, v, u.body_pos, u.body_mass)
    lz = angular_momentum_2d(r, v)
    ecc = eccentricity_vec_2d(r, v, mu)

    def put(name, vals):
        i, j, first = lay[name]
        if not spec.stage_active(first, stage):
            return
        arr = np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
        obs[i:j] = arr[: j - i]

    put("portal_rel_pos", dr_p / L0)
    put("craft_vel", v / V0)
    put("speed", speed / V0)
    put("closing_speed_portal", closing / V0)
    put("bearing_error",
        np.arctan2(v[0] * to_p[1] - v[1] * to_p[0], v[0] * to_p[0] + v[1] * to_p[1]) / np.pi)
    put("a_grav", a_grav / g0)
    put("a_grav_magdir", [np.linalg.norm(a_grav) / g0, np.arctan2(a_grav[1], a_grav[0]) / np.pi])
    put("potential", phi / (g0 * L0))
    put("nearest_body", [rel_n[0] / L0, rel_n[1] / L0, d_n / L0, u.body_mass[ni] / M0])
    put("tca_nearest", tca / T0)
    put("nearest_rel_vel", rel_v_n / V0)
    put("dphi_dt", dphi_dt_world / (g0 * L0 / T0))
    put("body_count", float(len(u.body_mass)))
    topk = []
    for idx in order[: spec.TOPK]:
        topk += [dbody[idx][0] / L0, dbody[idx][1] / L0, u.body_mass[idx] / M0]
    topk += [0.0] * (3 * spec.TOPK - len(topk))
    put("perbody_topk", topk)
    put("fuel_fraction", fuel / u.fuel0)
    put("craft_mass", m / u.craft_mass0)
    put("control_authority", u.a_thrust_mag / g0)
    put("time_fraction", t_frac)
    put("orbital_energy", E / (g0 * L0))
    put("angular_momentum", lz / (L0 * V0))
    put("eccentricity_vec", ecc)
    return obs


# ---------------------------------------------------------------------- episode

def run_episode(inp: dict) -> dict:
    t_start = time.time()
    stage = inp.get("stage", "A")
    dims = int(inp.get("dims", 2))
    if stage not in ("A", "B", "C", "D") or dims != 2:
        raise SystemExit(f'{{"error": "worker handles stages A-D in 2D; got {stage}/{dims}D"}}')

    want_hash = inp.get("obs_spec_hash")
    have_hash = spec.obs_spec_hash(dims)
    if want_hash and want_hash != have_hash:
        raise SystemExit(json.dumps({"error": "obs_spec_hash mismatch",
                                     "worker": have_hash, "controller": want_hash}))

    max_steps = int(inp.get("max_steps", 500))
    gamma = float(inp.get("gamma", 0.99))
    dt = float(inp.get("dt", 0.02))
    deterministic = bool(inp.get("deterministic", True))
    noise = inp.get("noise") or {}
    # S2 (plan.md 2.6). All three are OFF whenever `deterministic`, so stages A-C are
    # untouched no matter what the config carries.
    proc_sigma = 0.0 if deterministic else float(noise.get("process_sigma", 0.0))
    thrust_jitter = 0.0 if deterministic else float(noise.get("thrust_jitter", 0.0))
    point_sigma = 0.0 if deterministic else float(noise.get("pointing_sigma", 0.0))

    # The action-sampling seed is real even in the deterministic stages: S4 is what
    # makes M identical sandboxes disagree, so it is logged, not hidden (plan.md 3.4).
    action_seed = int(inp.get("seed") if inp.get("seed") is not None else 0)
    policy = make_policy(inp.get("policy_kind"), inp.get("policy_weights_b64"),
                         seed=action_seed, greedy=bool(inp.get("greedy", False)))

    # The key selects the universe instance: {} at A, {"layout_id": n} at B (plan.md 5.2).
    key = {k: inp[k] for k in ("layout_id", "orbital_phase", "seed_universe", "curriculum")
           if inp.get(k) is not None}
    u = build_universe(stage, key)
    ov = inp.get("overrides") or {}                # test hooks: craft_r0 / craft_v0 / a_thrust_mag
    r = np.asarray(ov.get("craft_r0", u.craft_r0), dtype=float).copy()
    v = np.asarray(ov.get("craft_v0", u.craft_v0), dtype=float).copy()
    if "a_thrust_mag" in ov:
        u.a_thrust_mag = float(ov["a_thrust_mag"])
    m = float(u.craft_mass0)
    fuel = float(u.fuel0)

    s = episode_scales(u, r, v)
    # The S2 noise stream. `worker_input` only ever sets `seed`, so defaulting this to
    # a constant would give EVERY rollout in a generation the identical noise sequence -
    # eight sandboxes drawing the same "random" disturbance, which is no disturbance at
    # all. Derive it from the episode seed, offset so the noise draw is not correlated
    # with the action draw that shares it.
    rng = np.random.default_rng(int(inp.get("rng_seed", action_seed + 5_150_000)))
    accel_fn = lambda rr: accel_gravity(rr, u.body_pos, u.body_mass)

    E0 = specific_energy(r, v, u.body_pos, u.body_mass)
    esc0 = float(np.sqrt(2.0 * abs(potential(r, u.body_pos, u.body_mass))))
    vmax = VEL_CLAMP_ESCAPE_MULT * max(esc0, s["V0"])

    R = spec.REWARD
    rho = R["proximity_rho_body_radii"] * float(u.body_radius.min())
    # Shaping potential for the A7 SPEED gate (spec.REWARD). A well around "slow and
    # near the portal" whose depth is the speed excess: sigma is set in capture radii
    # so braking pressure starts far enough out that a_thrust can actually shed the
    # speed before arrival - at a=0.30 dropping 0.3 of speed takes ~1.0 time unit.
    brake_sigma = R["brake_gate_radii"] * float(u.r_capture)
    def brake_potential(rr, vv):
        """Capture-readiness in [0, 1], maximal exactly at the A7 condition.

        It must never go NEGATIVE anywhere. A potential shaped as "penalty for
        arrival speed" is zero far away and negative near the portal, which makes
        staying away the best move: measured, that drove closest approach from 0.30
        out to 1.78 in 60 generations - the policy stopped approaching at all. Both
        factors positive keeps approach weakly uphill even while fast, and braking
        near the portal steeply uphill, so a monotone path to capture exists.
        """
        d = float(np.linalg.norm(u.portal_pos - rr))
        vrel = float(np.linalg.norm(vv - u.portal_vel))
        return (float(np.exp(-(d / brake_sigma) ** 2))
                * float(np.exp(-(vrel / u.v_capture) ** 2)))
    prev_brake_phi = brake_potential(r, v)
    prev_dist_p = float(np.linalg.norm(u.portal_pos - r))
    min_dist_p = prev_dist_p          # closest approach: the honest progress signal.
                                      # Mean reward is dominated by constant terminal
                                      # penalties, so it barely moves while the policy
                                      # is in fact learning to fly closer.

    # Necessity-study interventions (plan.md 6). `obs_ablate` zeroes whole groups;
    # `obs_resample` replaces a group's value with a draw from an empirical pool
    # collected from champion trajectories - that is the "shuffled across the batch"
    # arm of M2. Both act on the observation the POLICY sees; the physics is untouched.
    ablate_slices = [spec.OBS_LAYOUT_2D[g][:2] for g in (inp.get("obs_ablate") or [])]
    resample = {g: np.asarray(vals, dtype=float)
                for g, vals in (inp.get("obs_resample") or {}).items()}
    obs_rng = np.random.default_rng(action_seed + 90210)

    obs_l, act_l, logp_l, rew_l = [], [], [], []
    # [t, x, y, vx, vy, fuel] - the renderer needs velocity for ship rotation, and
    # carrying it beats making the adapter finite-difference positions (UI_SPEC 5).
    traj = [[0.0, float(r[0]), float(r[1]), float(v[0]), float(v[1]), round(fuel, 6)]]
    moving = u.orbit_omega is not None          # Stage C onward
    def bodies_row_at(bpos):
        return [[float(bp[0]), float(bp[1]), float(bm)]
                for bp, bm in zip(bpos, u.body_mass)]
    bodies_traj = [bodies_row_at(u.body_pos)]

    termination, success, done = "timeout", False, False

    for step in range(max_steps):
        t_frac = step / max_steps
        t_now = step * dt
        if moving:
            # Positions AND velocities at this instant: the velocities are what make
            # nearest_rel_vel and dPhi/dt real rather than zero (plan.md 5.2).
            u.body_pos, u.body_vel = u.bodies_at(t_now)
        dphi_dt = (dpotential_dt(r, u.body_pos, u.body_vel, u.body_mass)
                   if moving else 0.0)
        obs = build_observation(r, v, m, fuel, u, s, stage, t_frac, dphi_dt)
        for lo, hi in ablate_slices:
            obs[lo:hi] = 0.0
        for gname, pool in resample.items():
            lo, hi = spec.OBS_LAYOUT_2D[gname][:2]
            obs[lo:hi] = pool[obs_rng.integers(len(pool))]
        _nb = int(np.argmin(np.linalg.norm(u.body_pos - r, axis=1)))
        ctx = {"obs": obs, "portal_rel_pos": u.portal_pos - r, "craft_vel": v.copy(),
               "nearest_body_rel": r - u.body_pos[_nb],
               "v_ref": s["V0"], "r_capture": u.r_capture, "v_capture": u.v_capture}
        a, logp = policy.act(ctx)

        a_th = action_to_thrust(int(a), u.a_thrust_mag, dims)
        # S2 thrust misfire: the engine delivers (1+delta) of what was asked, pointed a
        # little off. Applied to the DELIVERED thrust but not to the fuel draw below -
        # a misfire wastes propellant, it does not refund it.
        if thrust_jitter > 0.0 or point_sigma > 0.0:
            mag = float(np.linalg.norm(a_th))
            if mag > 0.0:
                if thrust_jitter > 0.0:
                    a_th = a_th * max(0.0, 1.0 + float(rng.normal(0.0, thrust_jitter)))
                if point_sigma > 0.0:
                    phi = float(rng.normal(0.0, point_sigma))
                    cph, sph = np.cos(phi), np.sin(phi)
                    a_th = np.array([a_th[0] * cph - a_th[1] * sph,
                                     a_th[0] * sph + a_th[1] * cph])
        burn = float(np.linalg.norm(action_to_thrust(int(a), u.a_thrust_mag, dims)))
        d_fuel = 0.0
        if burn > 0.0 and fuel > 0.0:
            d_fuel = min(fuel, u.fuel_burn_c * burn * dt)     # discrete A5
            fuel -= d_fuel
        elif fuel <= 0.0:
            a_th = np.zeros(dims)

        pnoise = rng.normal(0.0, proc_sigma, dims) if proc_sigma > 0.0 else None
        # With moving bodies the half-steps must see the field at their own instants,
        # or the integrator silently solves a different problem than the one posed.
        a0 = accel_gravity(r, u.body_pos, u.body_mass) + a_th
        v_half = v + 0.5 * dt * a0
        r = r + dt * v_half
        if moving:
            u.body_pos, u.body_vel = u.bodies_at(t_now + dt)
        a1 = accel_gravity(r, u.body_pos, u.body_mass) + a_th
        v = v_half + 0.5 * dt * a1
        if pnoise is not None:
            v = v + pnoise
        sp = float(np.linalg.norm(v))
        if sp > vmax:
            v *= vmax / sp

        dist_p = float(np.linalg.norm(u.portal_pos - r))
        min_dist_p = min(min_dist_p, dist_p)
        d_n = float(np.min(np.linalg.norm(u.body_pos - r, axis=1)))
        prog = float(np.clip(R["k_progress"] * (prev_dist_p - dist_p) / s["L0"],
                             -R["progress_clip"], R["progress_clip"]))
        prox = -R["k_proximity"] * float(np.exp(-(d_n / rho) ** 2))
        v_rel_portal = float(np.linalg.norm(v - u.portal_vel))
        # Difference of a potential, exactly like `prog`: it telescopes over any closed
        # loop, so approaching fast and braking cannot be farmed for repeat credit, and
        # the optimal policy is unchanged (Ng et al. 1999) - it only makes the second
        # half of the capture condition visible to the gradient.
        brake_phi = brake_potential(r, v)
        brake = float(np.clip(R["k_brake"] * (brake_phi - prev_brake_phi),
                              -R["brake_clip"], R["brake_clip"]))
        prev_brake_phi = brake_phi
        rew = prog + prox + brake - R["k_fuel"] * (d_fuel / u.fuel0) - R["k_time"]
        prev_dist_p = dist_p

        hit = bool(np.any(np.linalg.norm(u.body_pos - r, axis=1) < u.body_radius))
        if dist_p < u.r_capture and v_rel_portal < u.v_capture:
            rew += R["r_goal"]; termination, success, done = "portal", True, True
        elif hit:
            rew += -R["r_crash"]; termination, done = "collision", True
        elif float(np.linalg.norm(r)) > u.arena_radius:
            rew += -R["r_escape"]; termination, done = "escape", True
        elif step == max_steps - 1:
            rew += -R["r_timeout"]; termination, done = "timeout", True

        obs_l.append([float(x) for x in obs])
        act_l.append(int(a))
        logp_l.append(float(logp))
        rew_l.append(float(rew))
        traj.append([round((step + 1) * dt, 6), float(r[0]), float(r[1]),
                     float(v[0]), float(v[1]), round(fuel, 6)])
        bodies_traj.append(bodies_row_at(u.body_pos))   # varies from Stage C on

        if done:
            break

    Ef = specific_energy(r, v, u.body_pos, u.body_mass)
    energy_drift = abs(Ef - E0) / (abs(E0) + 1e-12)

    g_t, disc = 0.0, []
    for rr in reversed(rew_l):
        g_t = rr + gamma * g_t
        disc.append(g_t)
    disc.reverse()

    checksum = hashlib.sha1(
        np.round(np.array([row[1:3] for row in traj]), 4).tobytes()
    ).hexdigest()[:8]

    return {
        "sandbox_id": os.environ.get("DAYTONA_SANDBOX_ID"),
        "stage": stage, "layout_id": getattr(u, "layout_id", inp.get("layout_id")),
        "orbital_phase": getattr(u, "orbital_phase", None) if stage == "C" else None,
        "seed_universe": getattr(u, "seed_universe", None),
        "curriculum": inp.get("curriculum"),
        "seed": action_seed,
        "policy_version": inp.get("policy_version"), "dims": dims,
        "policy_kind": policy.kind,
        "reward": round(float(sum(rew_l)), 4),
        "return_g0": round(float(disc[0]) if disc else 0.0, 4),
        "success": bool(success),
        "termination": termination,
        "steps": len(rew_l),
        "fuel_left": round(float(fuel), 6),
        "min_portal_dist": round(float(min_dist_p), 6),
        "energy_e0": E0, "energy_ef": Ef, "energy_drift": energy_drift,
        "obs_spec_hash": have_hash, "obs_dim": spec.OBS_DIM_2D,
        "gg_source_hash": source_hash(),
        "observations": obs_l,
        "actions": act_l,
        "logprobs": logp_l,
        "rewards": [round(x, 6) for x in rew_l],
        "trajectory": traj,
        "bodies_trajectory": bodies_traj,
        "wallclock_ms": round((time.time() - t_start) * 1000.0, 1),
        "physics_checksum": checksum,
    }


def main():
    out = run_episode(read_input())
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
