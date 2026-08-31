"""Build-order step 1 gate, run locally (no sandbox) for fast iteration.

Gate (plan.md 10.2):
  * energy drift < 1% over 500 steps with thrust off
  * the collision terminal fires
  * the capture (portal) terminal fires
Also exercises the timeout and escape terminals (step A2 asks for all four to be
reachable from hand-built states) and the frozen obs-spec hash.
"""
from __future__ import annotations

import numpy as np

from gg import spec
from gg.rollout_worker import run_episode

BASE = dict(stage="A", dims=2, deterministic=True, max_steps=500, gamma=0.99, dt=0.02,
            obs_spec_hash=spec.obs_spec_hash(2))


def _run(**over):
    inp = dict(BASE)
    inp.update(over)
    return run_episode(inp)


def main() -> int:
    fails = []

    # 1. thrust-off: energy conservation + timeout terminal --------------------
    coast = _run(policy_kind="no_thrust")
    drift = coast["energy_drift"]
    print(f"[coast]     steps={coast['steps']:3d}  term={coast['termination']:9s}  "
          f"E0={coast['energy_e0']:+.6f}  Ef={coast['energy_ef']:+.6f}  drift={drift:.3e}")
    if not (drift < 0.01):
        fails.append(f"energy drift {drift:.3e} >= 1%")
    if coast["termination"] != "timeout" or coast["steps"] != 500:
        fails.append(f"expected 500-step timeout, got {coast['steps']} / {coast['termination']}")

    # 2. collision terminal --------------------------------------------------
    crash = _run(policy_kind="no_thrust", overrides={"craft_v0": [-0.55, 0.0]})
    print(f"[crash]     steps={crash['steps']:3d}  term={crash['termination']:9s}  "
          f"reward={crash['reward']:+.2f}")
    if crash["termination"] != "collision":
        fails.append(f"expected collision, got {crash['termination']}")

    # 3. escape terminal -----------------------------------------------------
    # Hand-built state: radially outward at 2.0, well over the local escape speed
    # (sqrt(2GM/r) = 0.89 at r = 2.5) but under the |v| clamp, so the craft leaves
    # the arena instead of orbiting. This is the fourth terminal the A2 gate wants.
    esc = _run(policy_kind="no_thrust", overrides={"craft_v0": [2.0, 0.0]})
    print(f"[escape]    steps={esc['steps']:3d}  term={esc['termination']:9s}  "
          f"reward={esc['reward']:+.2f}")
    if esc["termination"] != "escape":
        fails.append(f"expected escape, got {esc['termination']}")

    # 4. capture terminal --------------------------------------------------
    cap = _run(policy_kind="scripted_pursuit")
    print(f"[pursuit]   steps={cap['steps']:3d}  term={cap['termination']:9s}  "
          f"success={cap['success']}  reward={cap['reward']:+.2f}  "
          f"return_g0={cap['return_g0']:+.2f}  fuel_left={cap['fuel_left']:.3f}")
    if cap["termination"] != "portal" or not cap["success"]:
        fails.append(f"pursuit did not capture: {cap['termination']}")

    # shape / spec sanity --------------------------------------------------
    d = len(cap["observations"][0])
    print(f"[spec]      obs_dim={d}  hash={cap['obs_spec_hash']}  "
          f"checksum={cap['physics_checksum']}  actions in {sorted(set(cap['actions']))}")
    if d != spec.OBS_DIM_2D:
        fails.append(f"obs dim {d} != {spec.OBS_DIM_2D}")
    if np.any(~np.isfinite(np.array(cap["observations"]))):
        fails.append("non-finite values in observations")

    print()
    if fails:
        print("GATE: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("GATE: PASS - energy drift < 1%, all four terminals fire "
          "(portal / collision / escape / timeout), obs spec frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
