"""Worker output (plan.md 4.2) -> UI rollout payload (web/UI_SPEC.md 5).

One function, no logic beyond renaming and reshaping. The worker contract is the
source of truth; the frontend adapts to it, never the reverse. If you find yourself
computing something here, it belongs in the worker.
"""
from __future__ import annotations

from gg.universe import build_universe

# web/UI_SPEC.md 5: termination -> card status
STATUS = {"portal": "success", "collision": "crashed",
          "timeout": "timeout", "escape": "escaped"}


def environment_of(stage: str, key: dict | None = None) -> dict:
    """The static world, in world units, straight off the Universe dataclass.
    `radius` is the REAL collision radius - the renderer must not guess it."""
    u = build_universe(stage, key)
    return {
        "planets": [
            {"x": float(bp[0]), "y": float(bp[1]), "mass": float(bm), "radius": float(br)}
            for bp, bm, br in zip(u.body_pos, u.body_mass, u.body_radius)
        ],
        "portal": {"x": float(u.portal_pos[0]), "y": float(u.portal_pos[1]),
                   "r_capture": float(u.r_capture), "v_capture": float(u.v_capture)},
        "ship_start": {"x": float(u.craft_r0[0]), "y": float(u.craft_r0[1]),
                       "vx": float(u.craft_v0[0]), "vy": float(u.craft_v0[1])},
        "obstacles": [],
    }


def adapt_rollout(out: dict, worker_id: int, generation: int,
                  key: dict | None = None) -> dict:
    """One worker payload -> one UI world card + its replayable trajectory."""
    traj, actions, rewards = out["trajectory"], out["actions"], out["rewards"]
    n = len(actions)

    # Row i is the state BEFORE action i, so the final row has no action or reward.
    steps = []
    for i, row in enumerate(traj):
        t, x, y, vx, vy = row[0], row[1], row[2], row[3], row[4]
        steps.append({"t": t, "x": x, "y": y, "vx": vx, "vy": vy,
                      "action": actions[i] if i < n else None,
                      "reward": rewards[i] if i < n else None})

    return {
        "worker_id": worker_id,
        "sandbox_id": out.get("sandbox_id"),
        "generation": generation,
        "policy_id": f"P{int(out.get('policy_version') or 0):03d}",
        "seed": out.get("seed"),
        "key": key if key is not None else {},
        "status": STATUS.get(out["termination"], out["termination"]),
        "termination": out["termination"],
        "success": bool(out["success"]),
        "reward": out["reward"],
        "steps": out["steps"],
        "fuel_left": out.get("fuel_left"),
        "wallclock_ms": out.get("wallclock_ms"),
        "trajectory": steps,
        "bodies_trajectory": out.get("bodies_trajectory"),
        "environment": environment_of(out.get("stage", "A"), key),
    }


def generation_payload(generation: int, policy_version: int, base_seed: int,
                       keys: list, rollouts: list[dict], failures: list | None = None) -> dict:
    """The generation envelope the UI reads in both live and replay mode
    (web/UI_SPEC.md 5). Means are over SUCCESSFUL rollouts only - a dropped
    sandbox must not be scored as a zero (plan.md 4.5)."""
    rewards = [r["reward"] for r in rollouts]
    successes = [r for r in rollouts if r["success"]]
    champion = max(range(len(rollouts)), key=lambda i: rollouts[i]["reward"]) if rollouts else None
    return {
        "generation": generation,
        "policy_id": f"P{policy_version:03d}",
        "base_seed": base_seed,
        "keys": keys,
        "rollouts": rollouts,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "best_reward": round(max(rewards), 4) if rewards else None,
        "success_rate": round(len(successes) / len(rollouts), 4) if rollouts else 0.0,
        "champion_worker": champion,
        "n_dispatched": len(rollouts) + len(failures or []),
        "n_failed": len(failures or []),
        "failures": failures or [],
    }
