"""Evaluate a stage champion on its HELD-OUT keys (the gate number).

Separated from `train.py` so a gate can be re-checked without retraining - which
matters, because a training run that is interrupted (or whose snapshot goes stale
mid-flight) still leaves a perfectly good champion behind.

    python -m controller.evaluate --stage A
    python -m controller.evaluate --stage B --episodes 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
from daytona import AsyncDaytona

from controller.build_snapshot import ensure_snapshot
from controller.config import (daytona_config, load_stage_config, load_stage_keys,
                               results_dir)
from controller.fanout import SandboxPool, worker_input
from controller.train import champion_meta, load_champion
from gg import net


def summarise(outs: list[dict]) -> dict:
    if not outs:
        return {"n": 0, "success_rate": None, "mean_reward": None}
    return {
        "n": len(outs),
        "success_rate": round(float(np.mean([bool(o["success"]) for o in outs])), 4),
        "mean_reward": round(float(np.mean([o["reward"] for o in outs])), 4),
        "mean_min_portal_dist": round(float(np.mean([o["min_portal_dist"] for o in outs])), 4),
        "terminations": {t: sum(o["termination"] == t for o in outs)
                         for t in sorted({o["termination"] for o in outs})},
    }


async def evaluate(stage: str, episodes: int, pool_size: int, seed: int = 777_000) -> dict:
    cfg = load_stage_config(stage)
    keyfile = load_stage_keys(stage)
    eval_sets = keyfile["eval_sets"] or {"eval_held_out": keyfile["eval_held_out"]}
    champ = results_dir(stage) / "champion.npz"
    if not champ.exists():
        raise SystemExit(f"no champion at {champ} - train the stage first")
    weights = net.pack(load_champion(champ))

    out = {"stage": stage, "champion": champion_meta(champ),
           "eval_sets": {k: v for k, v in eval_sets.items()},
           "episodes_per_mode": episodes, "by_set": {}}
    async with AsyncDaytona(daytona_config()) as d:
        await ensure_snapshot(d, quiet=True)
        async with SandboxPool(d, pool_size) as pool:
            for set_name, keys in eval_sets.items():
                out["by_set"][set_name] = {}
                for mode, greedy in (("sampled", False), ("greedy", True)):
                    pl = [worker_input(stage, cfg, weights_b64=weights,
                                       key=keys[i % len(keys)], seed=seed + i,
                                       policy_version=0, greedy=greedy)
                          for i in range(episodes)]
                    outs, bad = await pool.map(pl, f"eval-{set_name}-{mode}")
                    r = summarise(outs)
                    r["failed"] = len(bad)
                    if bad:
                        r["first_error"] = bad[0]["error"][:200]
                    out["by_set"][set_name][mode] = r
    # the headline pair stays at the top level: the union of every held-out set
    prim = max(out["by_set"], key=lambda k: len(eval_sets[k]))
    out["sampled"] = out["by_set"][prim]["sampled"]
    out["greedy"] = out["by_set"][prim]["greedy"]
    out["primary_set"] = prim
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="evaluate a stage champion on held-out keys")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--bar", type=float, default=0.8, help="success bar for 'reliably'")
    a = ap.parse_args()

    res = asyncio.run(evaluate(a.stage, a.episodes, a.pool))
    path = results_dir(a.stage) / "heldout_eval.json"
    path.write_text(json.dumps(res, indent=2))

    print(f"\nStage {res['stage']} champion "
          f"(gen {res['champion']['generation']}, {res['champion']['kind']} "
          f"score {res['champion']['score']})")
    for set_name, modes in res["by_set"].items():
        print(f"  [{set_name}]  keys={res['eval_sets'][set_name]}")
        for mode in ("sampled", "greedy"):
            m = modes[mode]
            print(f"    {mode:8s} n={m['n']:3d}  success={m['success_rate']}  "
                  f"reward={m['mean_reward']}  closest={m.get('mean_min_portal_dist')}  "
                  f"{m['terminations']}")
    ok = (res["sampled"]["success_rate"] or 0.0) >= a.bar
    print(f"\n  [{'PASS' if ok else 'FAIL'}] champion reaches the portal reliably "
          f"(held-out, sampled >= {a.bar})")
    print("  note: the policy IS the sampling distribution (S4 on from Stage A), so "
          "'sampled' is the deployed policy; greedy is a diagnostic of how peaked it is.")
    print(f"\nwrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
