"""One generation = M concurrent Daytona sandboxes (plan.md 4.3, 4.5).

`asyncio.gather` over M rollouts with `return_exceptions=True`. A sandbox that
errors or times out is DROPPED from the batch - never allowed to put a NaN into the
gradient - and the generation is only usable if at least `min_frac` of the rollouts
came back. That is plan.md 4.5, and it is the difference between a training run that
degrades gracefully and one that silently poisons its own gradient.

Two dispatch modes:

  fresh  - one sandbox per rollout, created and deleted around it. The literal
           demo story: M cards, M real sandbox ids, nothing reused.
  pooled - N sandboxes kept warm, rollouts distributed across them. Sanctioned by
           plan.md 4.6 ("keep a warm pool") and used for the necessity sweeps,
           where the rollout count is in the hundreds and create latency dominates.

    python -m controller.fanout --stage A -n 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

from controller.adapter import adapt_rollout, generation_payload
from controller.build_snapshot import ensure_snapshot
from controller.config import SNAPSHOT, daytona_config, load_stage_config, load_stage_keys
from controller.record import write_generation
from gg import net, spec
from gg.version import source_hash

WORKER_CMD = "python -m gg.rollout_worker"
EXEC_TIMEOUT = 180


TRAIN_SEED_CEILING = 800_000   # stage D: eval seeds live at 900000+, never below this


def sample_keys(train_keys: list, n: int, rng, distinct: int | None = None) -> list:
    """n keys for one generation, DISTINCT wherever the key set allows it.

    plan.md 7.1 is emphatic: a generation dispatches n distinct keys, not n copies of
    one - that is what makes the demo grid eight different universes instead of eight
    identical cards. Stage A has a single key so this degenerates to repetition, which
    is expected there and is exactly why Stage A is the calibration rung. When n
    exceeds the pool, deal successive shuffled passes so every key appears an equal
    number of times and none is over-represented.
    """
    # `distinct` caps how many DIFFERENT worlds a generation draws, then tiles them
    # across the batch. Stage D needs this: its keys are fresh seeds, so without it a
    # 32-episode batch is 32 different universes seen once each, and the per-key
    # advantage centring in Trainer.update degenerates to a batch-wide baseline over
    # worlds of wildly different difficulty - the advantage then encodes "was this an
    # easy draw", not "was this a good action". Eight worlds x four flights keeps the
    # eight-distinct-worlds fleet story AND gives each world a real baseline.
    if distinct is not None and 0 < distinct < n:
        base = sample_keys(train_keys, distinct, rng)
        return [dict(base[i % distinct]) for i in range(n)]

    out = []
    while len(out) < n:
        deck = list(train_keys)
        rng.shuffle(deck)
        out.extend(deck[: n - len(out)])
    # Stage C writes "orbital_phase": "sampled" - a placeholder meaning "draw a fresh
    # phase every time", so a repeated layout is never the same world twice. Stage D
    # writes "seed_universe": "sampled" and needs an INT, drawn from a block disjoint
    # from the frozen eval seeds at 900000+ so a training draw can never land on an
    # eval world (stages/stage_d_stochastic/keys.json).
    def resolve(name, value):
        if value != "sampled":
            return value
        if name == "seed_universe":
            return int(rng.integers(1, TRAIN_SEED_CEILING))
        return float(rng.random())

    return [{k: resolve(k, v) for k, v in key.items()} for key in out]


def worker_input(stage: str, cfg: dict, *, weights_b64=None, key=None, seed=0,
                 policy_version=0, policy_kind="mlp", greedy=False, **extra) -> dict:
    """Splice stage config.json into the worker contract (plan.md 4.2)."""
    ep = cfg["episode"]
    payload = {
        "stage": stage, "dims": cfg["dims"], "deterministic": cfg["deterministic"],
        "max_steps": ep["max_steps"], "dt": ep["dt"], "gamma": ep["gamma"],
        "noise": cfg["noise"],
        "obs_spec_hash": spec.obs_spec_hash(cfg["dims"]),
        "policy_kind": policy_kind, "policy_weights_b64": weights_b64,
        "policy_version": policy_version, "seed": int(seed), "greedy": bool(greedy),
    }
    payload.update(key or {})
    payload.update(extra)
    return payload


def _parse_worker_line(result: str, label: str) -> dict:
    lines = [ln for ln in (result or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"{label}: worker printed no JSON line:\n{result[-800:]}")
    return json.loads(lines[-1])


def _check_not_stale(out: dict) -> None:
    """Refuse a rollout produced by a stale baked copy of `gg` (see gg/version.py)."""
    remote, local = out.get("gg_source_hash"), source_hash()
    if remote and remote != local:
        raise RuntimeError(
            f"worker gg_source_hash {remote} != controller {local} - the snapshot is "
            f"stale. Rebuild: python -m controller.build_snapshot --rebuild --verify"
        )


async def exec_rollout(sbx, payload: dict, label: str = "rollout") -> dict:
    res = await sbx.process.exec(WORKER_CMD, cwd="/opt",
                                 env={"GG_INPUT": json.dumps(payload)},
                                 timeout=EXEC_TIMEOUT)
    if res.exit_code != 0:
        raise RuntimeError(f"{label}: worker exit {res.exit_code}:\n{(res.result or '')[-800:]}")
    out = _parse_worker_line(res.result, label)
    if "error" in out:
        raise RuntimeError(f"{label}: worker error {out['error']}")
    out["sandbox_id"] = out.get("sandbox_id") or sbx.id
    _check_not_stale(out)
    return out


async def run_rollout(d: AsyncDaytona, payload: dict, worker_id: int, on_event=None) -> dict:
    """One rollout in its own fresh sandbox, with the plan.md 4.4 lifecycle events."""
    emit = on_event or (lambda **kw: None)
    emit(worker_id=worker_id, phase="creating", seed=payload.get("seed"))
    sbx = await d.create(CreateSandboxFromSnapshotParams(
        snapshot=SNAPSHOT, auto_stop_interval=0, auto_delete_interval=5,
    ))
    emit(worker_id=worker_id, phase="live", sandbox_id=sbx.id)
    try:
        emit(worker_id=worker_id, phase="running", sandbox_id=sbx.id)
        out = await exec_rollout(sbx, payload, f"worker{worker_id}")
        emit(worker_id=worker_id, phase="finished", sandbox_id=sbx.id,
             reward=out["reward"], termination=out["termination"], success=out["success"])
        return out
    except Exception as e:
        emit(worker_id=worker_id, phase="error", sandbox_id=sbx.id, error=str(e)[:200])
        raise
    finally:
        await d.delete(sbx, wait=False)


async def gather_rollouts(d: AsyncDaytona, payloads: list[dict], on_event=None):
    """M rollouts, M sandboxes, all at once. Returns (rollouts, failures)."""
    results = await asyncio.gather(
        *(run_rollout(d, p, i, on_event) for i, p in enumerate(payloads)),
        return_exceptions=True,
    )
    rollouts, failures = [], []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            failures.append({"worker_id": i, "error": str(r)[:300]})
        else:
            rollouts.append((i, r))
    return rollouts, failures


# ------------------------------------------------------------------ warm pool

class SandboxPool:
    """N sandboxes kept alive while many rollouts stream through them (plan.md 4.6).

    Used by the necessity study, where the rollout count runs into the hundreds and
    paying 1-3 s of create latency per episode would dominate the wall clock.
    """

    def __init__(self, d: AsyncDaytona, size: int):
        self.d, self.size = d, size
        self.sandboxes: list = []
        self._free: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self):
        self.sandboxes = list(await asyncio.gather(*(
            # auto_stop_interval must NOT be 0 here. A pool outlives many rollouts,
            # so if the controller is killed mid-run these are orphaned - and with
            # "never auto-stop" they never become eligible for auto-delete either,
            # holding org CPU/memory quota until reaped by hand. 15 min of idle is
            # far longer than any gap between rollouts in a live run.
            self.d.create(CreateSandboxFromSnapshotParams(
                snapshot=SNAPSHOT, auto_stop_interval=15, auto_delete_interval=5))
            for _ in range(self.size)
        )))
        for sbx in self.sandboxes:
            self._free.put_nowait(sbx)
        return self

    async def __aexit__(self, *exc):
        await asyncio.gather(*(self.d.delete(s, wait=False) for s in self.sandboxes),
                             return_exceptions=True)

    async def run(self, payload: dict, label: str = "pooled") -> dict:
        sbx = await self._free.get()
        try:
            return await exec_rollout(sbx, payload, label)
        finally:
            self._free.put_nowait(sbx)

    async def map(self, payloads: list[dict], label: str = "pooled"):
        """All payloads through the pool. Returns (results, failures), failures dropped."""
        res = await asyncio.gather(*(self.run(p, f"{label}{i}") for i, p in enumerate(payloads)),
                                   return_exceptions=True)
        ok, bad = [], []
        for i, r in enumerate(res):
            (bad if isinstance(r, BaseException) else ok).append(
                {"i": i, "error": str(r)[:300]} if isinstance(r, BaseException) else r)
        return ok, bad


# ------------------------------------------------------------ one generation

async def run_generation(d: AsyncDaytona, *, stage: str, cfg: dict, weights_b64: str,
                         policy_version: int, keys: list, base_seed: int,
                         on_event=None, record: bool = True, generation: int = 1):
    """Dispatch one generation and return (payload, raw_rollouts, failures).

    NOTE at Stage A: `keys` is `[{}] * M` - one universe, so the M sandboxes run the
    SAME world and differ only by action sampling (S4). That is expected and is still
    a variance reduction (plan.md 5.3); visible world variety starts at Stage B.
    """
    m = len(keys)
    payloads = [
        worker_input(stage, cfg, weights_b64=weights_b64, key=keys[i],
                     seed=base_seed + i, policy_version=policy_version)
        for i in range(m)
    ]
    t0 = time.time()
    indexed, failures = await gather_rollouts(d, payloads, on_event)
    wall = round((time.time() - t0) * 1000.0, 1)

    raw = [out for _, out in indexed]
    adapted = [adapt_rollout(out, wid, generation, keys[wid]) for wid, out in indexed]
    payload = generation_payload(generation, policy_version, base_seed, keys, adapted, failures)
    payload["wallclock_ms"] = wall
    payload["sandbox_ids"] = [r["sandbox_id"] for r in adapted]
    if record:
        payload["recorded_to"] = str(write_generation(payload))
    return payload, raw, failures


async def _main(args) -> int:
    import numpy as _np
    cfg = load_stage_config(args.stage)
    keys_file = load_stage_keys(args.stage)
    keys = sample_keys(keys_file["train"], args.n, _np.random.default_rng(args.seed))

    if args.champion:
        from controller.train import load_champion
        from controller.config import results_dir
        cp = results_dir(args.stage) / "champion.npz"
        if not cp.exists():
            raise SystemExit(f"no champion at {cp}")
        params = load_champion(cp)
        label = f"champion {cp}"
    else:
        params = net.init_params(spec.OBS_DIM_2D, seed=args.seed)
        label = "random-init MLP"
    weights = net.pack(params)
    print(f"[fanout] stage {args.stage}  M={args.n}  policy={label} "
          f"({net.n_params(params)} params, {len(weights)/1024:.1f} KB packed)")

    def on_event(**kw):
        print("  event", json.dumps(kw), flush=True)

    async with AsyncDaytona(daytona_config()) as d:
        await ensure_snapshot(d, quiet=True)
        payload, raw, failures = await run_generation(
            d, stage=args.stage, cfg=cfg, weights_b64=weights, policy_version=0,
            keys=keys, base_seed=args.seed, on_event=on_event, generation=args.generation,
        )

    ids = payload["sandbox_ids"]
    print("\n" + "=" * 72)
    print(f"generation {payload['generation']}  policy {payload['policy_id']}  "
          f"mean_reward={payload['mean_reward']}  best={payload['best_reward']}  "
          f"success_rate={payload['success_rate']}  wall={payload['wallclock_ms']:.0f}ms")
    for r in payload["rollouts"]:
        print(f"  worker {r['worker_id']}  {r['sandbox_id']}  {r['status']:8s} "
              f"steps={r['steps']:3d}  reward={r['reward']:+8.2f}  seed={r['seed']}")
    for f in failures:
        print(f"  worker {f['worker_id']}  FAILED  {f['error'][:120]}")
    print("=" * 72)

    checks = {
        f"{args.n} rollouts returned": len(payload["rollouts"]) == args.n,
        f"{args.n} distinct real sandbox ids": len(set(ids)) == args.n and all(ids),
        "every rollout adapts to the UI payload": all(
            r["trajectory"] and "vx" in r["trajectory"][0] and r["environment"]["planets"]
            for r in payload["rollouts"]),
        "action sampling made the workers disagree (S4)": len(
            {(r["trajectory"][-1]["x"], r["trajectory"][-1]["y"])
             for r in payload["rollouts"]}) > 1,
        "generation recorded to runs/generations/": bool(payload.get("recorded_to")),
    }
    if len(load_stage_keys(args.stage)["train"]) >= args.n:
        # plan.md 7.1 - from Stage B on, the grid must show n DIFFERENT worlds
        checks[f"{args.n} distinct keys dispatched"] = len(
            {json.dumps(k, sort_keys=True) for k in keys}) == args.n
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(checks.values())
    print(f"\nrecorded: {payload.get('recorded_to')}")
    print("GATE A7:", "PASS - M concurrent sandboxes, M distinct ids, replayable payloads"
          if good else "FAIL - see above")
    return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="A7 - one generation of M sandboxes")
    ap.add_argument("--stage", default="A")
    ap.add_argument("-n", type=int, default=8, help="sandboxes per generation (M)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--champion", action="store_true",
                    help="use the stage's trained champion instead of a random-init net")
    return asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
