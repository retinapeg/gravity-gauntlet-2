"""Build-order step 1, the Daytona half: run the Stage-A episode in ONE real
sandbox created from the pre-baked snapshot (plan.md 4.1-4.3, 10.2).

  1. ensure snapshot `gravity-gauntlet-worker-v1` exists (build it if missing)
  2. create one sandbox from it
  3. exec `python -m gg.rollout_worker` twice in that sandbox:
       - scripted pursuit  -> expect termination "portal", success True
       - no thrust         -> expect energy drift < 1% over 500 steps + timeout
  4. print the lifecycle events (plan.md 4.4) and a gate table; delete the sandbox

No mocks: the sandbox id is real and printed; the physics runs inside Daytona.

The snapshot itself is step A5 and lives in `controller/build_snapshot.py`; this
script only consumes it, so the image is defined in exactly one place.

    python run_one_sandbox.py                # use / build the snapshot
    python run_one_sandbox.py --rebuild      # delete + rebuild the snapshot first
    python run_one_sandbox.py --no-torch     # numpy-only worker image
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

from controller.build_snapshot import ensure_snapshot
from controller.config import SNAPSHOT, daytona_config
from gg import spec

PURSUIT_INPUT = dict(
    stage="A", deterministic=True, dims=2, layout_id=0,
    policy_version=0, policy_kind="scripted_pursuit", policy_weights_b64=None,
    obs_spec_hash=spec.obs_spec_hash(2), max_steps=500, gamma=0.99, dt=0.02,
    noise={"process_sigma": 0.0, "thrust_jitter": 0.0, "obs_sigma": 0.0},
)
COAST_INPUT = dict(PURSUIT_INPUT, policy_kind="no_thrust")


def _event(**kw):
    """plan.md 4.4 lifecycle event -> stdout."""
    print("  event", json.dumps(kw))


async def exec_worker(sbx, payload: dict, label: str) -> dict:
    _event(world_id=label, phase="running", sandbox_id=sbx.id)
    res = await sbx.process.exec(
        "python -m gg.rollout_worker", cwd="/opt",
        env={"GG_INPUT": json.dumps(payload)}, timeout=120,
    )
    if res.exit_code != 0:
        raise SystemExit(f"[{label}] worker exit {res.exit_code}:\n{res.result}")
    line = [ln for ln in res.result.splitlines() if ln.strip().startswith("{")][-1]
    out = json.loads(line)
    out["sandbox_id"] = out.get("sandbox_id") or sbx.id
    _event(world_id=label, phase="finished", sandbox_id=sbx.id,
           reward=out["reward"], termination=out["termination"])
    return out


async def run(with_torch: bool, rebuild: bool) -> int:
    async with AsyncDaytona(daytona_config()) as d:
        await ensure_snapshot(d, with_torch=with_torch, rebuild=rebuild)

        _event(world_id="stage-a", phase="creating")
        sbx = await d.create(CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT, auto_stop_interval=0, auto_delete_interval=5,
        ))
        _event(world_id="stage-a", phase="live", sandbox_id=sbx.id)
        print(f"\n[sandbox] live: {sbx.id}\n")

        try:
            pursuit = await exec_worker(sbx, PURSUIT_INPUT, "pursuit")
            coast = await exec_worker(sbx, COAST_INPUT, "coast")
        finally:
            await d.delete(sbx, wait=False)
            print(f"\n[sandbox] delete requested: {sbx.id}")

    print("\n" + "=" * 68)
    print(f"{'run':>9}  {'term':>10}  {'steps':>5}  {'reward':>9}  {'return_g0':>9}  {'extra'}")
    print(f"{'pursuit':>9}  {pursuit['termination']:>10}  {pursuit['steps']:>5}  "
          f"{pursuit['reward']:>9.2f}  {pursuit['return_g0']:>9.2f}  "
          f"success={pursuit['success']} fuel_left={pursuit['fuel_left']:.3f}")
    print(f"{'coast':>9}  {coast['termination']:>10}  {coast['steps']:>5}  "
          f"{coast['reward']:>9.2f}  {coast['return_g0']:>9.2f}  "
          f"energy_drift={coast['energy_drift']:.2e}")
    print("=" * 68)

    checks = {
        "distinct real sandbox id": bool(pursuit["sandbox_id"]) and pursuit["sandbox_id"] == coast["sandbox_id"],
        "pursuit reached the portal": pursuit["termination"] == "portal" and pursuit["success"],
        "full trajectory returned": len(pursuit["trajectory"]) == pursuit["steps"] + 1
                                     and len(pursuit["observations"]) == pursuit["steps"],
        "obs spec hash matches controller": pursuit["obs_spec_hash"] == spec.obs_spec_hash(2),
        "energy drift < 1% over 500 steps (thrust off)": coast["energy_drift"] < 0.01 and coast["steps"] == 500,
        "coast run times out": coast["termination"] == "timeout",
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(checks.values())
    print("\nGATE:", "PASS - one real Daytona sandbox, full Stage-A trajectory, drift < 1%"
          if good else "FAIL - see above")
    print(f"sandbox_id: {pursuit['sandbox_id']}   physics_checksum: {pursuit['physics_checksum']}")
    return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true",
                    help="numpy-only worker image (CPU torch is baked by default, plan.md 4.1)")
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the snapshot first")
    args = ap.parse_args()
    return asyncio.run(run(not args.no_torch, args.rebuild))


if __name__ == "__main__":
    sys.exit(main())
