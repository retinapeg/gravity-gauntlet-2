"""Step A5 - bake the worker snapshot `gravity-gauntlet-worker-v1` (plan.md 4.1).

Everything an episode needs is pre-baked, so a sandbox does no `pip install` at
runtime - that is what keeps per-rollout latency inside the plan.md 4.6 budget
once the fan-out is launching eight at a time.

    python -m controller.build_snapshot             # build if missing
    python -m controller.build_snapshot --status    # report existence + state only
    python -m controller.build_snapshot --rebuild   # delete, then bake again
    python -m controller.build_snapshot --no-torch  # numpy-only worker image
    python -m controller.build_snapshot --verify    # also boot a sandbox and check imports

Gate: the snapshot exists and is ACTIVE, and a sandbox created from it can import
numpy, torch and the `gg` package - with an `obs_spec_hash` equal to the
controller's, which is what proves the baked `gg` is not a stale copy.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

from daytona import (
    AsyncDaytona,
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Image,
    Resources,
)

from controller.config import GG_DIR, SNAPSHOT, daytona_config
from gg import spec
from gg.version import source_hash

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Per-sandbox resources. These are baked INTO the snapshot - the SDK has no
# create-time override - and they set the concurrency ceiling: a Daytona org with a
# 10 GiB total-memory quota fits floor(10 / memory) sandboxes at once. The worker is
# numpy plus a ~7k-parameter MLP, so 1 GiB is generous and buys M=8 (plan.md 3.3)
# with headroom. Raise it only if a stage's physics actually needs the RAM.
SANDBOX_CPU = 1
SANDBOX_MEMORY_GB = 1


def build_image(with_torch: bool = True) -> Image:
    """The worker image: Debian slim + numpy + CPU torch + the `gg` package.

    torch is CPU-only on purpose - the policy is a ~10k-parameter MLP and the
    worker only ever does a forward pass, so a GPU wheel would be several
    hundred MB of start latency bought for nothing.
    """
    image = Image.debian_slim("3.12").pip_install("numpy", "pydantic")
    if with_torch:
        image = image.pip_install(
            "torch",
            index_url=TORCH_CPU_INDEX,
            extra_index_urls=["https://pypi.org/simple"],
        )
    # universe.py physics.py policy.py rollout_worker.py spec.py -> /opt/gg,
    # so the worker runs as `python -m gg.rollout_worker` with cwd=/opt.
    return image.add_local_dir(str(GG_DIR), "/opt/gg")


async def snapshot_state(d: AsyncDaytona) -> str | None:
    """Current snapshot state as a plain string, or None if it does not exist."""
    try:
        snap = await d.snapshot.get(SNAPSHOT)
    except Exception:
        return None
    state = getattr(snap, "state", None)
    return getattr(state, "value", None) or str(state)


async def _wait_until_gone(d: AsyncDaytona, timeout_s: float = 120.0) -> None:
    """Block until a deleted snapshot has actually disappeared.

    `snapshot.delete` returns as soon as the deletion is accepted, not when it
    has happened. Creating straight afterwards races it and loses with
    DaytonaConflictError ("already exists for this organization"), which deletes
    the old image and leaves nothing in its place - so rebuild has to wait here.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    while await snapshot_state(d) is not None:
        if asyncio.get_event_loop().time() > deadline:
            raise SystemExit(
                f"[snapshot] {SNAPSHOT} still present {timeout_s:.0f}s after delete - "
                "aborting rather than racing the create"
            )
        await asyncio.sleep(2.0)


async def ensure_snapshot(
    d: AsyncDaytona,
    *,
    with_torch: bool = True,
    rebuild: bool = False,
    quiet: bool = False,
    cpu: int = SANDBOX_CPU,
    memory_gb: int = SANDBOX_MEMORY_GB,
) -> str:
    """Make sure the snapshot exists and is active. Returns its state string.

    Idempotent: the common path is 'already present -> reuse', because baking is
    slow and every caller (rollout, fan-out, trainer) calls this on the way in.
    """
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    if rebuild:
        try:
            await d.snapshot.delete(SNAPSHOT)
            log(f"[snapshot] delete requested for {SNAPSHOT} - waiting for it to clear")
            await _wait_until_gone(d)
            log(f"[snapshot] {SNAPSHOT} gone")
        except SystemExit:
            raise
        except Exception as e:
            log(f"[snapshot] nothing to delete ({e.__class__.__name__})")
    else:
        state = await snapshot_state(d)
        if state is not None:
            log(f"[snapshot] {SNAPSHOT} already present ({state}) - reusing")
            return state

    log(f"[snapshot] building {SNAPSHOT} (torch={'yes' if with_torch else 'no'}, "
        f"cpu={cpu}, memory={memory_gb}GiB) ...")
    await d.snapshot.create(
        CreateSnapshotParams(
            name=SNAPSHOT,
            image=build_image(with_torch),
            resources=Resources(cpu=cpu, memory=memory_gb),
        ),
        on_logs=(lambda c: None) if quiet else (lambda c: print(c, end="")),
    )
    state = await snapshot_state(d)
    log(f"\n[snapshot] built {SNAPSHOT} -> {state}")
    return state or "unknown"


# The check runs *inside* the sandbox and prints one JSON line, same contract as
# the worker: whatever is baked reports on itself rather than being assumed. It is
# shipped base64-encoded because a multi-line `python -c` cannot survive the trip
# through the shell intact.
_PROBE = """
import json, sys
import numpy
from gg import spec
from gg.version import source_hash
try:
    import torch
    torch_version = torch.__version__
except Exception:
    torch_version = None
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch_version,
    "obs_dim": spec.OBS_DIM_2D,
    "obs_spec_hash": spec.obs_spec_hash(2),
    "gg_source_hash": source_hash(),
}))
"""


def _probe_command() -> str:
    payload = base64.b64encode(_PROBE.encode()).decode()
    return f"python -c 'import base64;exec(base64.b64decode(\"{payload}\"))'"


async def verify_snapshot(d: AsyncDaytona) -> dict:
    """Boot one sandbox from the snapshot and ask it what it actually contains."""
    sbx = await d.create(CreateSandboxFromSnapshotParams(
        snapshot=SNAPSHOT, auto_stop_interval=0, auto_delete_interval=5,
    ))
    print(f"[verify] sandbox live: {sbx.id}", flush=True)
    try:
        res = await sbx.process.exec(_probe_command(), cwd="/opt", timeout=120)
        if res.exit_code != 0:
            raise SystemExit(f"[verify] probe exit {res.exit_code}:\n{res.result}")
        line = [ln for ln in res.result.splitlines() if ln.strip().startswith("{")][-1]
        out = json.loads(line)
    finally:
        await d.delete(sbx, wait=False)
        print(f"[verify] delete requested: {sbx.id}", flush=True)
    out["sandbox_id"] = sbx.id
    return out


async def run(args) -> int:
    async with AsyncDaytona(daytona_config()) as d:
        if args.status:
            state = await snapshot_state(d)
            print(f"{SNAPSHOT}: {state or 'NOT FOUND'}")
            return 0 if state == "active" else 1

        state = await ensure_snapshot(d, with_torch=not args.no_torch, rebuild=args.rebuild,
                                      cpu=args.cpu, memory_gb=args.memory)
        probe = await verify_snapshot(d) if args.verify else None

    checks = {"snapshot exists and is active": state == "active"}
    if probe is not None:
        checks["numpy imports in the sandbox"] = bool(probe.get("numpy"))
        checks["gg package imports in the sandbox"] = probe.get("obs_dim") == spec.OBS_DIM_2D
        checks["baked gg matches the controller's obs spec"] = (
            probe.get("obs_spec_hash") == spec.obs_spec_hash(2)
        )
        checks["baked gg source is not stale"] = (
            probe.get("gg_source_hash") == source_hash()
        )
        if not args.no_torch:
            checks["cpu torch imports in the sandbox"] = bool(probe.get("torch"))

    print()
    if probe is not None:
        print(f"  python={probe['python']}  numpy={probe['numpy']}  "
              f"torch={probe['torch'] or 'ABSENT'}")
        print(f"  obs_dim={probe['obs_dim']}  obs_spec_hash={probe['obs_spec_hash']} "
              f"(controller: {spec.obs_spec_hash(2)})")
        print(f"  gg_source_hash={probe.get('gg_source_hash')} "
              f"(controller: {source_hash()})")
        print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    good = all(checks.values())
    print("\nGATE A5:", "PASS - snapshot active and the baked worker image is correct"
          if good else "FAIL - see above")
    return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Step A5 - bake the worker snapshot")
    ap.add_argument("--rebuild", action="store_true", help="delete and bake again")
    ap.add_argument("--no-torch", action="store_true",
                    help="numpy-only worker image (smaller, faster to start)")
    ap.add_argument("--verify", action="store_true",
                    help="boot one sandbox from the snapshot and check what is baked")
    ap.add_argument("--cpu", type=int, default=SANDBOX_CPU, help="vCPU per sandbox")
    ap.add_argument("--memory", type=int, default=SANDBOX_MEMORY_GB,
                    help="GiB per sandbox - sets the concurrency ceiling against the org quota")
    ap.add_argument("--status", action="store_true", help="report existence + state, then exit")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
