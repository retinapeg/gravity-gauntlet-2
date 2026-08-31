"""List and delete stray sandboxes.

Daytona quotas are per-organisation and count LIVE sandboxes, so an interrupted run
that never reached its `finally` will hold CPU and memory until something reaps it -
and the next run then fails with "Total CPU limit exceeded" for reasons that have
nothing to do with the code being run. This is that something.

    python -m controller.cleanup            # list what is alive
    python -m controller.cleanup --delete   # delete all of it
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from daytona import AsyncDaytona

from controller.config import daytona_config


async def run(delete: bool) -> int:
    async with AsyncDaytona(daytona_config()) as d:
        sandboxes = [s async for s in d.list()]
        cpu = sum(getattr(s, "cpu", 0) or 0 for s in sandboxes)
        mem = sum(getattr(s, "memory", 0) or 0 for s in sandboxes)
        print(f"live sandboxes: {len(sandboxes)}   cpu={cpu}   memory={mem}GiB")
        for s in sandboxes:
            print(f"  {s.id}  {getattr(s, 'state', None)}")
        if not sandboxes or not delete:
            if sandboxes and not delete:
                print("\n(pass --delete to remove them)")
            return 0
        res = await asyncio.gather(*(d.delete(s, wait=False) for s in sandboxes),
                                   return_exceptions=True)
        bad = [r for r in res if isinstance(r, BaseException)]
        print(f"\ndeleted {len(res) - len(bad)}/{len(res)}")
        for b in bad:
            print("  error:", str(b)[:160])
        return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="list / delete stray Daytona sandboxes")
    ap.add_argument("--delete", action="store_true")
    return asyncio.run(run(ap.parse_args().delete))


if __name__ == "__main__":
    sys.exit(main())
