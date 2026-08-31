"""
Daytona smoke test (Build order step 1, plan.md).

Launches 4 sandboxes in parallel, each one printing its own seed, and
confirms we get back distinct real sandbox IDs with a "finished" status.
Reads credentials from .env (DAYTONA_API_KEY, DAYTONA_API_URL).
"""

import asyncio
import os

from dotenv import load_dotenv
from daytona import AsyncDaytona, DaytonaConfig

load_dotenv()

API_KEY = os.environ.get("DAYTONA_API_KEY")
API_URL = os.environ.get("DAYTONA_API_URL")

if not API_KEY:
    raise SystemExit("DAYTONA_API_KEY is not set - check your .env file")


async def run_one(daytona: AsyncDaytona, seed: int) -> dict:
    sbx = await daytona.create()
    try:
        resp = await sbx.process.code_run(f"print({seed})")
        status = "finished" if resp.exit_code == 0 else f"error:{resp.exit_code}"
        return {"sandbox_id": sbx.id, "seed": seed, "status": status, "output": resp.result.strip()}
    finally:
        await daytona.delete(sbx)


async def main():
    config = DaytonaConfig(api_key=API_KEY, api_url=API_URL) if API_URL else DaytonaConfig(api_key=API_KEY)
    async with AsyncDaytona(config) as daytona:
        results = await asyncio.gather(*(run_one(daytona, seed) for seed in range(4)))

    for r in results:
        print(r)

    ids = {r["sandbox_id"] for r in results}
    ok = len(ids) == 4 and all(r["status"] == "finished" for r in results)
    print("\nGATE:", "PASS - 4 distinct sandboxes, all finished" if ok else "FAIL - check output above")


if __name__ == "__main__":
    asyncio.run(main())
