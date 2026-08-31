# Stage 00 — Daytona smoke test

**Goal.** Prove genuine parallel Daytona execution before any physics exists. Nothing else
starts until this is green.

**Why this rung.** The entire architecture assumes a generation is `asyncio.gather` over M
independent sandboxes. If that primitive doesn't work — or quietly serialises, or hands
back the same sandbox twice — every number produced later is suspect. This rung is cheap
and it is load-bearing.

## Deliverable

[`smoke_test.py`](../../smoke_test.py) — 4 sandboxes launched with `asyncio.gather`, each
running `print(seed)`, each deleted in a `finally`.

## Steps

1. **Credentials.** `.env` carries `DAYTONA_API_KEY` and `DAYTONA_API_URL`. Confirm the
   file is gitignored before committing anything.
2. **Environment.** `make setup` — creates `.venv` and installs `requirements.txt`.
3. **Run.** `make smoke`.
4. **Read the output.** Four dicts of `{sandbox_id, seed, status, output}`, then the gate
   line.
5. **Confirm on the Daytona side.** The sandboxes should be gone afterwards — the
   `finally: delete` ran. A pile of orphans means the cleanup path is broken and the
   fan-out will leak at 8× the rate.

## Gate

- Four **distinct** `sandbox_id` values — not one sandbox reused four times.
- All four `status == "finished"`.
- Each sandbox's `output` equals its own seed — the right result came back from the right
  sandbox.
- No mocks, no local fallback path.

## Notes

- `daytona.delete(sbx)` blocks until deletion. That is fine at N=4; the fan-out
  (`controller/fanout.py`) should use `wait=False` so teardown doesn't serialise the
  generation.
- Seeds are `range(4)` here. The demo narrative uses 1001–1004 — cosmetic, ignore.
- This step uses `process.code_run()` (stateless snippet). From Stage A onward the worker
  is invoked with `process.exec("python -m gg.rollout_worker")` against a **snapshot**
  instead, so no runtime `pip install` happens.

## Status

⬜ **Written, not yet run.** `numpy` is missing from the venv, but this script doesn't need
it — `make smoke` should work as soon as `make setup` has run.
