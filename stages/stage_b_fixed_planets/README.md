# Stage B — star + fixed planets

**Goal.** Find what a **static hazard** adds to the necessary set.

**Why this rung.** The field is now multi-well but still frozen in time. The craft can be
destroyed by something other than its own trajectory, and the gravitational field is no
longer a single clean `1/r²`. The question this rung answers is sharp: does the policy
need to *enumerate the bodies*, or is the aggregated field plus one nearest-body term
enough?

## World

The Stage A star plus 2–3 stationary planets, from a small fixed layout set. Planets have
mass and a contact radius; they do not move and they do not pull on each other.

To implement: a `"B"` branch in `gg/universe.py` and a `FIXED_LAYOUTS` table. Layouts must
differ in *character*, not just coordinates — one with a planet straddling the direct
path, one with a mass concentration off to a side, one nearly clear. Five near-identical
layouts teach the policy nothing.

## Keys — and the demo grid starts here

`{"layout_id": n}` for `n` in 0–11. **Layouts 10 and 11 are held out** and never trained on.

**Twelve layouts, not five**, because a generation dispatches **8 distinct keys**
(plan.md §7.1) — eight different worlds per generation, not eight copies of one. This is
the rung where the demo grid finally shows what the pitch claims: one frozen policy being
evaluated across eight visibly different gravitational universes, every one of them exactly
replayable from its key.

Each generation samples 8 of the 10 training layouts without replacement, so the batch
rotates and no single layout gets memorised. See `keys.json` for the intended *character*
of each layout — they must differ in geometry, not just coordinates. Eight near-identical
fields teach the policy nothing and read as a rendering bug on screen.

## Newly live observation groups

`nearest_body` (4) · `body_count` (1) · `perbody_topk` (9, k=3) — flip these from
zero-filled to live in `gg/obs.py`. The vector length does not change.

## Predicted necessity delta vs A

**+ nearest-body range and mass** — the acute collision hazard.
**Not** per-body enumeration: `perbody_topk` and `body_count` are predicted to stay
unnecessary, because the aggregated `a_grav` already carries the far field and the
nearest-body term carries the hazard. This is the sharpest falsifiable claim in the
project — M4's nested feature sets B vs. C vs. D decide it.

If `perbody_topk` *does* come back necessary, that is a finding worth reporting, not a
bug. Report measured against predicted.

## Steps

1. **Extend** `gg/universe.py` with the `"B"` branch and the layout table.
2. **Flip** `nearest_body`, `body_count`, `perbody_topk` live in `gg/obs.py`.
3. **Collision check** — verify contact with a *planet* terminates the episode, not just
   contact with the star (`tests/`, no network).
4. **Scripted sanity** — `PursuitPolicy` has no avoidance term, so expect it to crash on
   the layouts with a planet on the direct path. That's the correct baseline: it shows the
   hazard is real before any learning happens.
5. **Train** warm-started from the Stage A champion (`stages/stage_a_fixed_star/results/champion.pt`).
6. **Attribute** — M1–M4 on the Stage B champion.
7. **Record** `results/necessity.json` **and the delta vs column A** — which rows changed
   verdict, and by how much `ΔReturn`.

## Gate

- Success rate high and stable on the **held-out** layouts 3 and 4.
- Necessity ranking stable across 3 re-runs.
- The B-vs-A delta is explainable in one sentence. If it isn't, something moved that
  shouldn't have — check that the observation superset really was frozen.

## Status

🟨 **Implemented and running end-to-end. Fan-out gate green; training gate red.**

**Implementation.** `gg/universe.py::_stage_b` + a 12-entry `FIXED_LAYOUTS` table.
Flipping `nearest_body` / `body_count` / `perbody_topk` live required **no observation
code at all** — `build_observation` already computes every group and gates it on
`spec.stage_active`, which is the frozen-superset design paying for itself.

*Correction to the step list above:* it says `PursuitPolicy` "has no avoidance term". It
does (`avoid_gain=2.2`), which is why the scripted baseline crashes on only **1 of 12**
layouts (layout 0, the planet straddling the direct path) rather than on all the hard ones.
The hazard is still real: a *random-init* policy crashes on 4 of 8.

**Fan-out — the demo grid starts here.** ✅

```
generation 2   8 concurrent sandboxes, 8 distinct real ids, wall 3.2s
  worker 0  048b0f52…  crashed  reward -106.85     <- 8 DIFFERENT worlds,
  worker 2  fd05d5cd…  timeout  reward  -45.36        not 8 copies of one
  ...
  [PASS] 8 distinct keys dispatched
```

`runs/generations/gen_2.json` replays it exactly.

**Training.** ❌ Warm-started from the Stage A champion, 150 generations, 32 episodes/gen.

| | value |
|---|---|
| generation 1 (the inherited policy) | success **0.53** |
| best held-out probe (layouts 10 & 11, never trained on) | **0.50** at gen 10 |
| final champion, held-out sampled | **0.38** (3/8 portal) |
| training success across the run | 0.22 → 0.04 |

**The gate fails on the criterion that matters**: success did not *rise*. The champion is
essentially the transferred Stage A policy with minor adaptation — Stage B training made it
no better, and mostly made it worse. 150 generations at ~3 episodes per layout is not
enough signal per world to improve on a warm start.

That said, **0.38–0.50 on layouts 10 and 11** is a real result: those geometries were never
trained on, so the transferred policy does generalise to unseen fields.

### Two things this rung taught, both now fixed in code

**A warm start across a rung that switches groups on is not warm.** Those groups were
zero-filled on every earlier rung, so their `W1` rows never took a gradient and were still
random init — 14 of 38 input dimensions at Stage B, at the *same magnitude* as the trained
weights. `controller/train.py::quiet_new_inputs` now zeroes them so the inherited policy
behaves exactly as it did on the previous rung. **Measured honestly: this changed
generation-1 success by less than noise (0.467 → 0.433). Correct in principle, but it was
not the cause of the failure.** Recorded because the wrong diagnosis is worth as much as
the right one.

**The actual cause: a batch-wide advantage baseline across mixed layouts.** From Stage B a
batch spans several worlds whose returns differ by the full ±100 goal bonus, so the
advantage mostly encodes *which layout you drew* rather than whether the action was good —
the same pathology the value head fixes for elapsed time at Stage A, one level up.
Centring returns **per key** recovered a stalled run from 0.03 success to ~0.4, and turned
every held-out probe from 0.00 into 0.50.

**Next.** More generations and a deeper batch per layout, not a different algorithm. The
necessity study (M1–M4) is deliberately **not** run here: with a champion this weak the
±100 goal bonus swamps ΔReturn, which Stage A already demonstrated produces a column of
noise. Column B is worth measuring only against a champion that clears its own gate.
