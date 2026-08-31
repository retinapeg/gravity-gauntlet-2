# The stage system

Six rungs, one gate each. A rung adds **exactly one physical complication** and then asks
the ablation study which Newtonian variables that complication forced into the
observation. The change from one column of the necessity matrix to the next is the
finding — it attributes each variable to the physics that demands it.

## What a stage folder holds

```
stages/<stage>/
├── README.md      goal · why this rung · world · ordered steps · gate · status
├── config.json    stage parameters, shaped like the worker input (plan.md §4.2)
├── keys.json      train keys and held-out eval keys for this rung
└── results/       metrics.json · champion.pt · necessity.json   (committed — the output)
```

A stage folder holds **no code**. Physics, the observation builder and the policy live
once in `gg/`; `universe.build_universe(stage, key)` branches per rung. Forking the
physics per stage would break the one property the whole study rests on: that necessity
scores from rung A are comparable with rung C.

## Keys

A *key* identifies one universe instance.

| Stages | Key shape | Why |
|---|---|---|
| A | `{}` | one fixed universe — nothing to vary |
| B | `{"layout_id": n}` | a small fixed set of planet layouts |
| C | `{"layout_id": n, "orbital_phase": p}` | same layouts, planets advanced to a phase |
| D–F | `{"seed": s}` | full per-seed randomisation |

Held-out keys are frozen **before** training starts. Success is only ever reported on
held-out keys. The train/eval split is trivial at Stage A by design — the generalisation
gap only becomes a meaningful number at Stage D (plan.md §8).

## The five-beat loop

Every rung from A onward runs the same sequence:

1. **Extend** — add the stage's branch to `gg/universe.py`.
2. **Flip** — switch the stage's new observation groups from zero-filled to live. The
   vector never changes size (`gg/spec.py`, `FIRST_ACTIVE_STAGE` column).
3. **Train** — REINFORCE over M parallel Daytona sandboxes, warm-starting `θ` from the
   previous rung's champion.
4. **Attribute** — run M1 (learned gate), M2 (ablation / permutation), M3 (saliency),
   M4 (nested feature sets) on the champion.
5. **Record** — write `results/necessity.json` and the **delta versus the previous rung**.
   That delta is the logged headline, not the raw column.

## Gate discipline

A gate is green only when **both** hold:

- the stage's success criterion is met on **held-out** keys, and
- the necessity ranking is **stable across 3 re-runs** with different action-sampling
  streams.

The second half is the one that gets skipped under time pressure. Don't — an unstable
ranking means the finding is noise, and every later column inherits the error.
