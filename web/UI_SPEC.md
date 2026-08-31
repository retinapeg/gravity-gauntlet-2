# Gravity Gauntlet — UI specification (reconciled)

The authoritative UI contract. Reconciles the animation/UI spec with [plan.md](../plan.md)
and the `gg/` worker that already exists. Where the two disagreed, the resolution is
recorded in §2 — read that first.

---

## 1. The product message

> **One universe gives an agent one experience. Daytona gives it many universes at once.**

The screen must make this legible in five seconds:

```
POLICY  →  8 DAYTONA UNIVERSES  →  TRAJECTORIES / REWARDS  →  CHAMPION  →  NEXT GENERATION
```

Then, a few seconds later: *those experiences are improving the policy.* Then: *Daytona is
what makes the parallel collection possible.*

This is a **visualisation-first** product. The physics is the visual centrepiece; metrics
support it. It is not a dashboard, not a spreadsheet, not a static orbital simulator.

**Direction:** near-black space ground, luminous/glowing treatments, restrained star field.
The simulation dominates the viewport.

---

## 2. Reconciliation — where the UI spec and plan.md met

| Conflict | Resolution |
|---|---|
| **UI wants 8 visibly different universes with distinct seeds. plan.md Stages A–C are deterministic — Stage A is one single universe.** | **A generation dispatches 8 *distinct keys*, not 8 copies of one.** "Deterministic" means *reproducible given the key*, not *one world*. Stage B/C layout sets expand from 5 to **12**, so a generation fans out over 8 different `(layout_id, orbital_phase)` pairs and the grid is genuinely varied while every world stays exactly replayable. |
| **Stage A can't do that** — it has one universe by definition. | **Stage A is the calibration rung, not a demo rung.** It proves physics, the worker contract, the fan-out and the training loop. The 8-universe *grid* demo begins at Stage B. |
| **UI reward scale (+500 / −300 / −200) vs `gg/spec.py` (100 / 100 / 50).** | Keep `gg/spec.py`. The UI spec says values "should be tuned rather than treated as fixed requirements", and plan.md's scale already yields the ±100–300 range the mockups show. **Never tune reward to make the UI look better** — tune it for learning and let the UI render what comes back. |
| **UI worker payload field names differ from plan.md §4.2** (`policy_id` vs `policy_version`, `status` vs `termination`, per-point trajectory dicts vs flat rows). | **Thin adapter in `controller/adapter.py`.** The worker contract is the source of truth and does not change to suit the frontend (UI spec §30, §41). Mapping in §5 below. |
| **UI wants non-gravitational obstacles (§10).** | **Deferred.** plan.md has gravitating bodies only. The adapter emits `obstacles: []` and the renderer handles an empty list. Add later as a `Universe` field if a rung needs it — not before. |
| **UI describes continuous `[thrust_x, thrust_y]` (§19).** | Discrete 9-action compass (plan.md §2.3), which the spec explicitly permits. The adapter emits each action as a **unit thrust vector** so the exhaust animation works identically either way. |
| **UI wants pixel sizes (planets 18–45 px, portal 20–35 px).** | The simulation is nondimensional (`arena_radius ≈ 12`). The **renderer** owns the world→screen transform: fit `arena_radius` to the canvas, scale body radius by mass, then clamp into the spec's pixel bands. Physics units never leak into the UI and pixel units never leak into physics. |

**The rule that survives all of it:** the backend is the source of truth. No second physics
engine in the frontend. No fabricated trajectories, rewards, successes, sandbox IDs or
generation results.

---

## 3. Layout

```
┌───────────────────────────────────────────────────────────────┐
│ GRAVITY GAUNTLET     GENERATION 12     DAYTONA: 8 ACTIVE      │
│ One policy. Eight universes. Learning in parallel.            │
├─────────────────────────────────────┬─────────────────────────┤
│                                     │ Generation stats        │
│          CHAMPION UNIVERSE          │  best / mean reward     │
│      large animated simulation      │  success rate           │
│      + ghost trails                 │  policy · generation    │
├─────────────────────────────────────┴─────────────────────────┤
│ PARALLEL DAYTONA UNIVERSES                                    │
│   [ U1 ] [ U2 ] [ U3 ] [ U4 ]                                 │
│   [ U5 ] [ U6 ] [ U7 ] [ U8 ]                                 │
├───────────────────────────────────────────────────────────────┤
│               [ TRAIN NEXT GENERATION ]                       │
└───────────────────────────────────────────────────────────────┘
```

**Header** carries identity and state: generation, policy id, active sandbox count,
success rate, best reward. Daytona parallelism is never hidden in a settings page.

**Champion panel** is substantially larger than a card and replays the best rollout at
cinematic speed with ghost trails beneath it.

**Universe card** — one per sandbox:

```
┌────────────────────────────┐
│ UNIVERSE 03      ● RUNNING │
│ sbx 7fd1…        seed 4183 │
│                            │
│      ◯ planet              │
│   ▶───────╮                │
│           ╰──────◎ portal  │
│ REWARD +84.3               │
└────────────────────────────┘
```

Drawn shapes, never emoji. Card header carries **universe number, real sandbox ID, seed,
lifecycle state**. That is what makes the infrastructure tangible.

---

## 4. Lifecycle states

```
CREATING  →  READY  →  RUNNING  →  COMPLETE
                                 ├→ SUCCESS
                                 ├→ CRASHED
                                 ├→ TIMEOUT
                                 └→ ESCAPED
                    (any point)  →  WORKER FAILED
```

Mapped from backend events (plan.md §4.4). Cards **must not** complete in lockstep —
staggered completion is what communicates distributed execution. A failed worker shows
`WORKER FAILED` on its own card and never takes down the page; the trainer drops that
sample from the batch (plan.md §4.5).

---

## 5. The adapter — `controller/adapter.py`

Worker output (plan.md §4.2, implemented in `gg/rollout_worker.py`) → UI rollout payload.
**One function, no logic beyond renaming and reshaping.**

| UI field | Source |
|---|---|
| `worker_id` | index within the generation's `asyncio.gather` |
| `sandbox_id` | passthrough |
| `generation`, `policy_id` | controller; `policy_id = f"P{policy_version:03d}"` |
| `seed` | the rollout's action-sampling seed — real even in deterministic stages |
| `key` | `{layout_id, orbital_phase}` (A–C) or `{seed}` (D+) — what actually identifies the world |
| `status` | `termination` mapped: `portal→success`, `collision→crashed`, `timeout→timeout`, `escape→escaped` |
| `reward`, `steps` | passthrough |
| `trajectory[]` | `{t, x, y, vx, vy, action, reward}` per step, zipped from the worker's `trajectory`, `actions`, `rewards` |
| `environment.planets[]` | `{x, y, mass, radius}` from the universe — **radius is a real collision radius**, not a display guess |
| `environment.portal` | `{x, y, r_capture, v_capture}` |
| `environment.ship_start` | `{x, y, vx, vy}` |
| `environment.obstacles` | `[]` — deferred |
| `bodies_trajectory` | passthrough — required from Stage C, where bodies move |

**Velocity.** The worker's trajectory rows should carry `vx, vy` (a one-line change:
rows become `[t, x, y, vx, vy, fuel]`). Until they do, the adapter finite-differences
consecutive positions. Ship rotation is `atan2(vy, vx)` either way.

**Generation payload** (`controller/fanout.py`):

```json
{ "generation": 12, "policy_id": "P012", "base_seed": 4180,
  "keys": [...], "rollouts": [...],
  "mean_reward": 42.6, "best_reward": 184.2,
  "success_rate": 0.25, "champion_worker": 3 }
```

---

## 6. Renderer rules

**Replay, don't simulate.** Simulation time and display time are separate. A 500-step
rollout completes in a fraction of a second; the UI replays it over 4–8 s (champion:
6–12 s). Interpolate between recorded points for 60 fps — **never invent positions**.

```
display_frame_time = display_duration / simulation_steps
```

| Element | Treatment |
|---|---|
| **Spacecraft** | Drawn ship/arrow, rotated to velocity. Exhaust + glow pulse when the recorded action is a burn — driven by the real action, not a timer. Never an anonymous circle. |
| **Trajectory** | Smooth trail from actual recorded positions, fading toward the oldest points. Launch, gravitational bending, close passes, crashes and captures must all be readable. |
| **Ghost trails** | Last 3–5 champion trajectories at low opacity, behind the live craft. This is what makes learning visible — it is not decoration. |
| **Champion trail** | Brighter, stronger glow, higher opacity, longer persistence than any ghost. |
| **Planets** | Luminous edge, atmospheric glow, radius scaled by mass. 18–45 px on cards, 45–60 px on the champion. Must not crowd out navigable space. |
| **Gravity fields** | Translucent rings / halo / field lines so curvature is explicable. **Cosmetic only** — the curve itself comes from backend physics. |
| **Portal** | Unmistakable: pulsing, rotating shimmer, glow. 20–35 px. Stronger animation on approach, flash on capture. |
| **Near miss** | Proximity below a threshold intensifies planet glow and brightens the trail. Presentation only, driven by real distances. |
| **Crash** | Playback stops at the collision point, brief flash + particle burst (300–600 ms), craft fades, trajectory **stays visible** so the failure is readable. |
| **Success** | Portal flare (500–1000 ms), craft enters and fades, trajectory stays highlighted. |
| **Background** | Restrained star field; optional slow parallax. Never busy enough to compete with trajectories. |
| **Status transitions** | 150–300 ms. |

**Technology:** Canvas 2D for the arenas. No thousands of DOM nodes for trail points. No
WebGL unless smoothness actually demands it.

---

## 7. `TRAIN NEXT GENERATION`

One prominent control. Each press must run a **genuine** generation — never replay the
same animation.

```
freeze policy → new generation id → new base seed → 8 distinct keys
  → launch 8 sandboxes concurrently (asyncio.gather, never sequential)
  → collect states / actions / rewards / trajectories / outcomes
  → stream to UI and animate
  → aggregate batch → update policy → P(n+1)
  → best reward / mean / success rate → select champion
  → previous champion becomes a ghost trail
```

All eight workers evaluate the **same frozen policy**. They are collecting experience for
one agent — not eight agents training independently.

`AUTO TRAIN` / `PAUSE` optional; the one-click button is what matters for a live demo.

---

## 8. Live mode and replay mode

Every live generation is written to `runs/generations/gen_<n>.json` as it completes.
Replay mode reads those back through the **identical renderer**.

```
LIVE    →  Daytona  →  runs/generations/gen_012.json  →  UI
REPLAY  →            runs/generations/gen_012.json    →  UI
```

Replay uses **genuine stored rollouts** — precomputed real data is fine, fabricated data
never is. The header shows which mode is active. This is the fallback if the API is
unstable during judging, and it costs almost nothing because the recording happens anyway.

---

## 9. Honesty rules

Non-negotiable, and the reason the UI is worth trusting:

- No fabricated trajectories, rewards, successes, sandbox IDs, policy improvements or
  generation results. Loading animations, star fields and glow effects are fine.
- A panel showing anything that is not a real measured result — a mocked sandbox ID, a
  scripted policy standing in for a trained one, *predicted* rather than *measured*
  necessity values — says so on its face.
- A failed worker is shown as failed. It is never quietly replaced with demo data.
- If a fallback mode ever does synthesise data, it is labelled as such in the UI, not just
  in a code comment.

## 10. Boundaries

Do not rewrite the RL or Daytona infrastructure to make the frontend convenient. Do not
build a second physics engine, a second environment, or a second repository. Preserve the
existing pipeline:

```
gg/universe.py + gg/physics.py  →  gg/rollout_worker.py  →  Daytona sandboxes
    →  controller/fanout.py  →  controller/train.py  →  generation result  →  web/
```
