# web — the demo UI

**Full contract: [UI_SPEC.md](UI_SPEC.md).** Read that before writing any UI code — it
carries the layout, the adapter mapping, renderer rules, animation timings and the honesty
rules, plus a reconciliation table recording where the animation spec and `plan.md`
disagreed and how it was settled.

## The one-line version

> One universe gives an agent one experience. Daytona gives it many universes at once.

```
POLICY → 8 DAYTONA UNIVERSES → TRAJECTORIES / REWARDS → CHAMPION → NEXT GENERATION
```

**Daytona calculates. The browser renders.** The worker never draws anything and the UI
never simulates anything — it replays recorded rollouts, interpolating between real points
for 60 fps. No second physics engine in the frontend.

## Panels

1. **Header** — generation, policy id, active sandbox count, success rate, best reward.
   Daytona parallelism is never hidden in a settings page.
2. **Champion universe** — the best rollout replayed at cinematic speed (6–12 s), with the
   last 3–5 champions as ghost trails beneath it. Substantially larger than a card.
3. **Parallel Daytona universes** — 8 cards, one per sandbox, each a miniature simulation
   with universe number, **real** `sandbox_id`, seed, lifecycle state, trajectory and
   reward. Staggered completion, never lockstep.
4. **Generation stats** — best / mean reward, success rate, policy, generation. Compact;
   supports the simulation rather than dominating it.
5. **TRAIN NEXT GENERATION** — one prominent control that runs a genuine generation.

## Build order

The UI lands at build-order step 7, **after** `controller/` step A7 has the SSE stream, the
adapter and generation recording working. That ordering is deliberate: the UI is built
against real rollout payloads from its first commit, so there is never a fake-data phase to
unwind.

## Data in

- `GET /events` (SSE) — `{world_id, key, phase, sandbox_id, reward?, status?}` on every
  lifecycle transition.
- `GET /generation/{n}` — the adapted generation payload for replay.
- `GET /necessity/{stage}` — the measured necessity column.
- Replay mode reads `runs/generations/gen_<n>.json` through the identical renderer.

## Honesty rule

If a panel shows something that is not a real measured result — a mocked sandbox ID, a
scripted policy standing in for a trained one, *predicted* rather than *measured* necessity
values — the panel says so on its face. A judge discovering an unlabelled mock costs more
than the panel was ever worth.
