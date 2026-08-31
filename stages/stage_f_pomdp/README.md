# Stage F — partial observability · *plan only, stretch*

**Not demoed.** The last rung, and the only one that changes the *shape* of the agent
rather than the shape of the world.

**Goal.** Show what the policy needs when it can no longer see the whole system — and
watch the necessary set shift from global quantities to local and remembered ones.

**Why this rung.** Every rung so far handed the policy a complete, instantaneous, exact
picture. Real spacecraft have sensors with range and noise. Once distant bodies are hidden,
the observation stops being a sufficient statistic for the state, and a feed-forward MLP is
provably the wrong tool: the missing information has to be reconstructed from history.
That makes this the rung where **memory itself** becomes a necessary variable.

## World

Stage E's 3D universes, plus:

- **S3 observation noise** — `o_t = o_t^true + N(0, Σ)`.
- **Sensor range** — bodies beyond a radius are masked out of the observation entirely.
  The masked groups are zero-filled, exactly as unborn groups were in earlier rungs, so the
  vector length still never changes.
- Heavier S2 process noise than Stage D.

## Agent change

Swap the MLP trunk for a small GRU, keeping the same `act(ctx) -> (action, logprob)`
interface that `gg/policy.py` has used since the scripted policies. Training becomes
truncated BPTT over each trajectory — the worker already returns whole trajectories in
order, so no change to the Daytona contract is required.

## Predicted necessity delta vs E

- **Local and recent observations dominate**; the learned gate re-weights hard toward the
  nearest-body group.
- Global aggregates (`Φ`, total-field terms) **degrade** — they are now computed from a
  partial view and are actively misleading, so a good policy should learn to distrust them.
- The **hidden state** carries what the sensor cannot: a body seen thirty steps ago and now
  out of range still matters.

## Steps

1. Sensor-range masking in `gg/obs.py` — mask, don't resize.
2. S3 observation noise in `gg/episode.py`, driven by the existing `noise.obs_sigma` field
   in the worker input contract.
3. GRU trunk in `gg/policy.py` behind the unchanged interface.
4. Truncated BPTT in `controller/train.py`.
5. **Control experiment** — evaluate the Stage E feed-forward champion under Stage F
   conditions. It should fail. That failure is the evidence that memory is necessary; without
   it, "we added a GRU" is an unsupported claim.
6. M1–M4; record column F.

## Gate

The recurrent policy stays above a success threshold under noise and masking that
measurably breaks the feed-forward Stage E champion.

## Status

⬜ Documented, not implemented. Stretch goal, post-hackathon.
