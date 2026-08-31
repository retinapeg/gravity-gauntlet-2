# Stage D — randomised universes + noise

**Goal.** Find which variables become necessary for **robustness** rather than for control.

**Why this rung.** Everything up to here is deterministic: the policy could in principle
succeed by exploiting one specific world. Turning on per-seed randomisation and process
noise removes that option and asks a different question — not "what do you need to fly?"
but "what do you need to keep flying when the world is not the one you trained on?"

This is also the first rung where the **fleet's diversity argument** applies. Through
Stages A–C the sandboxes justify themselves on throughput and gradient variance alone
(only action sampling varies); here each sandbox is genuinely a different universe.

## World

Full per-seed generation (plan.md §2.6, S1): `N ∈ {2..5}` bodies, log-uniform masses over
~2 decades, positions in an annulus, near-circular velocities with eccentricity jitter,
random portal arc and craft start.

Plus **S2 process noise** — an additive impulse `η_t ∼ N(0, σ²I)` each step and a
multiplicative thrust misfire `(1+δ)`, `δ ∼ N(0, 0.03²)`, with a small pointing error.
S3 observation noise stays off until Stage F.

## Keys

`{"seed": s}`. Training seeds are resampled **every generation**; a fixed eval-seed set is
frozen before the first run and never trained on.

## Newly live observation groups

None — the vector is unchanged. What changes is which components *matter*. `control_authority`
(`Π₁`) and the fuel margin have been present and unnecessary since Stage A; the prediction
is that they become necessary here.

## Predicted necessity delta vs C

- **+ control authority `Π₁`** — with universes of varying mass scale, "how hard can I
  push relative to local gravity" stops being a constant.
- The nearest-body term **up-weights**; far-field and global features **down-weight**.
- `|a_grav|, φ_g` may turn marginal: redundant encodings help a small network under noise.

## Steps

1. Implement `sample_universe(seed, dims=2)` in `gg/universe.py`.
2. Turn on S2 in `gg/episode.py`, driven by the `noise` block already in the worker input
   contract (plan.md §4.2) — the plumbing exists, the values are just zero.
3. Freeze the eval-seed set; start reporting the **generalisation gap** (train vs.
   held-out success), which is the first rung where that number means anything.
4. Curriculum ramp if learning stalls: start at `N=2` with near-circular orbits, raise `N`
   and eccentricity by generation.
5. Train warm-started from the Stage C champion.
6. M1–M4; record column D and the delta vs C.

## Gate

Success rate on **held-out seeds** climbs and plateaus; generalisation gap stays small;
necessity ranking stable across 3 re-runs.

## Risk specific to this rung

Gradient variance jumps when the universe varies. If the curve stops climbing: raise `M`,
add the fitted value baseline, then switch the objective to clipped PPO. The worker output
already carries `logprobs`, so that switch is controller-side only.

## Implementation

**Universe (`gg/universe.py::sample_universe`).** A whole world from one integer. `N` in
2..5 bodies counting the star, planet masses log-uniform over ~1.2 decades, semi-major
axes in an annulus, eccentricity 0..0.28, random argument of periapsis and mean anomaly.
The craft start and the portal are drawn as a random *arc*: absolute orientation uniform
on the circle, angular separation 25-100 degrees. Randomising the orientation while
keeping the separation in family with Stages A-C is deliberate - it removes the fixed
corridor the earlier rungs could memorise without moving the difficulty.

Everything is a pure function of the seed, so a Stage D world replays exactly the way an
A/B/C one does and `runs/generations/*.json` stays replayable.

**The key is `seed_universe`, not `seed`.** `seed` is the action-sampling seed (S4). They
must stay separable: the same world flown with different action draws is exactly the
Monte-Carlo comparison the fleet exists to make (plan.md 5.3).

**Orbits become real ellipses.** `Universe.bodies_at` gained a Kepler branch - Newton on
`E - e sin E = M` with a FIXED five iterations, so the trajectory is a pure function of `t`
rather than of floating-point luck. Stage C passes `orbit_ecc=None` and keeps its exact
circular arithmetic, which is why C's recorded physics checksums are unchanged. The
velocity returned is the true derivative of the position (checked against a central
difference in `tests/test_stage_d.py`), because the velocity-Verlet half-steps consume it.

**S2 noise.** Three sources, all gated on `deterministic` so stages A-C cannot be touched
no matter what their config carries:

| source | value | effect |
|---|---|---|
| `process_sigma` | 0.002 | per-step velocity impulse; a ~0.045 random walk over 500 steps against `v_capture` 0.6 |
| `thrust_jitter` | 0.03 | multiplicative misfire `(1+delta)` on delivered thrust |
| `pointing_sigma` | 0.05 rad | the burn is aimed slightly wrong |

Fuel is drawn on the **commanded** burn, not the delivered one - a misfire wastes
propellant rather than refunding it, or noise would be a fuel bonus the policy learns to
want (`test_a_misfire_wastes_fuel_rather_than_refunding_it`).

**Train/eval seeds are disjoint by construction.** Training seeds are drawn fresh every
generation from `[1, 800000)`; the 24 frozen eval seeds live at `900000+`. A training draw
can never land on an eval world, which is what makes the generalisation gap mean something.
This is the first rung where train and held-out are different *worlds* rather than
different keys into one fixed set, so `generalisation_gap` starts being recorded here.

**Observation vector unchanged.** No group goes live at D - `control_authority` and
`fuel_fraction` have been present and unnecessary since Stage A, and the prediction is that
robustness promotes them. `obs_spec_hash` stays `16a7d04e`.

## A bug this rung's tests caught

`worker_input` sets `seed` but never `rng_seed`, and the worker defaulted the noise stream
to `rng_seed=0`. Every rollout in a generation would therefore have drawn the **identical**
noise sequence - eight sandboxes receiving the same "random" disturbance, which is no
disturbance at all, and S2 would have been silently inert for the whole rung. The noise
stream is now derived from the episode seed (offset so it does not correlate with the
action draw). Caught by `test_noise_actually_changes_the_trajectory`.

## Is the rung fair? — yes, and that is measured first

Same guard Stage C needed. Before reading anything into a low learned score, the scripted
`PursuitPolicy` was run on the same seeds:

| distribution | scripted pursuit |
|---|---|
| difficulty 0 (easy anchor) | 0.96 |
| difficulty 0.5 | 0.96 |
| **difficulty 1.0 (full spec, S2 noise on)** | **0.88** |

The worlds are flyable and the A7 gate is reachable under noise. Any low number from the
learned policy is about the policy, not about the rung being impossible.

## Training gate: RED, with the cause localised

Three training strategies were run, all warm-started from the Stage C champion
(held-out 0.83 phase / 0.96 layout):

| strategy | generations | held-out (difficulty 1.0) |
|---|---|---|
| straight into full randomisation | 120 | 0.04 - 0.08, no trend |
| linear curriculum, all knobs together | 150 | 0.00 - 0.08, no trend |
| staged curriculum + per-world advantage baseline | 200 | 0.04 - 0.08, no trend |

Closest approach improves slowly (1.53 -> 1.36 over 200 generations) but never reaches the
braking basin, whose sigma is ~1.05. The policy is not learning to fly these worlds.

### What the transfer actually looks like

The Stage C champion, evaluated directly on Stage D worlds:

| distribution | C champion |
|---|---|
| difficulty 0 (one planet, Stage-C-like corridor) | **0.62** |
| difficulty 0.25 | 0.08 |
| difficulty 1.0 | 0.06 |

So the easy anchor is genuinely the right anchor - the champion transfers to it - and there
is a cliff immediately after. Opening ONE knob at a time from the anchor:

| knob opened | C champion |
|---|---|
| nothing | 0.62 |
| **absolute orientation free** | **0.17** |
| two planets instead of one | 0.42 |
| three planets instead of one | 0.21 |

Orientation is the single largest drop. The reason is structural: `craft_vel`, `a_grav` and
`a_grav_magdir` are in the **absolute frame**, so a rotated world is unrecognisable to a
policy trained on one fixed corridor - and Stages A-C all ran the same corridor. The policy
would have to learn a rotation-equivariant mapping inside a 2x64 MLP from absolute-frame
inputs, which is a strictly harder problem than any earlier rung posed.

### A hypothesis that was tested and rejected

The obvious response is to stop randomising the absolute orientation and randomise only the
*arc* (the separation), which is all the spec above literally asks for. Measured at FULL
randomisation:

| | C champion | scripted |
|---|---|---|
| absolute orientation random | 0.062 | 0.875 |
| absolute orientation **fixed** | 0.031 | 0.938 |

Fixing it does **not** help. Orientation dominates only while every other knob is held at
the easy anchor; once radii, launch speed, planet count, masses and orbits are all open,
the rest of the randomisation is just as damaging. The full spec therefore stands, and the
narrower reading was dropped rather than shipped as a fix that does not fix anything.

### What would actually unblock it

1. **Add a portal-relative frame to the observation.** The real fix, and the reason it was
   NOT done here: it changes `obs_spec_hash` away from `16a7d04e` and invalidates the
   necessity columns already measured for A, B and C. That is a ladder-wide decision, not
   a Stage D one.
2. **Far more samples.** Stage A needed ~200 generations for ONE fixed universe; learning
   the same skill in every orientation from absolute-frame inputs plausibly needs an order
   of magnitude more than the 120-250 generations tried here.
3. **PPO instead of REINFORCE**, as this rung's own risk note anticipates. The worker
   already emits `logprobs`, so that switch is controller-side only.

## Fleet run and result

60 generations on Daytona, 8 sandboxes, 32 episodes per generation, warm-started from the
Stage C champion, lr 3e-4, curriculum ramp over 30 generations
(`runs/generations/gen_*.json`, `results/metrics.json`).

**Fan-out gate: GREEN, and this is the rung where it finally means what plan.md 5.3 says.**
A recorded generation:

```
gen_40.json   distinct WORLDS: 8    distinct sandbox ids: 8
seeds: 353092, 69218, 440533, 306678, 5626, 296214, 185773, 339154   (x4 flights each)
```

Through Stages A-C the eight sandboxes justified themselves on throughput and gradient
variance alone - only the action draw varied. Here each sandbox is a genuinely different
universe, which is the fleet's *diversity* argument arriving for real.

**Training gate: RED.** Champion on 48 sampled held-out episodes
(`results/heldout_eval.json`):

| | success | mean reward | closest | terminations |
|---|---|---|---|---|
| sampled | **0.042** (2/48) | -64.6 | 1.63 | 25 collision, 21 timeout, 2 portal |
| greedy | 0.042 | -53.3 | 1.83 | 15 collision, 25 timeout, 6 escape, 2 portal |

Against a 0.80 bar. `GATE A8: FAIL` on all three checks - reward did not improve, success
did not rise, the champion does not reach the portal.

**The training made the policy worse, not better.** Generation 1 - which is the Stage C
champion at difficulty 0 - scored **0.62**. By generation 3 it was 0.12, with gradient
norms of 3.6-4.0 against a 1.0 clip. The best Stage D policy available is still the Stage C
champion; nothing this rung learned improved on it.

**The generalisation gap is recorded** (`metrics.json`, `generalisation_gap` per probe) and
reads ~0.0 throughout - but only because train and held-out are both ~0. The metric is
wired and correct; it has nothing to measure until the policy works at all.

## Status

🟥 **Implemented, tested and run end-to-end. Fan-out gate GREEN; training gate RED.**

What is done: the universe sampler (Kepler ellipses, per-seed worlds, curriculum), S2
noise, disjoint train/eval seed blocks, per-world advantage grouping, the generalisation-gap
metric, config and keys, worker and controller plumbing, the Daytona snapshot, and 20
dedicated tests (95 across the suite). The rung runs on the fleet and produces replayable
recorded generations.

What is not done: the policy does not learn these worlds, for the reason documented above.
This is a measured negative result with a localised cause, not an unfinished implementation
- the same shape of finding that, at Stage C, turned out to be the thing worth knowing.
