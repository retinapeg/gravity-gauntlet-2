# Stage A — one fixed star

**Goal.** Establish the **irreducible control set**: the smallest group of Newtonian
variables without which the craft cannot be flown at all.

> **This is the calibration rung, not a demo rung.** Stage A has exactly one universe, so
> its eight sandboxes all run the *same* world — the grid would show eight identical
> cards. That is expected. Stage A's job is to prove the physics, the worker contract, the
> fan-out and the training loop end to end. The eight-visibly-different-universes demo
> begins at **Stage B** (plan.md §7.1).

**Why this rung.** No planets, no motion, no randomness — the only forces are one central
mass and the craft's own thrust. Anything the ablation study finds necessary *here* is
necessary because of `F = ma` and gravity themselves, not because of any hazard,
crowding, or noise added later. That makes column A the baseline every other column is
read against.

## World

One solar mass at the origin (`G = 1`). The craft starts at `r = (2.5, 0)` with
`v = (0, 0.55)` — just under the circular speed of 0.632 — so if it never thrusts it falls
into a decaying arc and misses. The portal sits **outward on the craft's own side** at
`(2.6, 2.7)`, `r_capture = 0.35`, with a capture **speed gate** `v_capture = 0.6` (A7), so
arriving fast is not arriving. Thrust gives `Π₁ = a_thrust/g₀ ≈ 1.9`.

The portal is deliberately NOT across the star: the craft→portal corridor never crosses
the well, so the whole episode is a fight against inward gravity that a proportional
controller can close. The transfer-orbit slingshot is Stage C's job (plan.md §7), not
Stage A's.

Implemented in `gg/universe.py::_stage_a`.

## Predicted necessity (to be confirmed or overturned)

Necessary: `Δr_portal`, `v`, `a_grav`, `fuel`, `t/T`.
Predicted **un**necessary in a single Kepler well: `E`, `L`, `e`, and the field
magnitude/direction pair — all derivable from what's already there.

If Stage A comes back with *everything* unnecessary, the fuel budget is too loose. Tighten
it until control precision actually costs something.

## Steps

Each step has its own gate. Don't batch them.

| # | Step | File | Gate | |
|---|---|---|---|---|
| **A1** | Observation builder — fill the frozen superset from world state; zero-fill every group whose `FIRST_ACTIVE_STAGE` is later than A | `gg/rollout_worker.py::build_observation` | vector length equals `spec.OBS_DIM_2D`; B/C-only groups are exactly zero | ✅ |
| **A2** | Episode loop — nondimensionalisation (`L₀, g₀, T₀`), the §2.5 reward terms, the four terminations | `gg/rollout_worker.py::run_episode` | all four terminations reachable from hand-built states | ✅ **PASS** |
| **A3** | Worker entrypoint — read `GG_INPUT`, verify `obs_spec_hash`, run one episode, print **one** JSON line (plan.md §4.2) | `gg/rollout_worker.py::main` | runs locally; output parses; a wrong hash is rejected loudly | ✅ |
| **A4** | Physics gate — energy drift and terminations | `stage_a_selftest.py`, `tests/test_physics_gate.py` | **drift < 1% over 500 steps with `NoThrust`**; collision and capture both fire | ✅ **PASS** |
| **A5** | Snapshot — Debian slim + numpy + CPU torch + the `gg` package | `controller/build_snapshot.py`, `controller/config.py` | `gravity-gauntlet-worker-v1` exists and is active | ✅ **PASS** |
| **A6** | One real rollout — create from snapshot, `exec`, parse, `delete` in `finally` | `run_one_sandbox.py` | a real `sandbox_id` and a scripted-policy trajectory that reaches the portal | ✅ **PASS** |
| **A7** | Fan-out ×8 + SSE lifecycle + UI adapter + generation recording | `controller/fanout.py`, `controller/adapter.py`, `controller/record.py`, `controller/api.py` | 8 **concurrent** sandboxes, 8 distinct real IDs; events step `CREATING → READY → RUNNING → COMPLETE`; each rollout adapts to the UI payload (`web/UI_SPEC.md` §5); the generation lands in `runs/generations/gen_1.json` | ✅ **PASS** |
| **A8** | REINFORCE — **numpy** MLP behind the existing `act()` interface, learned input gate, baseline, entropy bonus, Adam | `gg/net.py`, `gg/policy.py`, `controller/train.py` | success on held-out keys **rises across generations**; champion reaches the portal reliably | 🟨 |
| **A9** | Necessity study — M1–M4 | `controller/necessity.py` | ranking **stable across 3 re-runs** → writes `results/necessity.json` | 🟨 |

A1–A4 need no network. **Get A4 green before spending a single sandbox** — `run_one_sandbox.py`
runs the same episode remotely, so a physics bug found there costs a sandbox to learn what
`stage_a_selftest.py` would have told you for free.

A1–A3 landed together inside `rollout_worker.py` rather than as separate modules. That's
fine — one file, one episode, one JSON line matches the worker contract. Split it only if
Stage C's moving-body logic makes it unwieldy.

A5 was extracted out of `run_one_sandbox.py` into `controller/build_snapshot.py` (the
Makefile already expected it there), so the worker image is now defined in exactly one
place and the fan-out can call `ensure_snapshot()` on its way in.

## Gate (the rung as a whole)

- A4 green: energy drift < 1% over 500 steps, coasting.
- A8 green: champion reliably captured at the portal, within the A7 speed gate.
- A9 green: necessity ranking stable across 3 re-runs.

## Artifacts

`results/metrics.json` · `results/champion.npz` · `results/necessity.json` (= column A of the
matrix in plan.md §6.1).

## Status

🟨 **In progress. A1–A6 green; A7–A9 remain.**

**A1–A4 — local, no network.** `make test` runs both the human-readable gate and the
pytest suite:

```
[coast]     steps=500  term=timeout    E0=-0.248670  Ef=-0.248673  drift=1.355e-05
[crash]     steps=120  term=collision  reward=-103.52
[escape]    steps=253  term=escape     reward=-55.15
[pursuit]   steps=183  term=portal     success=True  reward=+99.03  fuel_left=0.674
[spec]      obs_dim=38  hash=16a7d04e  checksum=667550d4
GATE: PASS - energy drift < 1%, all four terminals fire, obs spec frozen
16 passed
```

Energy drift is 1.4e-05 — roughly **700× inside** the 1% budget, so the leapfrog integrator
is not the thing that will limit this project. All four terminals now fire from hand-built
states (`escape` was the one the gate claimed but never exercised), and
`tests/test_physics_gate.py` adds the A7 speed-gate check the reward function most needs:
a craft that passes 0.017 from the portal at speed 3.0 is **not** captured.

**A5 — snapshot.** `python -m controller.build_snapshot --verify`:

```
python=3.12.10  numpy=2.5.2  torch=2.13.0+cpu
obs_dim=38  obs_spec_hash=16a7d04e (controller: 16a7d04e)
GATE A5: PASS - snapshot active and the baked worker image is correct
```

The verify step boots a real sandbox and asks the image what it contains rather than
assuming: the baked `gg` reports the **same** `obs_spec_hash` as the controller, which is
what rules out a stale `add_local_dir` copy silently drifting from `spec.py`.

**A6 — one real rollout.** `python run_one_sandbox.py`, sandbox
`e305c7d8-81c0-4b00-9e3d-779477afcca3`: pursuit reached the portal in 183 steps
(`success=True`, `fuel_left=0.674`), coast timed out at 500 steps with drift 1.36e-05, and
the sandbox `physics_checksum` is `667550d4` — **identical to the local one**, so the
physics that runs in Daytona is the physics that runs here.

**A7 — fan-out ×8.** `python -m controller.fanout --stage A -n 8`: one generation, eight
**concurrent** sandboxes, eight distinct real ids, 3.1 s wall, recorded to
`runs/generations/gen_1.json`.

```
generation 1  policy P000  mean_reward=-35.94  success_rate=0.0  wall=3107ms
  worker 0  2b44dcef-3fa2-4ad4-af4c-3d76a8cf9331  timeout  steps=500  reward=-35.98
  ... 8 rows, 8 distinct sandbox ids ...
GATE A7: PASS - M concurrent sandboxes, M distinct ids, replayable payloads
```

The eight rewards all differ, which is the point of the rung: one universe, eight
sandboxes, and the only thing separating them is action sampling (S4). That is already a
variance reduction for the gradient (plan.md §5.3) even though the grid would show eight
identical worlds.

`controller/api.py` serves the same data live (SSE, one event per real Daytona transition)
and in replay (`GET /api/generations/1` returns the recorded payload — 8 rollouts, 501
trajectory points each).

**A8 — REINFORCE.** 200 generations, 8 concurrent sandboxes, 32 episodes per gradient
step, warm pool. The policy learns to fly:

```
gen   1  mean= -36.04  succ=0.00  H=2.197      <- uniform over 9 actions, log(9)=2.197
gen 100  mean= +27.51  succ=0.47  H=1.836      held-out probe 0.88, closest 0.353
gen 103  mean= +98.66  succ=1.00                <- 16/16 rollouts captured
run      mean reward -36.13 -> +14.86,  success 0.00 -> 0.37
```

Held-out evaluation of the champion (48 rollouts, action seeds it never trained on):

| | success | mean reward | closest approach | terminations |
|---|---|---|---|---|
| **sampled** (the deployed policy) | **0.77** | +68.0 | 0.361 | 37 portal, 11 timeout |
| greedy (argmax, diagnostic) | 0.00 | −56.8 | 1.301 | 48 escape |

🟨 **A8 is a near miss on its own gate.** "Reaches the portal reliably" was read as
≥ 0.80 held-out success; the champion measures **0.77**. Learning is not in doubt — 0 % to
77 %, with individual generations at 100 % — but the run does not clear the bar it was
given, and the bar has been left where it was rather than lowered to meet the result.

Two things are worth reading off that table.

**Evaluate the sampled policy, not the argmax one.** S4 (action sampling) is on from
Stage A and every worker draws `a_t ~ Categorical(pi_theta)`, so the *stochastic* policy is
the one that was trained and the one the fleet deploys. Its argmax is a policy that never
existed during training, and here it escapes the arena every time. Earlier runs showed the
mirror image — 100 % greedy with 0 % sampled — which is the same fact from the other side:
while entropy is high the two are simply different policies. Reporting greedy as *the*
number would have been the more flattering choice and the wrong one.

**Training is high-variance, not monotone.** Held-out probes across the run read
0.00 (×9) → 0.88 → 0.50 → 0.63 → 0.75 → 0.63 → 0.50 → 0.00, and the run's best training
generation scored 16/16. Judging this policy from a handful of sampled generations gives
whatever answer the sample happens to support; that is why the champion is selected by a
held-out probe every 10 generations and kept, rather than by taking whatever the last
generation produced.

**The honest next step** is more generations at a lower entropy floor, not a different
algorithm: the gap between 0.77 and "reliably" is variance, and plan.md §3.3 itself budgets
150+ generations for a "real run" against the 25-40 it calls a demo.

**A9 — necessity study.** M1 gate + M2 ablation/permutation (3 repeats × 16 episodes per
arm, on the fleet) + M3 saliency, against a champion whose baseline is return +81.9,
success 0.875.

| group | M1 gate | M2 ΔR(zeroed) | ± | M2 ΔSucc | M3 sal | verdict | predicted |
|---|---|---|---|---|---|---|---|
| `portal_rel_pos` | 0.870 | **38.99** | 3.9 | **0.29** | 0.384 | **necessary** | necessary ✓ |
| `time_fraction` | 0.899 | **22.44** | 4.0 | **0.17** | 1.000 | **necessary** | necessary ✓ |
| `closing_speed_portal` | 0.862 | **19.52** | 7.9 | **0.15** | 0.045 | **necessary** | ~~unnecessary~~ ✗ |
| `fuel_fraction` | 0.892 | **16.77** | 13.5 | **0.12** | 0.728 | **necessary** | necessary ✓ |
| `speed` | 0.867 | 11.14 | 3.9 | 0.08 | 0.128 | marginal | marginal ✓ |
| `a_grav_magdir` | 0.882 | 11.06 | 3.9 | 0.08 | 0.260 | marginal | ~~unnecessary~~ ✗ |
| `orbital_energy` | 0.833 | 8.36 | 6.8 | 0.06 | 0.164 | marginal | ~~unnecessary~~ ✗ |
| `potential` | 0.865 | 8.34 | 6.8 | 0.06 | 0.065 | marginal | marginal ✓ |
| `bearing_error` | 0.823 | 0.03 | 6.8 | 0.00 | 0.237 | unnecessary | unnecessary ✓ |
| **`a_grav`** | 0.871 | **0.00** | 6.8 | 0.00 | 0.165 | **unnecessary** | ~~necessary~~ ✗ |
| `eccentricity_vec` | 0.840 | −0.01 | 6.8 | 0.00 | 0.333 | unnecessary | unnecessary ✓ |
| `angular_momentum` | 0.856 | −0.01 | 6.8 | 0.00 | 0.082 | unnecessary | unnecessary ✓ |
| `control_authority` | 0.842 | −0.03 | 6.8 | 0.00 | 0.034 | unnecessary | unnecessary ✓ |
| **`craft_vel`** | 0.883 | −2.74 | 10.4 | −0.02 | 0.398 | **unnecessary** | ~~necessary~~ ✗ |
| `craft_mass` | 0.835 | −5.54 | 7.9 | −0.04 | 0.186 | unnecessary | unnecessary ✓ |

**Two overturns worth the whole exercise.**

`a_grav` was predicted **necessary** and measures **exactly 0.00** — zeroing the
gravitational acceleration vector costs the policy nothing. `craft_vel` goes the same way.
Meanwhile `closing_speed_portal`, predicted unnecessary, comes back clearly necessary.

Read together these say something specific about Stage A rather than about gravity: with
**one** universe the field along the flown corridor is a fixed function of position, so
`a_grav` carries no information the policy cannot get from `portal_rel_pos` — it is
constant-given-where-you-are. And because A7 gates capture on *speed*, the quantity the
controller actually regulates is the closing speed, not the velocity vector. The policy did
not learn gravity; it learned *this* trajectory. That is precisely why Stage A is the
calibration rung and why the interesting number is the **delta** at Stage B, where the
field stops being a function of position alone.

🟨 **A9 fails its gate as written.** Full-rank Spearman across the 3 re-runs is 0.596
(bar 0.7).

The failure is informative, though, and worth not papering over: the top-3
(`portal_rel_pos`, `closing_speed_portal`, `time_fraction`) is **identical in all three
re-runs**, and every disagreement sits at rank 4 and below — among seven groups whose
ΔReturn is within ±0.03 of zero. Their relative order is noise by construction, and a
full-rank Spearman is dominated by re-shuffling exactly those ties. `controller/necessity.py`
now also reports **verdict-class stability** (does a group keep its necessary/marginal/
unnecessary label on every re-run) and top-3 set identity, which is the question the study
actually asks. The plan's gate has been left as the gate; the extra statistics are reported
beside it, not instead of it.

The other honest caveat: at ±100 for a capture, ΔReturn moves by ~100/n per lucky rollout,
so 16 episodes per arm buys a ±4-13 spread — visible in the ± column. More episodes per arm
is the direct fix and costs only fleet time.

### On `torch`

`torch` cannot be installed on this machine (Intel Mac, Python 3.13 — no wheel exists at
that intersection). Rather than split the environment, the **whole network is numpy**
(`gg/net.py`): forward pass, hand-written backward pass, and Adam. That is option 1 in
`requirements-train.txt`, and it buys something beyond convenience — the controller's
gradient step and the sandbox's forward pass run the *identical* arithmetic, so there is
no training/serving skew to chase. The hand-written gradient is checked against central
finite differences on every parameter block in `tests/test_reinforce.py` (worst relative
error 6.6e-06); that test is the only thing standing between a subtly wrong derivative and
a training run that looks plausible and learns nothing.

CPU torch stays baked in the snapshot: it costs nothing at runtime (the worker never
imports it) and keeps option 3 — training inside a sandbox — open for later stages.

### Daytona quota, the hard way

Two limits bind this organisation, both discovered by hitting them:

| Limit | Symptom | Fix |
|---|---|---|
| 10 GiB total memory | 3 of 8 sandboxes refused: *"Total memory limit exceeded"* | snapshot rebuilt at `memory=1` (was 2) — resources are baked in, the SDK has **no** create-time override, so a fat snapshot silently caps concurrency at 5 |
| 10 total vCPU | a later run died at startup: *"Total CPU limit exceeded"* | a killed run had orphaned 10 sandboxes; `auto_stop_interval=0` meant they could never become eligible for auto-delete either |

`controller/cleanup.py` (`make sandboxes ARGS=--delete`) reaps strays, and the pool now
sets `auto_stop_interval=15` so an interrupted run self-cleans instead of holding the
whole quota hostage.
