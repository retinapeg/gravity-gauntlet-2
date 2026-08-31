# Gravity Gauntlet

One reinforcement-learning policy learns to fly a spacecraft to a portal through a
gravitational field. **Every rollout runs inside a real Daytona sandbox**; a training
generation is `asyncio.gather` over a fleet of them. If Daytona stops, training stops.

We begin in a deterministic 2D world with a single fixed star and add **one physical
complication per stage**. At each rung an ablation study reads back **which Newtonian
variables the policy must observe and alter** to keep flying. That growing list — the
variable-necessity matrix — is the research output.

Full specification: **[plan.md](plan.md)**. This README is the map; each stage folder is
the working checklist.

---

## Repository layout

```
gg/                     the worker package — runs INSIDE the Daytona sandbox
  spec.py               frozen observation superset + reward config + obs_spec_hash
  physics.py            softened gravity (A1), velocity Verlet (A2), invariants
  universe.py           build_universe(stage, key) — one branch per rung
  net.py                the policy network in numpy: input gate + 2x64 tanh + head,
                        forward AND the hand-written gradient (finite-difference checked)
  policy.py             NoThrust / PursuitPolicy / MLPPolicy behind one act() interface
  version.py            source_hash() — catches a stale copy baked into the snapshot
  rollout_worker.py     build_observation + run_episode + entrypoint:
                        GG_INPUT -> one JSON line on stdout

controller/             host side — never simulates, only orchestrates and learns
  config.py             credentials + the snapshot name, read in exactly one place
  build_snapshot.py     bakes gravity-gauntlet-worker-v1 (numpy + CPU torch + gg)
  fanout.py             asyncio.gather over M sandboxes + SandboxPool for the sweeps
  adapter.py            worker payload -> UI payload (thin rename/reshape)
  record.py             writes each generation to runs/generations/
  train.py              REINFORCE + value baseline + entropy bonus (plan.md 3.6)
  necessity.py          M1–M4 ablation study -> the matrix column
  api.py                SSE lifecycle events for the demo UI
  cleanup.py            list / reap stray sandboxes when a run dies before its finally

stages/                 one folder per rung: README + config + keys + results
tests/                  local validation — no Daytona, no network (51 tests)
web/                    the demo UI — UI_SPEC.md is the contract
runs/generations/       recorded generations (genuine) — powers replay mode

smoke_test.py           stage 00 gate — 4 parallel sandboxes
stage_a_selftest.py     stage A gate, run locally: energy drift + all four terminations
run_one_sandbox.py      stage A in ONE real sandbox — builds the snapshot if missing
tests/test_physics_gate.py   the same gate as pytest, plus the A7 speed-gate check
```

**Stages are workflow units, not code silos.** Physics and the observation builder exist
**once**, in `gg/`. `spec.py` freezes the observation layout from Stage A and later rungs
only flip a group from zero-filled to live — that is what keeps necessity scores
comparable across the ladder (plan.md §5.1). Never fork the physics per stage.

---

## The stage ladder

| Stage | World | Determinism | Gate | Status |
|---|---|---|---|---|
| **[00 · smoke](stages/stage_00_smoke/)** | — | — | 4 distinct `sandbox_id`s, all `finished` | ⬜ written, not run |
| **[A · fixed star](stages/stage_a_fixed_star/)** | one fixed central mass | deterministic, single universe | energy drift < 1%; champion reaches the portal; necessity ranking stable ×3 | 🟨 A1–A4 ✅ green |
| **[B · fixed planets](stages/stage_b_fixed_planets/)** | + 2–3 stationary planets | deterministic, fixed layout set | success on held-out layouts; column B stable | ⬜ blocked on A |
| **[C · orbiting planets](stages/stage_c_orbiting/)** | planets on deterministic ellipses | deterministic, randomised phase | success on unseen phases; column C stable — **demo endpoint** | ⬜ blocked on B |
| **[D · stochastic](stages/stage_d_stochastic/)** | randomised n-body + noise | stochastic | plan only — not demoed | ⬜ |
| **[E · 3D lift](stages/stage_e_3d/)** | inclined orbits, ℝ³ | det. → stochastic | plan only — not demoed | ⬜ |
| **[F · POMDP](stages/stage_f_pomdp/)** | + sensor-range masking | stochastic POMDP | plan only — stretch | ⬜ |

Each rung repeats the same five-beat loop:

```
extend universe branch  ->  flip the new obs groups live  ->  train (warm-started)
        ->  run M1–M4 ablation  ->  record the necessity delta vs the previous rung
```

**Do not open a stage until the previous gate is green.**

### One generation = eight *distinct* worlds

A generation fans out over **8 different keys**, not 8 copies of one. "Deterministic" means
*reproducible given the key*, not *a single universe* — so the demo grid shows eight
visibly different gravitational worlds that are each exactly replayable. It also lowers
gradient variance for free, because the batch now spans real environment diversity.

Stage A is the exception: it has one universe by definition, so it is the **calibration
rung** — it proves the physics, the worker contract, the fan-out and the training loop.
The eight-universe grid demo begins at **Stage B**.

## The demo UI

**[web/UI_SPEC.md](web/UI_SPEC.md)** is the UI contract.

> One universe gives an agent one experience. Daytona gives it many universes at once.

Daytona calculates; the browser renders. The UI replays recorded rollouts — it never
simulates, never invents a position, and never fabricates a trajectory, reward, sandbox ID
or generation result. Every live generation is recorded to `runs/generations/`, so replay
mode is a free fallback that still shows genuine data if the API is flaky during judging.

---

## Quickstart

```bash
make setup                    # .venv + requirements
make test                     # stage A gate, locally: energy drift + all four terminations
make smoke                    # stage 00 gate: 4 real Daytona sandboxes in parallel
make snapshot                 # step A5: bake gravity-gauntlet-worker-v1
python -m controller.build_snapshot --verify   # A5 gate: boot a sandbox, check what is baked
python run_one_sandbox.py     # stage A in one REAL sandbox (builds the snapshot if missing)
make help                     # every target
```

Credentials come from `.env` (`DAYTONA_API_KEY`, `DAYTONA_API_URL`), which is gitignored.

**On Daytona quotas.** This organisation allows **10 vCPU and 10 GiB total across live
sandboxes**, so with the snapshot's `cpu=1, memory=1` the ceiling is 10 concurrent —
which is why `M = 8` fits and why the snapshot is not provisioned larger than the worker
needs. Two traps, both hit and both now fixed in code: a snapshot baked at `memory=2`
silently caps you at 5 concurrent sandboxes (resources are baked in; the SDK has no
create-time override), and a run killed before its `finally` leaves sandboxes alive
holding the whole quota, so the *next* run fails with "Total CPU limit exceeded" for
reasons unrelated to its own code. `make sandboxes` lists them; `make sandboxes
ARGS=--delete` reaps them.

**On `torch`:** deliberately not in `requirements.txt`. This dev machine is an Intel Mac
(darwin x86_64) on Python 3.13, and PyTorch's last macOS x86_64 release is 2.2.2, which
supports Python ≤3.11 — no wheel exists at that intersection. It is only needed for the
trainer (step A8); the physics, worker, fan-out and API all run on numpy alone. The
Daytona snapshot is Debian linux, where torch installs normally — and it is now baked in
and verified importable there (2.13.0+cpu), so the sandbox is currently the only place a
torch MLP can run at all. See [`requirements-train.txt`](requirements-train.txt) for the
three ways forward.

## Where things stand

The whole `gg/` worker package is written and plan-faithful: `spec.py` (frozen observation
layout + hash), `physics.py` (A1/A2 + invariants), `universe.py` (Stage A branch),
`policy.py` (`NoThrust`, `PursuitPolicy`), and `rollout_worker.py` (observation builder,
episode loop, JSON entrypoint). `stage_a_selftest.py`, `tests/test_physics_gate.py` and
`run_one_sandbox.py` cover the Stage A gate and the first real sandbox.

**Stage A steps A1–A6 are green.** The local gate (`make test`):

```
[coast]     steps=500  term=timeout    E0=-0.248670  Ef=-0.248673  drift=1.355e-05
[crash]     steps=120  term=collision  reward=-103.52
[escape]    steps=253  term=escape     reward=-55.15
[pursuit]   steps=183  term=portal     success=True  reward=+99.03  fuel_left=0.674
GATE: PASS - all four terminals fire, obs spec frozen        (+ 16 passed)
```

Energy drift is ~700× inside the 1% budget, all four terminations fire from hand-built
states, and the A7 speed gate correctly refuses a craft that passes 0.017 from the portal
at speed 3.0 — the physics is not going to be what limits this project.

The snapshot (A5) is baked and verified from the inside — `python=3.12.10 numpy=2.5.2
torch=2.13.0+cpu`, and the baked `gg` reports the same `obs_spec_hash` as the controller,
which is what rules out a stale copy. One real rollout (A6) reached the portal in 183 steps
with a `physics_checksum` identical to the local one.

**Stage A steps A1–A9 are all implemented and run**; **Stages B and C are implemented and
running end-to-end on the fleet.** Every rung's infrastructure gate is green — 8 concurrent
sandboxes, 8 distinct real ids, and from Stage B **8 distinct worlds** per generation:

| rung | fan-out gate | training gate | champion (held-out, sampled) |
|---|---|---|---|
| **A** — one fixed star | ✅ | 🟨 0.77 vs a 0.80 bar | 0.77 |
| **B** — fixed planets | ✅ 8 distinct layouts | ❌ success did not rise | 0.38 on layouts never trained on |
| **C** — orbiting bodies | ✅ 8 distinct (layout, phase) | ❌ | 0.00 — arrives at the portal, too fast to be captured |

The learning results are honest partials and are written up as such in each stage README,
with the measured cause in each case. What is *not* partial is the machinery: physics,
the frozen observation superset, the worker contract, the snapshot, the fan-out, the
adapter, generation recording, the SSE API, a finite-difference-checked numpy REINFORCE
gradient, and the M1–M4 necessity study — all of it runs against real Daytona sandboxes,
and `runs/generations/gen_{1,2,3}.json` replay Stages A, B and C exactly.

Still open: the UI (`web/` has only its spec), Stages D–F, and the training budget needed
to turn the partials into passes.
