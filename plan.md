# Gravity Gauntlet — Project Plan

**One policy, many universes.** A single reinforcement-learning policy learns to fly a
spacecraft to a portal through a gravitational field. Every rollout runs inside a real
Daytona sandbox created from a prepared snapshot; a training generation is
`asyncio.gather` over a fleet of them. We start in a **deterministic 2D** world with a
single fixed star and add exactly **one physical complication per stage**; at each stage an
ablation study reads back **which Newtonian state variables the policy must observe and
alter** to keep flying. That growing list is the research output.

- **Demo endpoint:** deterministic 2D with orbiting planets (Stage C) + the 8-way parallel
  Daytona fan-out and lifecycle UI.
- **Plan only (not demoed):** stochastic training (Stage D), the 3D lift (Stage E), partial
  observability (Stage F).
- **Companion artifacts:**
  [RL Plan](https://claude.ai/code/artifact/bb02de5d-e297-411d-b100-f21be1ad8809) ·
  [Newtonian Variable Ladder](https://claude.ai/code/artifact/fd74c422-edfe-4b9c-bda5-355795dc72bc)

---

## 1. The learning objective

A spacecraft starts somewhere in a randomly (later: deterministically) generated solar
system with limited fuel and a fixed step budget. Its job: reach a **portal** without
hitting a body and without escaping to infinity. It has never seen this exact universe.

The agent is a policy `π_θ(a | o)` mapping an **observation vector** `o_t` to a **thrust
action** `a_t`. Training has two coupled goals:

1. **Control.** Find `θ` such that expected return is high across the whole universe
   distribution — a genuine feedback law, not one memorised trajectory.
2. **Attribution.** Determine which components of `o_t` carry the control signal. We feed
   the policy a deliberately over-complete state and use the trained policy plus targeted
   ablations to rank every variable by necessity.

**Testable thesis.** The necessary variables are a small dimensionless set:
portal-relative position and velocity, the net gravitational acceleration vector and
potential, the nearest body's range and mass, and the fuel and time fractions — roughly
12–16 scalars in 2D. Enumerating every body is predicted to be **unnecessary**: the
aggregated field plus the nearest-body term is sufficient. The 3D lift is predicted to add
inclination, the angular-momentum vector direction, and the out-of-plane field component.

---

## 2. The environment (MDP)

A discounted MDP `(S, A, P, r, γ)` with a designed observation function. The true state
(positions/velocities of the craft and all bodies) is Markov; the observation is a
designed projection of it — a mild POMDP, deliberately, because the projection is the
object of study.

### 2.1 Observation vector — the variable catalogue

Over-complete, grouped by physical role, **nondimensionalised** (§2.2). "Necessity" is a
hypothesis tested by the ablation study, not an assertion.

| Variable | Symbol | Comp. 2D | Group | Why it might matter | Necessity | In 3D |
|---|---|---|---|---|---|---|
| Portal-relative position | `Δr_p = r_p − r` | 2 | kinematic | Where to go; the primary error signal | essential | 3 |
| Craft velocity | `v` | 2 | kinematic | Inertia; you steer momentum | essential | 3 |
| Speed | `|v|` | 1 | kinematic | Capture gate; kinetic energy | likely | 1 |
| Closing speed to portal | `−d‖Δr_p‖/dt` | 1 | kinematic | Instantaneous progress rate | likely | 1 |
| Bearing error | `∠(v, Δr_p)` | 1 | kinematic | Steering cue (derivable) | test | 2 |
| Net gravitational acceleration | `a_grav` | 2 | field | The dominant external disturbance | essential | 3 |
| Field magnitude & direction | `|a_grav|, φ_g` | 2 | field | Redundant with the vector; helps a small net | test | 2 |
| Gravitational potential | `Φ = −Σ G M_i / d_i` | 1 | field | Energy landscape; bound vs. escape | likely | 1 |
| Nearest body: vector, range, mass | `(r_n−r), d_n, M_n` | 4 | field | Collision avoidance; the acute hazard | essential | 5 |
| Time to closest approach | `TCA_n` | 1 | field | Act early, not reactively | test | 1 |
| Nearest-body relative velocity | `v_n − v` | 2 | field | Anticipate a moving hazard | test (C+) | 3 |
| Field rate of change | `dΦ/dt` | 1 | field | Time-varying field cue | test | 1 |
| Body count | `N` | 1 | field | How crowded the world is | test | 1 |
| Per-body top-k features | `{(r_i−r), M_i}_{1..k}` | 3k | field | Full field detail — thesis: **not** needed | test | 4k |
| Fuel fraction | `f = m_fuel / m_fuel,0` | 1 | resource | Hard budget; changes optimal aggression | essential | 1 |
| Craft mass | `m` | 1 | resource | Thrust → accel via `F = ma`; drops as fuel burns | likely | 1 |
| Control authority | `Π₁ = a_thrust^max / g₀` | 1 | resource | How much the craft can defy gravity | likely | 1 |
| Time fraction elapsed | `t / T_budget` | 1 | clock | Urgency; end-game behaviour | likely | 1 |
| Specific orbital energy | `E = ½|v|² + Φ` | 1 | invariant | Engineered; is the trajectory bound? | test | 1 |
| Angular momentum | `L = r × v` | 1 | invariant | Orbit shape; scalar 2D, **vector** 3D | test | 3 |
| Eccentricity vector | `e` | 2 | invariant | Orbit geometry vs. dominant mass | test | 3 |
| Orbital inclination | `i` | — | 3D geom | Out-of-plane orientation | — | 1 |
| Right ascension of node | `Ω` | — | 3D geom | Node line orientation | — | 1 |
| Out-of-plane field component | `a_grav,z` | — | 3D geom | Cross-plane pull | — | 1 |

2D total ≈ **16 core + 3k optional** (≈25 with `k=3`). 3D core ≈ 28 + 4k, plus the three
genuinely new geometry variables.

### 2.2 Nondimensionalisation

Per episode, at `t=0`, compute characteristic scales and divide everything by them:

- **Length** `L₀` = initial craft–portal distance.
- **Acceleration** `g₀ = |a_grav|` at `t=0` (fallback `G M_tot / L₀²`).
- **Time** `T₀ = √(L₀ / g₀)`; **velocity** scale `L₀ / T₀`.

The policy then operates on dimensionless Π-groups, so a tiny fast system and a vast slow
one look identical. This is itself a partial answer: the necessary variables are **ratios,
not absolutes**.

### 2.3 Action space

- **2D:** discrete, `|A| = 9` — no thrust, or a fixed-magnitude burn in one of eight
  compass directions (`Linear(9)` head). Each burn costs fuel `∝ |a_thrust|·dt` and
  reduces craft mass (A5).
- **3D options** (preference order for the demo):
  1. **Factored discrete** — three independent categorical heads, one per axis, each
     `{−, 0, +}`. 9 logits, `3³ = 27` effective actions, log-prob = sum across heads.
  2. **Flat discrete 7** — rest plus `±x, ±y, ±z`. Simplest; no diagonals.
  3. **Continuous** — a diagonal Gaussian over a clipped thrust 3-vector. Most expressive;
     use only if discrete plateaus.

### 2.4 Dynamics

The field is **time-varying**: massive bodies orbit a central star (Stage C+). Integrate
with a symplectic scheme (velocity Verlet / leapfrog) for energy stability over ~500 steps
(A2). Softening `ε` removes the `1/r²` singularity; sub-step near close approaches; clamp
velocities to a sane multiple of local escape speed so NaNs never enter a gradient. The
force model (A1) is used unchanged in 3D; only the vector length changes.

### 2.5 Reward

Dense potential-based shaping for a usable gradient; sparse terminals for the real
objective. Shaping cannot change the optimal policy and is capped per step so it cannot be
farmed by oscillating.

| Term | Expression | Sign | Purpose | Guard | Start |
|---|---|---|---|---|---|
| Progress shaping | `k_p (d_{t−1} − d_t)` | + | Dense pull toward the portal | Potential-based; clip `±c`/step | `k_p = 1.0` |
| Proximity penalty | `−k_x exp(−(d_n/ρ)²)` | − | Smooth collision-risk gradient | `ρ ≈ 3 × body radius` | `k_x = 0.3` |
| Fuel cost | `−k_f · Δfuel_t` | − | Economy; discourage idle burns | — | `k_f = 0.05` |
| Time cost | `−k_t` per step | − | Urgency | Keep small vs. shaping | `k_t = 0.01` |
| Capture bonus | `+R_goal` | + | Terminal success | Requires the speed gate (A7) | `100` |
| Collision | `−R_crash` | − | Terminal failure on contact | — | `100` |
| Timeout | `−R_timeout` | − | Ran out of steps | — | `30` |
| Escape | `−R_escape` | − | Left the arena / unbound past a radius | — | `50` |

### 2.6 Stochasticity — four sources

| ID | Source | Where | Effect |
|---|---|---|---|
| **S1** | Universe randomisation (per seed) | `N ∈ {2..5}`, masses log-uniform, orbits, portal, start pose | Epistemic variety — the thing the fleet parallelises. **Off for Stages A–C.** |
| **S2** | Process noise (per step) | additive impulse `η_t ∼ N(0, σ²I)`; multiplicative thrust misfire `(1+δ)` + pointing error | Forces a closed-loop controller. **Off for A–C.** |
| **S3** | Observation noise (per step) | `o_t = o_t^true + N(0, Σ)`; optional sensor range → POMDP | Motivates a recurrent policy (Stage F). **Off for A–D.** |
| **S4** | Policy stochasticity (per step) | `a_t ∼ Categorical(π_θ(o_t))` | Exploration + the gradient signal. **On from Stage A.** |

Consequence: the return is a random variable; the policy gradient estimates its
expectation; the parallel Daytona rollouts are its Monte-Carlo samples. More sandboxes per
generation → lower-variance gradient. This holds even in the deterministic stages (only S4
active), which is why the fleet is justified from Stage A.

---

## 3. The RL algorithm — REINFORCE with a baseline

Simplest thing that is unambiguously RL: **REINFORCE** + advantage baseline + entropy
bonus, ≈20 lines of PyTorch. Upgrade path: a clipped PPO objective (localised change; the
worker output already carries `logprobs`).

### 3.1 Policy network

```
o (d)  →  Linear(d, 64)  →  Tanh
       →  Linear(64, 64) →  Tanh
       →  Linear(64, |A|)          # logits; softmax for the Categorical
       [ optional Linear(64, 1) value head V(o) for the baseline ]
       [ input gate:  o' = o ⊙ sigmoid(g),  g learnable  →  M1 necessity probe ]
```

~6–10k parameters. Serialised (`torch.save(state_dict)` → base64) it is a few KB — cheap
to ship into every sandbox as an env var. The trunk is **dimension-generic**; only
`in_dim` and the head change between 2D and 3D. Warm-start each stage from the previous
stage's champion.

### 3.2 One generation

1. Freeze `θ_k`. Serialise weights.
2. Launch `M` Daytona sandboxes from the snapshot, each with `(θ_k, key_j, cfg)`;
   `asyncio.gather`.
3. Each returns a trajectory `(o_t, a_t, r_t)_{t=0..T_j}` plus scalars.
4. Discounted returns `G_t = Σ_{τ≥t} γ^{τ−t} r_τ` (A3).
5. Baseline: batch-mean of `G_0` to start; add a fitted `V_φ(o)` (MSE to `G_t`) if variance
   is high. Advantage `A_t = G_t − b`.
6. Normalise `A` across the batch (zero mean, unit std).
7. Loss = policy-gradient term − entropy bonus (A4). Adam, `lr 3e-4`, grad-norm clip 1.0.
8. `θ_{k+1}`. Log metrics, animate, repeat.

Vanilla REINFORCE is on-policy — **one** gradient step per batch. Reuse a batch only by
switching to the PPO clipped ratio.

### 3.3 Hyperparameters

| Name | Demo | Real run | Note |
|---|---|---|---|
| Sandboxes / generation `M` | 8 | 32–64 | Bounded by Daytona concurrency quota |
| Max steps `T` | 500 | 500–800 | Longer horizon in 3D |
| Generations | 25–40 | 150+ | Visible climb by ~gen 15 |
| Discount `γ` | 0.99 | 0.99 | — |
| Entropy `β` | 0.01 → 0.001 | same | Linear decay over generations |
| Optimiser | Adam, `lr 3e-4`, grad clip 1.0 | | Lower `lr` if generation-to-generation KL spikes |

### 3.4 Reproducibility

Every sandbox seeds NumPy and Torch from its own key; universe generation is a pure
function of it. Persist the key list per generation so any world — especially the champion
— replays exactly for the UI.

### 3.5 Estimator

```
∇_θ J(θ) ≈ (1 / Σ_j T_j) · Σ_{j,t} A_{j,t} ∇_θ log π_θ(a_{j,t} | o_{j,t})
```

### 3.6 Update step in code

```python
# rollouts: list of dicts returned from Daytona (drop the failed ones)
obs  = torch.tensor([o for r in rollouts for o in r["observations"]], dtype=torch.float32)
acts = torch.tensor([a for r in rollouts for a in r["actions"]])
rets = torch.tensor([g for r in rollouts for g in discounted(r["rewards"], gamma)])
adv  = (rets - rets.mean()) / (rets.std() + 1e-8)          # batch-mean baseline

logits = policy(obs)
logp   = torch.log_softmax(logits, -1).gather(1, acts[:, None]).squeeze(1)
probs  = torch.softmax(logits, -1)
ent    = -(probs * torch.log_softmax(logits, -1)).sum(-1).mean()
loss   = -(adv * logp).mean() - beta * ent

opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
opt.step()                                                 # one step per batch (on-policy)
```

---

## 4. Daytona execution architecture

Three artefacts, one strict interface. The controller and worker share a single `spec.py`
(observation layout + reward config); the worker rejects a mismatched `obs_spec_hash` so
they can never silently drift.

> API shapes below follow the current Daytona async SDK
> ([AsyncDaytona](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/),
> [AsyncSnapshot](https://www.daytona.io/docs/en/python-sdk/async/async-snapshot/)).
> Verify method names against the installed SDK version.

### 4.1 Snapshot — `gravity-gauntlet-worker-v1`

Everything the episode needs, pre-baked so a sandbox does no `pip install` at runtime.

```python
from daytona import AsyncDaytona, CreateSnapshotParams, Image, Resources

image = (
    Image.debianSlim("3.12")
         .pipInstall([
             "numpy",
             "torch --index-url https://download.pytorch.org/whl/cpu",  # CPU only
             "pydantic",
         ])
         .addLocalDir("gg", "/opt/gg")   # universe.py physics.py policy.py rollout_worker.py spec.py
)

async with AsyncDaytona() as d:
    await d.snapshot.create(
        CreateSnapshotParams(
            name="gravity-gauntlet-worker-v1",
            image=image,
            resources=Resources(cpu=1, memory=2),
        ),
        on_logs=lambda c: print(c, end=""),
    )
```

Phase E rebuilds this as `…-v2` with the 3D action space and three.js-free 3D physics.
If `addLocalDir` is not in your SDK build, bake the files with `runCommands` or ship the
`gg` package on a Daytona **Volume**.

### 4.2 Worker I/O — JSON in, one JSON line out

```jsonc
// INPUT  (env var GG_INPUT or stdin)
{
  "stage": "C",              // A | B | C | D | E | F  — selects the universe branch
  "deterministic": true,     // true for A–C: no process noise, no obs noise
  "layout_id": 3,            // A–C: index into the fixed layout set
  "orbital_phase": 0.41,     // C: planet phase at t=0, randomised per rollout
  // "seed": 18473,          // D onward, replaces layout_id / orbital_phase
  "dims": 2,
  "policy_version": 20,
  "policy_weights_b64": "<base64(torch.save state_dict)>",
  "obs_spec_hash": "a1b2c3d4",
  "max_steps": 500,
  "gamma": 0.99,
  "dt": 0.02,
  "noise": { "process_sigma": 0.0, "thrust_jitter": 0.0, "obs_sigma": 0.0 }
}
```

```jsonc
// OUTPUT  (single line on stdout)
{
  "sandbox_id": "sbx_7ca2f091",
  "stage": "C", "layout_id": 3, "seed": null,
  "policy_version": 20, "dims": 2,
  "reward": 318.4, "return_g0": 271.9,
  "success": false,
  "termination": "collision",          // portal | collision | timeout | escape
  "steps": 214, "fuel_left": 0.41,
  "observations": [[/* o_t, length d */], "..."],
  "actions":      [3, 3, 0, 5, "..."],
  "logprobs":     [-1.20, -1.14, "..."],
  "rewards":      [0.42, 0.51, "..."],
  "trajectory":   [[0.0, 12.4, -3.1, 1.0], "..."],   // t, x, y(, z), fuel  — for the browser
  "bodies_trajectory": [[[/* x, y, M per body */]], "..."],
  "wallclock_ms": 1840,
  "physics_checksum": "9f3c7a10"
}
```

The browser renders from `trajectory` + `bodies_trajectory`. The controller learns from
`observations` / `actions` / `rewards` / `logprobs`. **Daytona never renders — it only
calculates.**

### 4.3 Controller

```python
import asyncio, json
from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

SNAPSHOT = "gravity-gauntlet-worker-v1"

async def run_rollout(daytona, weights_b64, key, version, stage="C", dims=2):
    payload = {
        "stage": stage, "deterministic": stage in ("A", "B", "C"),
        "dims": dims, "policy_version": version,
        "policy_weights_b64": weights_b64, "max_steps": 500, "gamma": 0.99,
    }
    payload.update(key)  # {"layout_id": ., "orbital_phase": .} or {"seed": .}

    sbx = await daytona.create(CreateSandboxFromSnapshotParams(
        snapshot=SNAPSHOT,
        auto_stop_interval=0,        # no idle window — run and die
        auto_delete_interval=5,
        env_vars={"GG_INPUT": json.dumps(payload)},
    ))
    try:
        res = await sbx.process.exec("python -m gg.rollout_worker", timeout=120)
        return json.loads(res.result)          # worker prints one JSON line
    finally:
        await daytona.delete(sbx, wait=False)

async def run_generation(weights_b64, version, keys, stage="C"):
    async with AsyncDaytona() as daytona:
        tasks = [run_rollout(daytona, weights_b64, k, version, stage) for k in keys]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### 4.4 Lifecycle events → UI

Emit a server-sent event at each transition so the world cards show real state:

```
{world_id, seed|layout_id, phase: "creating" | "live" | "running" | "finished" | "error",
 sandbox_id, reward?, termination?}
```

Card sequence per world:
`◌ creating` → `● sandbox live  sbx_…` → `🚀 running episode` →
`💥 collision  R−238` → `✓ result collected`. Every card maps to one real `sandbox_id`.

### 4.5 Fault handling

- A sandbox that errors or times out → drop that sample from the batch (never let a NaN
  trajectory into the gradient), mark the card `✗`, continue. `return_exceptions=True`
  keeps `gather` alive.
- Require `≥ M/2` successful rollouts before applying an update; else retry the generation.
- Weights over ~100 KB: write to a file via `sbx.fs` or mount a Volume instead of an env var.

### 4.6 Cost & latency

- Sandbox create ≈ 1–3 s; the pre-baked snapshot means no runtime install.
- Prefer **fewer, larger generations** over keeping sandboxes warm — cleaner story, lower
  cost. If create latency dominates, keep a warm pool or use `linked_sandbox`.
- Small `auto_delete_interval` so orphans self-clean. Workers are CPU-only.
- The published Daytona RL example scales this exact pattern to 256 parallel sandboxes —
  `M` is a quota question, not an architecture one.

---

## 5. The deterministic-first curriculum — the Newtonian variable ladder

Six rungs, one gate each. Do not open a stage until the previous gate is green. The
observation superset is **frozen from Stage A**; later stages only switch components from
constant/zero to live, so necessity scores stay directly comparable. Warm-start each
stage's trunk from the previous champion.

| Stage | D | World | Determinism | Newly in play | Predicted necessity delta | Demo |
|---|---|---|---|---|---|---|
| **A** · fixed star | 2D | One fixed central mass. Portal fixed. Craft starts from one fixed pose on a partial orbit. | det., single universe | `Δr_p, v, a_grav` (1 body), `Φ, f, t`; engineered `E, L, e` | Necessary = `{Δr_p, v, a_grav, f, t}` — the irreducible control set. `E, L, e` predicted unnecessary in a single Kepler well. | ✔ |
| **B** · + fixed planets | 2D | Star + 2–3 stationary planets. Multi-well but static field. | det., small fixed layout set (2 held out for eval) | nearest-body vector / range / mass; per-body top-k; `N` | + nearest-body range & mass (collision hazard). Per-body enumeration predicted **not** necessary — aggregate `a_grav` + nearest term suffices. | ✔ |
| **C** · orbiting planets | 2D | Planets orbit on fixed deterministic ellipses. Time-varying field. No inter-planet coupling. | det. ellipses; fixed layouts; **randomised orbital phase per rollout** | `TCA_n`, nearest-body relative velocity, `dΦ/dt` | + `TCA_n` and/or nearest-body relative velocity — the hazard now moves, so an anticipatory variable becomes necessary. `L, e` turn informative. | ✔ **endpoint** |
| **D** · randomised + noise | 2D | Per-seed universe generation (S1); process noise (S2); optional obs noise (S3). | **stochastic** | control authority `Π₁`, fuel margin; generalisation-gap metric | Nearest-body term up-weights; distant / global-field features down-weight. Robustness promotes `Π₁` and fuel margin to necessary. | plan only |
| **E** · 3D lift | **3D** | Every vector `ℝ²→ℝ³`. Orbits gain random inclination. Action → factored 27 or continuous. Snapshot `…-v2`; three.js rendering. | det. then stochastic | `z`-components, inclination `i`, RAAN `Ω`, `L`-vector direction, `a_grav,z` | The variables necessary **only** out of plane: `i`, the direction of `L`, the out-of-plane field component. | plan only |
| **F** · partial observability | 3D | Sensor range hides distant bodies; heavier process noise. | stochastic POMDP | recurrent hidden state | Local / recent observations dominate; swap the MLP trunk for a small GRU. Learned gate re-weights toward the nearest-body term. | plan only (stretch) |

### 5.1 Per-stage checklist

- **Universe.** Extend `universe.build_universe(stage, key)` with the new generation
  branch. A–C take a `layout_id` (+ `orbital_phase` for C); D onward take a `seed`.
- **Observation.** Flip the new components from constant/zero to live. Never remove a
  component — the builder is a frozen superset.
- **Train.** Warm-start `θ` from the previous champion; run REINFORCE until the gate metric
  plateaus.
- **Attribute.** Run M1–M4 (§6) on the champion. Record the necessity column and the
  **delta vs. the previous stage** — that delta is the logged headline.
- **Gate.** Success on held-out layouts/seeds is high and stable, **and** the necessity
  ranking is stable across 3 re-runs.

### 5.2 `universe.py` — one new function per stage

```python
def build_universe(stage, key):
    if stage == "A":                       # one fixed star, fixed start
        return one_star(), fixed_portal(), fixed_start()
    if stage == "B":                       # + fixed planets, small fixed layout set
        return FIXED_LAYOUTS[key]           # key = layout_id
    if stage == "C":                       # planets orbit; key = (layout_id, phase)
        u = FIXED_LAYOUTS[key.layout_id]
        return advance_orbits(u, key.phase)
    if stage in ("D", "E", "F"):           # key = seed; full randomisation
        return sample_universe(seed=key, dims=3 if stage in ("E", "F") else 2)
```

### 5.3 Why deterministic stages still use the fleet

With a fixed universe the only randomness is S4 (action sampling), but that is enough: `M`
sandboxes running the same layout with independent action samples are Monte-Carlo draws of
the same return expectation, so more sandboxes still means a lower-variance gradient. The
*diversity* argument for the fleet kicks in at Stage D; the *throughput and variance*
argument holds from Stage A. The Daytona story stays literal at every rung of the demo.

### 5.4 What changes at the 2D → 3D boundary (Stage E)

| Aspect | 2D — `ℝ²` | 3D — `ℝ³` |
|---|---|---|
| Position / velocity | `(x,y)`, `(v_x,v_y)` | `+ z`, `+ v_z` — one extra component each |
| Gravitational acceleration | 2-vector, formula A1 | 3-vector, **formula A1 unchanged** |
| Angular momentum | scalar `L = x v_y − y v_x` | vector `L = r × v` → introduces inclination and node |
| Orbit description | `a, e, ν` | `a, e, i, Ω, ω, ν` — two new necessary-in-3D variables |
| Action space | 9 (rest + 8 compass) | 7, or factored `3³ = 27`, or continuous 3-vector |
| Observation dim `d` | ~16–25 | ~28–40 |
| Integrator | leapfrog on 2-arrays | identical code on 3-arrays |
| Universe generation | coplanar orbits | random orbital inclinations |
| Rendering | 2D canvas, top-down | three.js, orbiting camera |
| Training cost | fast | ~1.5–2× samples for equivalent coverage |

`physics.py` and the policy trunk are written dimension-generic from Stage A, so the lift
is a config change + a new action head + a new renderer — not a rewrite.

---

## 6. Which variables are necessary? — methodology

Run these every few generations on the current champion; pipe results to a UI panel (bar
chart of importance per variable group, updating as training proceeds).

- **M1 — Learned input gating (always on).** First layer `o' = o ⊙ σ(g)`, `g ∈ ℝ^d`
  learnable (per variable or per group), plus `λ‖σ(g)‖₁` in the loss. After training,
  `σ(g_i) → 0` ⇒ unnecessary; `→ 1` ⇒ load-bearing. The network emits a necessity score
  per variable.
- **M2 — Ablation & permutation importance (the headline number).** Freeze the champion;
  on a fixed held-out key set, re-evaluate with variable `i` (a) zeroed and (b) shuffled
  across the batch. `ΔReturn` = importance. Per variable and per group.
- **M3 — Gradient saliency.** Average `|∂ logits / ∂ o_i|` (or integrated gradients) over
  champion trajectories. Fast, per-timestep; render as a heatmap along the trajectory.
- **M4 — Representation-sufficiency experiments.** Train separate policies on nested feature
  sets and compare final success rate + sample efficiency:

  | Set | Features | Prediction |
  |---|---|---|
  | A | kinematics + portal only (no gravity) | fails near massive bodies |
  | B | A + `a_grav` and `Φ` only (no per-body data) | **likely sufficient** |
  | C | B + nearest-body vector / mass / TCA | sufficient, matches full set |
  | D | C + per-body top-k | no improvement over C |
  | E | D + engineered invariants `E, L, e` | faster convergence, same ceiling |

  The smallest set whose learning curve matches the full set is the **empirically
  necessary variable set**. Thesis predicts **B or C**.
- **M5 — Buckingham Π cross-check.** Independently derive the dimensionless groups that
  *must* govern the problem (A6). If M1–M4 flag the same quantities, the agent has, in
  effect, rediscovered the natural variables of orbital mechanics.

### 6.1 The variable-necessity matrix (predicted; to be replaced by measured)

Legend: `●` necessary (ablation drops success materially) · `◐` marginal / speeds
convergence only · `○` available, predicted unnecessary · `·` not present at this stage.

| Variable | A | B | C | D | E |
|---|:--:|:--:|:--:|:--:|:--:|
| Portal-relative position `Δr_p` | ● | ● | ● | ● | ● |
| Craft velocity `v` | ● | ● | ● | ● | ● |
| Speed `|v|` | ◐ | ◐ | ● | ● | ● |
| Closing speed to portal | ○ | ○ | ◐ | ◐ | ◐ |
| Net gravitational acceleration `a_grav` | ● | ● | ● | ● | ● |
| Field magnitude & direction | ○ | ○ | ○ | ◐ | ◐ |
| Gravitational potential `Φ` | ◐ | ◐ | ◐ | ◐ | ◐ |
| Nearest body: vector, range, mass | · | ● | ● | ● | ● |
| Time to closest approach `TCA_n` | · | ○ | ● | ● | ● |
| Nearest-body relative velocity | · | · | ● | ● | ● |
| Field rate of change `dΦ/dt` | · | · | ◐ | ◐ | ◐ |
| Body count `N` | · | ○ | ○ | ○ | ○ |
| Per-body top-k features | · | ○ | ○ | ○ | ○ |
| Fuel fraction `f` | ● | ● | ● | ● | ● |
| Craft mass `m` | ○ | ○ | ◐ | ◐ | ◐ |
| Control authority `Π₁` | ○ | ○ | ○ | ● | ● |
| Time fraction elapsed `t/T` | ● | ● | ● | ● | ● |
| Specific orbital energy `E` | ○ | ○ | ◐ | ◐ | ◐ |
| Angular momentum `L` (scalar→vector) | ○ | ○ | ◐ | ◐ | ● |
| Eccentricity vector `e` | ○ | ○ | ○ | ○ | ○ |
| Orbital inclination `i` | · | · | · | · | ● |
| Right ascension of node `Ω` | · | · | · | · | ◐ |
| Out-of-plane field component `a_grav,z` | · | · | · | · | ● |

Each `●` is a hypothesis the corresponding stage's ablation study will confirm or overturn.
Reporting measured vs. predicted is the result.

---

## 7. Demo composition — Stage C

- **Champion panel.** The Stage-C champion flies the champion world (one fixed layout,
  fixed orbital phase) — a clean slingshot to the portal; fuel and success read out live.
- **Live Daytona parallel universes.** Press *Run generation* → 8 sandboxes spawn from the
  snapshot, each a Stage-C layout at a different orbital phase and start offset. Cards step
  `◌ creating → ● sandbox live sbx_… → 🚀 running → ✓ collected` on real lifecycle events.
- **Variable-necessity panel.** The matrix from §6.1, columns A–C filled with **measured**
  values from the ablation runs, as a bar chart of importance per variable group. Gravity
  forces `a_grav`; planets force the nearest-body term; motion forces an anticipatory term.
- **Learning overlay.** Faint past-generation trajectories under the champion path — grey
  spaghetti at gen 1, curved near-misses by gen 10, a clean transfer by gen 20.

Stages D, E, F appear only as this document + the pre-filled (predicted) matrix columns. No
stochastic or 3D training runs during the demo.

### 7.1 One generation = eight *distinct* keys

A generation fans out over **8 different worlds**, not 8 copies of one. "Deterministic"
means *reproducible given the key*, not *a single universe*:

```
GEN 12 · policy P012 frozen · 8 distinct keys
  U1 layout 0  phase .07      U5 layout 4  phase .61
  U2 layout 1  phase .29      U6 layout 5  phase .13
  U3 layout 2  phase .53      U7 layout 6  phase .88
  U4 layout 3  phase .81      U8 layout 7  phase .42
```

This is what lets the demo grid show eight visibly different gravitational worlds while
every one of them stays exactly replayable. It also lowers gradient variance for free —
the batch now spans real environment diversity, not just action sampling.

Consequences:

- The Stage B/C layout set expands from 5 to **12** (8 dispatched per generation, the rest
  held out).
- **Stage A cannot do this** — it has one universe by definition. Stage A is therefore the
  *calibration* rung: it proves physics, the worker contract, the fan-out and the training
  loop. The eight-universe grid demo begins at **Stage B**.

### 7.2 Live mode and replay mode

Every live generation is written to `runs/generations/gen_<n>.json` as it completes.
Replay mode reads those back through the identical renderer, so a flaky API during judging
costs nothing:

```
LIVE    →  Daytona  →  runs/generations/gen_012.json  →  UI
REPLAY  →            runs/generations/gen_012.json    →  UI
```

Replay uses **genuine stored rollouts**. Precomputed real data is fine; fabricated data
never is. The header shows which mode is active.

---

## 7A. The visualisation layer

Full contract: **[web/UI_SPEC.md](web/UI_SPEC.md)**. The essentials that constrain the
backend:

**Daytona calculates, the browser renders.** The worker never draws; the UI never
simulates. Simulation time and display time are separate — a 500-step rollout finishes in a
fraction of a second and the UI replays it over 4–8 s (champion 6–12 s), interpolating
between **recorded** points for 60 fps. The frontend never invents a position.

**A thin adapter, not a contract change.** `controller/adapter.py` renames and reshapes the
worker output (§4.2) into the UI payload — `policy_version → policy_id`,
`termination → status`, flat trajectory rows → per-step `{t,x,y,vx,vy,action,reward}`,
plus `environment.{planets,portal,ship_start}`. The worker contract stays the source of
truth; the frontend adapts to it, never the reverse.

One small worker addition earns its place: trajectory rows carry `vx, vy`
(`[t, x, y, vx, vy, fuel]`) so ship rotation is `atan2(vy, vx)` from real data rather than
a finite-difference approximation. The adapter tolerates the 4-column form meanwhile.

**Honesty rules.** No fabricated trajectories, rewards, successes, sandbox IDs or
generation results. Every universe card shows its **real** `sandbox_id`. A failed worker is
shown as failed, never silently replaced with demo data. Any panel showing something that
is not a measured result — a scripted policy standing in for a trained one, *predicted*
rather than *measured* necessity values — says so on its face.

**Reward stays tuned for learning, never for the screen.** The UI renders whatever comes
back from `gg/spec.py::REWARD`.

---

## 8. Metrics & instrumentation

| Metric | Note |
|---|---|
| Mean / best return, success rate | On held-out layouts (A–C) or held-out seeds (D+). Never training keys. |
| **Necessity delta vs. previous stage** | First-class artefact: which rows changed verdict, and by how much `ΔReturn`. |
| Generalisation gap (train vs. held-out success) | Only meaningful from Stage D — deterministic stages have a trivial split. |
| Fuel at capture, steps to capture | Qualitative behaviour change across generations. |
| Gradient norm, policy entropy, `KL(π_k ‖ π_{k−1})` | Optimisation health; trigger for switching to PPO. |
| Learned-gate values `σ(g)` | Logged every generation — the live necessity trace. |
| Sandboxes launched, wall-clock / rollout, total cost | Daytona budget. |
| Return vs. total Daytona-episodes consumed | Sample efficiency — justifies the parallelism. |
| Per-stage training-curve gallery | Return vs. generation for A, B, C… side by side — shows warm-start transfer. |

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Deterministic stages memorise one trajectory instead of a control law | Small fixed *set* of layouts (not one); hold two out for eval; randomise orbital phase per rollout in C; keep advantage normalisation on. |
| Necessity scores not comparable across stages | Freeze the observation-builder as a strict superset from Stage A; zero-fill absent components rather than shrinking the vector. |
| Curriculum stalls at a rung | Warm-start the trunk from the previous champion; if stuck, insert an intermediate rung (e.g. B with one planet). |
| Stage A too trivial — every variable looks unnecessary | Expected: A establishes the irreducible baseline. Tighten the fuel budget so control precision matters. |
| Predicted matrix wrong (e.g. per-body features necessary at B) | A finding, not a failure. Report measured vs. predicted. |
| Gate criteria slip; stochastic work starts before C is solid | Necessity ranking must be stable across 3 re-runs before advancing — a hard gate. |
| Gradient variance too high | More seeds/generation; advantage norm; value baseline; PPO-lite; potential-based shaping only. |
| Reward hacking (skim the portal at speed, oscillate for progress reward) | Speed gate on capture (A7); potential-based shaping; cap per-step shaping magnitude. |
| Daytona create latency dominates wall-clock | Pre-baked snapshot; fewer larger generations; warm pool / `linked_sandbox`; more steps per sandbox. |
| Diverging physics / NaNs | Softening `ε`; sub-stepping; velocity clamp; drop the rollout from the batch, don't crash the generation. |
| 3D action space blows up | Factored per-axis categorical, or continuous Gaussian — never a flat 27-way softmax if it stalls. |
| Controller / worker spec drift | One shared `spec.py`; worker rejects a mismatched `obs_spec_hash`. |

---

## 10. Build order (concrete)

1. **Daytona smoke test.** 4 sandboxes, each prints its seed, launched with
   `asyncio.gather`. Collect `{sandbox_id, seed, status}`. Gate: four distinct
   `sandbox_id`s, all `finished`, no mocks.
2. **Stage A — one 2D episode in one sandbox.** `universe.py` + `physics.py` (leapfrog,
   softened, one fixed star) + a scripted policy. Worker returns a full trajectory. Gate:
   energy drift < 1% over 500 steps with thrust off; collision and capture fire correctly.
3. **Stage A — fan-out, frozen policy.** 8 sandboxes from the snapshot, one random-init
   network, `asyncio.gather`, SSE lifecycle events + `controller/adapter.py` + generation
   recording to `runs/generations/`. Gate: 8 concurrent sandboxes, 8 distinct real
   `sandbox_id`s, 8 rollout payloads that a renderer could replay. Stage A's eight worlds
   are identical (one universe) — that is expected; the grid gets its variety at Stage B.
4. **Stage A — REINFORCE loop.** Returns, baseline, advantage norm, entropy bonus, Adam
   step, generation loop, learned input gate. Gate: success rate on frozen eval keys rises;
   champion reaches the portal reliably.
5. **Stage A — necessity study.** Run M1–M4; fill matrix column A. Gate: stable ranking
   across 3 re-runs.
6. **Stage B**, then **Stage C** — repeat 3→5 with the new universe branch, warm-starting
   the trunk, dispatching **8 distinct keys per generation** (§7.1). Stage C is the demo
   endpoint.
7. **Build the UI** ([web/UI_SPEC.md](web/UI_SPEC.md)) against the real SSE stream and real
   rollout payloads — champion panel, 8-card universe grid with live lifecycle, ghost
   trails, necessity bar chart, `TRAIN NEXT GENERATION`, replay mode. There is no fake-data
   phase: the backend event stream lands first (step 3), so the UI is real-data-backed from
   its first commit.
8. **Write up Stages D / E / F** — this document + predicted matrix columns. Implement only
   if time remains.

---

## Appendix A — key equations

**A1 — force model (2D & 3D)**

```
a = Σ_i  [ G M_i (r_i − r) / (‖r_i − r‖² + ε²)^{3/2} ]  +  a_thrust
```

**A2 — leapfrog / velocity-Verlet step, with process noise**

```
v_{t+½} = v_t + (dt/2) a(r_t)
r_{t+1} = r_t + dt · v_{t+½}
v_{t+1} = v_{t+½} + (dt/2) a(r_{t+1}) + η_t
```

**A3 — discounted return / advantage**

```
G_t = Σ_{k≥0} γ^k r_{t+k}          A_t = G_t − b(o_t)
```

**A4 — entropy-regularised loss (minimise)**

```
L(θ) = −(1/N) Σ_{j,t} [ A_{j,t} log π_θ(a_{j,t} | o_{j,t}) + β H(π_θ(· | o_{j,t})) ]
```

**A5 — mass / fuel update (Tsiolkovsky, discrete)**

```
m_{t+1} = m_t − |T| dt / (I_sp g₀)   ≈   fuel_{t+1} = fuel_t − c |a_thrust| dt
```

**A6 — governing dimensionless groups**

```
Π₁ = a_thrust^max / g₀                 (control authority)
Π₂ = |v₀| / √(G M_tot / L₀)            (speed vs. escape)
Π₃ = T_budget / T₀                     (time budget vs. dynamical time)
Π₄ = R_portal / L₀                     (portal size vs. system size)
Π₅ = d_body,min / L₀                   (closest approach vs. system size)
```

**A7 — portal-capture condition**

```
‖r − r_p‖ < R_capture   and   ‖v − v_p‖ < v_capture
```

---

## Appendix B — references

- Daytona Python SDK — [AsyncDaytona](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/),
  [AsyncSnapshot](https://www.daytona.io/docs/en/python-sdk/async/async-snapshot/),
  [SDK reference](https://www.daytona.io/docs/en/python-sdk/),
  [daytona on PyPI](https://pypi.org/project/daytona/).
- Companion artifacts —
  [Gravity Gauntlet RL Plan](https://claude.ai/code/artifact/bb02de5d-e297-411d-b100-f21be1ad8809),
  [Newtonian Variable Ladder](https://claude.ai/code/artifact/fd74c422-edfe-4b9c-bda5-355795dc72bc).

*Drafted 2026-08-30. Demo endpoint: Stage C (deterministic 2D, orbiting planets) + the
8-way parallel Daytona fan-out.*
