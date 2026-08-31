# Stage C — orbiting planets · **demo endpoint**

**Goal.** Find what a **moving** hazard adds — and stop here for the hackathon.

**Why this rung.** The field becomes time-varying. A position that is safe now is not safe
in fifty steps, so a purely reactive controller starts failing in a way it never did at
Stage B. The prediction is that this rung is where an **anticipatory** variable becomes
necessary for the first time. That is the most interesting single claim in the ladder, and
it is reachable in deterministic 2D — which is why the demo ends here rather than pushing
into stochastic or 3D.

## World

The Stage B layouts, with planets now orbiting the star on **deterministic** ellipses. No
inter-planet coupling — each planet is on a fixed analytic orbit, advanced to
`orbital_phase` at `t = 0` and integrated forward with the craft.

To implement: a `"C"` branch in `gg/universe.py` (`advance_orbits(u, phase)`), and
`Universe.body_vel` — already on the dataclass, zero until now — starts carrying real
values.

## Keys

`{"layout_id": n, "orbital_phase": p}`, `n` in 0–11. Layouts 0–9 for training with `p`
sampled per rollout; **layouts 10–11 held out entirely**, plus a set of **unseen phases** on
the training layouts. The phase axis is what stops the policy memorising a single
choreography while the universe is still deterministic.

A generation dispatches **8 distinct `(layout_id, orbital_phase)` pairs** (plan.md §7.1) —
this rung is the demo grid at its strongest, because the eight worlds differ in both
geometry *and* where the planets happen to be when the craft launches.

## Newly live observation groups

`tca_nearest` (1) · `nearest_rel_vel` (2) · `dphi_dt` (1).

## Predicted necessity delta vs B

**+ time-to-closest-approach and/or nearest-body relative velocity.** One of the two
should be enough; M2 ablation decides which, and whether the other is redundant.
`L` and `e` are predicted to turn from unnecessary to *informative* here — the orbit
geometry now carries usable structure. `dΦ/dt` marginal.

## Steps

1. **Extend** `gg/universe.py` with the `"C"` branch: analytic ellipse advance +
   `orbital_phase`.
2. **Moving-body integration** — bodies advance every step alongside the craft; `a_grav`
   is evaluated against their **current** positions.
3. **Flip** `tca_nearest`, `nearest_rel_vel`, `dphi_dt` live in `gg/obs.py`.
4. **TCA correctness test** — for a body on a known straight-line relative path, the
   computed time-to-closest-approach must match the closed-form answer (`tests/`).
5. **Train** warm-started from the Stage B champion, randomising `orbital_phase` per
   rollout.
6. **Attribute** — M1–M4 on the Stage C champion.
7. **Record** `results/necessity.json` and the delta vs column B.
8. **Wire the demo** — champion panel, the 8-card Daytona fan-out, the learning overlay,
   and the necessity bar chart, all fed by real rollouts (build-order step 7).

## Gate

- Success rate high on **unseen phases** and on the held-out layouts.
- Necessity ranking stable across 3 re-runs.
- An anticipatory variable is measurably necessary here and measurably *un*necessary at
  Stage B — that contrast is the demo's headline finding.

## Demo composition

Full UI contract: **[web/UI_SPEC.md](../../web/UI_SPEC.md)**.

`TRAIN NEXT GENERATION` freezes the current policy, draws a fresh base seed, builds 8
distinct keys and launches 8 sandboxes **concurrently**. Cards step
`CREATING → READY → RUNNING → SUCCESS / CRASHED / TIMEOUT / ESCAPED`, completing at
different moments — that stagger is what communicates distributed execution. Each card
carries its real `sandbox_id`, seed and returned reward.

The champion panel replays the best rollout at cinematic speed with the last few champions
as ghost trails beneath it, so improvement is visible without reading a number. The
necessity panel shows columns A → B → C filling with **measured** values.

Every generation is recorded to `runs/generations/` as it completes, which is what makes
replay mode free (plan.md §7.2). Stages D, E and F appear only as documentation.

## Status

🟩 **Implemented, trained and gated. Fan-out gate green; training gate GREEN.**

**Implementation.** `gg/universe.py::_stage_c` puts each Stage B planet on the circular
orbit through its own position (`omega = sqrt(GM_star / a^3)`), so a layout keeps its
character while the field becomes time-dependent. `orbital_phase` advances every planet,
which turns 12 layouts into an effectively continuous key space. The orbits are
**kinematic on purpose** - planets orbit the star but do not pull on each other - so a
universe stays a pure function of its key and replays exactly (plan.md §3.4).

The episode loop refreshes body positions *and velocities* every step, and the
velocity-Verlet half-steps evaluate the field at their own instants; otherwise the
integrator quietly solves a different problem than the one posed. `dPhi/dt` is a new
hand-derived formula (`physics.dpotential_dt`) and is checked against a central difference
of the actual potential in `tests/test_stages_bc.py`, same treatment as the policy gradient.

**Fan-out - the demo endpoint.** ✅

```
generation 3   8 concurrent sandboxes, 8 distinct real ids
  worker 0  {layout_id: 7, orbital_phase: 0.282}  ->  timeout
  worker 1  {layout_id: 6, orbital_phase: 0.754}  ->  timeout
  ...  8 distinct (layout, phase) pairs
  [PASS] 8 distinct keys dispatched
```

## The braking fix - how the training gate went from 0.00 to 0.90

Stage C's first run ended at 0.00 held-out success, and that flat zero hid the real
finding. Instrumenting closest approach *and arrival speed* showed the policy had already
solved navigation: **41 of 48 held-out episodes reached inside `r_capture`** (some as
close as 0.007 against a 0.35 bubble) and every one was refused by the A7 speed gate,
arriving at 0.76-0.96 against `v_capture = 0.6`.

That the gate was fair rather than impossible was settled by running the scripted
`PursuitPolicy` on the same held-out keys: **24/24 captures with 68% fuel left**, braking
to v ≈ 0.596 right at the bubble edge. The rung was solvable; the learned policy was at
fault.

**Cause.** A7 capture has two conditions - `dist < r_capture` AND `|v - v_p| < v_capture` -
and the reward only ever shaped the first. `k_progress` pays for closing distance and is
completely indifferent to arrival speed, so "dive at the portal" is *optimal* under the
reward as written. The policy was not failing to learn; it had learned exactly what it was
asked to.

**Fix.** `spec.REWARD["k_brake"]` adds the matching potential for the speed condition:

    Phi(r, v) = exp(-(d / (3 * r_capture))^2) * exp(-(|v - v_p| / v_capture)^2)

a capture-readiness score in [0, 1] whose global maximum *is* the A7 condition. It enters
the reward as `k_brake * (Phi_t+1 - Phi_t)`, a difference of a potential exactly like
`prog`, so it telescopes over any closed loop and cannot be farmed by oscillating
(`tests/test_brake_shaping.py`, 9 tests).

### Three things that were measured, not guessed

1. **The potential must never go negative.** The first version was a penalty on arrival
   speed - zero far away, negative near the portal - which makes *not approaching* the
   best available move. Measured: closest approach drifted 0.30 -> 1.78 over 60
   generations and the craft stopped going to the portal at all. Both factors are now
   positive, so approach stays weakly uphill even while fast.
2. **`k_brake` has to be large.** At 1 and 10 the policy still dives at ~1.1 and never
   captures; the term is too small next to a +/-100 terminal bonus. Swept 1/10/40/100/250:
   40 and 100 both peak at 0.844 locally, 100 has the steadier tail (mean 0.70 vs 0.61),
   250 has a lower ceiling. `brake_clip` must rise with it - at `k_brake=40` a 0.10 clip
   binds on 6.9% of steps, which stops the shaping telescoping.
3. **A warm start must not reuse the previous rung's lr.** At 4e-3 a warm start from the
   Stage C champion DIVERGES - closest approach drifts back out to 1.70 and the craft
   stops approaching. 1e-3 keeps the inherited navigation while the brake term does its
   work. Same lesson Stage B learned about `beta_start`, one rung later.

## Result

150 generations on the fleet, warm-started from the pre-brake champion, lr 1e-3:

```
mean reward   first 30 gens:  -6.83   last 30 gens: +64.40
success rate  first 30 gens:   0.18   last 30 gens:   0.53
held-out probe  0.00 -> 0.50 -> 0.75 -> 0.88 -> 1.00   (champion taken at gen 100)
```

Champion on 48 sampled episodes per held-out axis (`results/heldout_eval.json`):

| held-out axis | success | mean reward | terminations |
|---|---|---|---|
| unseen **phases** | **0.833** (40/48) | +126.6 | 40 portal, 8 timeout |
| unseen **layouts** 10 & 11 | **0.958** (46/48) | +147.3 | 46 portal, 2 timeout |

Up from **0.00** on both. The rung generalises *better* across unseen layouts than across
unseen phases - the answer this stage's two-axis split was built to get: the policy
learned the physics, not one choreography.

Greedy stays at 0.00, which is expected rather than a defect - S4 is on from Stage A, so
the policy IS the sampling distribution and argmax is a policy that was never trained.

The pre-brake champion and metrics are kept in `results/pre_brake/` for comparison.

*Caveat on the trainer's own final line:* it evaluates only 8 episodes and reported 0.75,
one episode short of the bar. At n=8 the 95% interval runs roughly 0.35-0.97, which cannot
settle a 0.80 gate in either direction - hence the 48-episode evaluation above.

## Known cost of this change, at Stage B

`spec.REWARD` is global, so the brake term applies to every rung. Isolated at Stage B (its
own lr 4e-3, 50 generations, k_brake the only difference):

| | collisions of 16 | best held-out | champion |
|---|---|---|---|
| k_brake=0 | 0-1 | 0.00 | 0.000 |
| k_brake=100 | 4-14 | 0.38 | 0.375 |

The term *helps* Stage B's success - 0.38 in 50 generations, matching B's original
150-generation result - but clearly raises collisions. The pull toward "slow and near the
portal" is outcompeting `k_proximity = 0.3`, and Stage B is the rung with planets
straddling the direct path. Raising `k_proximity` alongside `k_brake` is the obvious next
experiment and has NOT been run.

## Necessity column C - measured, and it refutes the prediction

M1-M3 on the 0.83/0.96 champion, 21 active groups, 3 re-runs, 16 episodes per arm
(`results/necessity.json`). Baseline: return 134.1, success 0.875.

**The predicted headline finding does not hold.** plan.md 5 predicted that an anticipatory
variable becomes necessary at C, because the hazard now moves. Measured:

| group (first live at B/C) | verdict | ΔSuccess |
|---|---|---|
| `tca_nearest` | unnecessary | 0.000 |
| `nearest_rel_vel` | unnecessary | 0.000 |
| `dphi_dt` | unnecessary | 0.000 |
| `nearest_body` | unnecessary | -0.042 |
| `perbody_topk` | unnecessary | -0.062 |
| `body_count` | unnecessary | 0.000 |

Every group this rung switched on measures as unnecessary. The B-vs-C contrast that was
meant to be the demo's headline is not there.

### Why - and the methodological catch that makes it readable

The zeroed arm initially reported two large "necessary" results, `fuel_fraction`
(ΔReturn +143.6, ΔSuccess +0.771) and `craft_vel` (+51.3, +0.375). The **resampled** arm
contradicts both: **-4.0** and **-4.4**.

Zeroing a group hands the policy a value it may never see - zeroing `fuel_fraction` says
"you have no fuel" on step one - so a big zeroed delta can be pure out-of-distribution
shock. The resampled arm replaces the group with a draw from the champion's own trajectory
pool, which stays in distribution, and is the honest control. `classify()` uses the ZEROED
arm alone, so both verdicts were shock rather than necessity.

`corroborated_by_resampling` now cross-checks every verdict. It discriminates rather than
flagging everything - re-scored against column A, 3 of its 4 `necessary` verdicts survive
(`portal_rel_pos` 39.0/16.7, `closing_speed_portal` 19.5/8.4, `fuel_fraction` 16.8/5.6)
and only `time_fraction` (22.4/2.9) does not.

**So column C has ZERO corroborated necessary variables, against three at column A.**

### What that actually says about the champion

`portal_rel_pos` measures *unnecessary* at C (ΔSuccess -0.021) having been the strongest
variable at A. That is not noise - it is the finding. In Stages A-C the portal sits at the
same fixed position in every layout, so a policy can fly a memorised trajectory without
ever consulting where the portal is. Resampling any single feature barely moves it because
it is not really reading its observations.

This corroborates Stage D from the other direction: the same champion collapses from 0.62
to 0.06 when the world is rotated (`stages/stage_d_stochastic/README.md`). Two independent
measurements, one conclusion - **this policy learned a fixed trajectory in a fixed frame,
not a control law.**

### Necessity gate: RED

- Full-rank Spearman over 3 re-runs by ΔReturn: **min 0.287** (bar 0.7) - fails, and worse
  than column A's 0.596.
- By closest approach: min 0.773, mean 0.809 - passes, and is the better-conditioned
  statistic, as column A already argued.
- Verdict class: 13/21 groups keep their verdict on every re-run.
- The predicted anticipatory-variable contrast is refuted.

## Still open

- **M4 (nested sets) was not run** - it trains five nested policies and is the expensive
  arm; M1-M3 are what is reported above.
- Column C should be re-measured against a champion that actually uses its observations.
  The fix is the same one Stage D needs: a portal-relative observation frame, which changes
  `obs_spec_hash` and is a ladder-wide decision.
