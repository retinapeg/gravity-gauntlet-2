# Stage E — the 3D lift · *plan only*

**Not demoed.** Specified here so the dimensional lift stays a config change rather than a
rewrite.

**Goal.** Identify the variables that are necessary **only out of plane** — the ones no
amount of 2D training could have discovered.

**Why this rung.** This is the payoff for writing `gg/physics.py` dimension-generic from
day one. The force model (A1) and the integrator (A2) are already shape-agnostic: they
operate on `(d,)` and `(n, d)` arrays and never assume `d == 2`. What genuinely changes is
smaller than it looks, and it is exactly the interesting part — angular momentum stops
being a scalar and becomes a vector with a *direction*, and a whole orbital-geometry group
appears that has no 2D analogue.

## What actually changes

| Aspect | 2D | 3D |
|---|---|---|
| Force model A1 | 2-vector | 3-vector — **formula unchanged** |
| Integrator A2 | leapfrog on 2-arrays | **identical code** on 3-arrays |
| Angular momentum | scalar `x v_y − y v_x` | vector `r × v` → introduces inclination and node |
| Orbit description | `a, e, ν` | `a, e, i, Ω, ω, ν` |
| Action space | 9 (rest + 8 compass) | factored `3³ = 27`, or 7 flat, or continuous |
| Observation dim | ~28 | ~40 |
| Universe | coplanar | random orbital inclinations |
| Rendering | 2D canvas | three.js, orbiting camera |

The guard rails already in the code: `gg/spec.py::obs_layout` and
`gg/physics.py::action_to_thrust` both raise `NotImplementedError` for `dims != 2` rather
than silently producing a wrong-shaped vector. Those two raises are the checklist — make
them pass, honestly, and the lift is done.

## Newly live observation groups

`inclination` (1) · `raan` (1) · `a_grav_z` — plus `angular_momentum` widening from 1 to 3
components. **This is the one rung where the frozen-superset rule is deliberately broken**:
the 3D vector is a different vector. Necessity scores are therefore comparable *within* 3D
(E vs. F) and *within* 2D (A–D), but the 2D→3D comparison is qualitative — which is the
point of the rung, not a flaw.

## Action space — pick in this order

1. **Factored discrete** — three categorical heads, one per axis, each `{−, 0, +}`.
   9 logits, 27 effective actions, log-prob is the sum across heads. Smallest change from
   the 2D head.
2. **Flat discrete 7** — rest plus `±x, ±y, ±z`. Simplest; no diagonal burns.
3. **Continuous** — a diagonal Gaussian over a clipped thrust 3-vector. Most expressive;
   swaps `Categorical` for `Normal` in the log-prob and entropy terms. Only if discrete
   plateaus.

## Steps

1. Make `obs_layout(3)` real — add the 3D group table; widen `angular_momentum` to 3.
2. Make `action_to_thrust(..., dims=3)` real — factored head first.
3. Inclined-orbit generation in `sample_universe(seed, dims=3)`.
4. Verify the integrator on 3-arrays: energy drift < 1% over 500 steps, coasting, in 3D.
   The same test as Stage A's gate, on a 3-vector state.
5. Rebuild the snapshot as `gravity-gauntlet-worker-v2`.
6. Train — optionally warm-starting the trunk (the head cannot transfer).
7. M1–M4; record column E.

## Predicted necessity delta

**+ inclination `i`**, **+ the direction of `L`**, **+ the out-of-plane component of
`a_grav`**. `Ω` marginal. Everything necessary in 2D stays necessary.

## Gate

Success climbs in 3D on held-out seeds; energy drift gate passes on 3-arrays; the ablation
identifies at least one variable necessary in 3D that was absent or unnecessary in 2D.

## Status

⬜ Documented, not implemented. Post-hackathon.
