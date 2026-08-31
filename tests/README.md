# Tests — local physics validation

**No network, no Daytona, no sandboxes.** These run in under a second and they are the
cheapest possible place to catch a bug that would otherwise cost eight sandboxes per
generation to discover.

Run with `make test`.

## The Stage A gate lives here

`test_physics_gate.py` must assert:

1. **Energy drift < 1% over 500 steps, coasting.** With `NoThrust`, the specific energy
   `E = ½|v|² + Φ` is conserved by the exact dynamics, so drift measures integrator error
   alone. `gg/physics.py::potential` is deliberately the softened potential whose gradient
   is exactly `-accel_gravity`, so this is a true invariant of the *simulated* system —
   not an approximation that would make the test meaningless.
2. **All four terminations fire**: `portal`, `collision`, `timeout`, `escape`. Build a
   state that forces each rather than hoping a rollout wanders into it.
3. **The A7 capture gate rejects a fast pass.** Arriving inside `r_capture` while faster
   than `v_capture` must *not* count as success. This one guards against the most likely
   reward hack.

## Also worth covering

- `spec.obs_layout` — total width equals the sum of the group sizes, and
  `obs_spec_hash()` changes when a group is added or reordered. That hash is the only
  thing standing between a silently misaligned observation vector and a week of confusing
  results.
- `spec.stage_active` — an observation group is live at its own `FIRST_ACTIVE_STAGE` and
  at every rung after it, and zero-filled before.
- `physics.action_to_thrust` — action 0 is exactly zero thrust; the eight compass
  directions are unit vectors; `dims=3` raises rather than returning a wrong shape.
- Per-stage additions: planet-contact collision (Stage B), closed-form time-to-closest-
  approach (Stage C), 3-array energy drift (Stage E).

## What does not belong here

Anything that needs a sandbox. Daytona execution is validated by
`stages/stage_00_smoke/` and by step A6, not by the test suite.
