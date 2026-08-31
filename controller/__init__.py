"""Host-side orchestration. The controller never simulates - it only launches
sandboxes, aggregates what they return, and takes the gradient step (plan.md 4).

Module contracts, in build order:

    config.py           credentials, the snapshot name, and the stage-folder loaders,
                        read in exactly one place.

    build_snapshot.py   Image.debian_slim("3.12").pip_install([numpy, cpu-torch, pydantic])
                        .add_local_dir("gg", "/opt/gg")  ->  snapshot
                        "gravity-gauntlet-worker-v1".  Rebuilt as -v2 at Stage E.
                        `--verify` boots a sandbox and makes the image report what is
                        actually baked, including gg's source hash.

    fanout.py           run_rollout() creates one sandbox from the snapshot, execs
                        `python -m gg.rollout_worker` with GG_INPUT, parses the single
                        JSON line and deletes in a `finally`.  SandboxPool keeps N warm
                        for the necessity sweeps (plan.md 4.6).
                        run_generation(...) = asyncio.gather over M rollouts with
                        return_exceptions=True.  Dispatches M DISTINCT keys, not M copies
                        of one (plan.md 7.1) - that is what makes the demo grid show eight
                        different universes.  A failed sandbox is DROPPED from the batch,
                        never allowed to put a NaN into the gradient; require >= M/2
                        successes before applying an update.

    adapter.py          Worker output (plan.md 4.2) -> UI rollout payload
                        (web/UI_SPEC.md 5).  Pure renaming and reshaping: policy_version
                        -> policy_id, termination -> status, flat trajectory rows ->
                        per-step {t,x,y,vx,vy,action,reward}, plus environment.{planets,
                        portal,ship_start}.  The worker contract is the source of truth;
                        the frontend adapts to it, never the reverse.

    record.py           Writes each completed generation to runs/generations/gen_<n>.json
                        as it finishes.  This is what makes replay mode free - the same
                        renderer reads live SSE or a stored file, and the stored data is
                        genuine, never fabricated.

    train.py            REINFORCE: discounted returns -> advantage (batch-mean baseline,
                        then a fitted value head if variance demands it) -> normalise ->
                        -(adv * logp).mean() - beta * entropy -> Adam, grad-clip 1.0.
                        ONE step per batch: the estimator is on-policy.

    necessity.py        M1 learned gate / M2 ablation + permutation / M3 saliency /
                        M4 nested feature sets, on the stage champion.  Writes
                        stages/<stage>/results/necessity.json and the delta vs the
                        previous rung.

    api.py              FastAPI + SSE.  Emits {world_id, key, phase, sandbox_id, ...}
                        on every sandbox lifecycle transition so the UI cards show real
                        state rather than theatre.
"""
