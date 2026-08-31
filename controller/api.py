"""FastAPI + SSE - real sandbox lifecycle events for the demo UI (plan.md 4.4).

Every event on this stream corresponds to a real Daytona state transition, so the
world cards show genuine state rather than theatre (web/UI_SPEC.md 9). There is no
fake-data mode: if the fleet is not running, the stream is simply quiet.

    make api        # uvicorn controller.api:app --reload

Endpoints
    GET  /api/health              is the controller up, is the snapshot active
    GET  /api/state               current generation, policy id, champion
    GET  /api/events              SSE: {worker_id, phase, sandbox_id, ...}
    POST /api/train/next          run ONE genuine generation, then one gradient step
    GET  /api/generations         recorded generation ids (replay mode)
    GET  /api/generations/{n}     one recorded generation payload
    GET  /api/necessity           the stage's necessity column, once A9 has run
"""
from __future__ import annotations

import asyncio
import json
from contextlib import suppress

import numpy as np

from daytona import AsyncDaytona
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from controller.build_snapshot import ensure_snapshot, snapshot_state
from controller.config import (REPO_ROOT, SNAPSHOT, daytona_config, load_stage_config,
                               load_stage_keys, results_dir)
from controller.fanout import run_generation, sample_keys
from controller.record import list_generations, load_generation
from controller.train import Trainer, load_champion
from gg import net, spec

STAGE = "A"


class EventBus:
    """Fan one lifecycle event out to every connected browser."""

    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, **event) -> None:
        for q in list(self._subs):
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)


class Session:
    """One training session, held across requests so TRAIN NEXT GENERATION
    genuinely continues from the policy the last press produced."""

    def __init__(self, stage: str = STAGE):
        self.lock = asyncio.Lock()
        self.generation = max(list_generations() or [0])
        self.switch(stage)

    def switch(self, stage: str) -> None:
        self.stage = stage
        self.cfg = load_stage_config(stage)
        self.keys = load_stage_keys(stage)
        params = None
        champ = results_dir(stage) / "champion.npz"
        if champ.exists():
            params = load_champion(champ)          # start from this rung's champion
        self.trainer = Trainer(stage, self.cfg, seed=0, params=params)
        self.warm_started = params is not None

    @property
    def m(self) -> int:
        return int(self.cfg["training"]["sandboxes_per_generation"])


app = FastAPI(title="Gravity Gauntlet controller")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
bus = EventBus()
session = Session()


@app.get("/api/health")
async def health():
    async with AsyncDaytona(daytona_config()) as d:
        state = await snapshot_state(d)
    return {"ok": True, "snapshot": SNAPSHOT, "snapshot_state": state,
            "stage": session.stage, "obs_spec_hash": spec.obs_spec_hash(2),
            "obs_dim": spec.OBS_DIM_2D}


@app.get("/api/state")
async def state():
    ch = session.trainer.champion
    return {"stage": session.stage, "generation": session.generation,
            "policy_id": f"P{session.trainer.version:03d}",
            "policy_version": session.trainer.version,
            "warm_started_from_champion": session.warm_started,
            "sandboxes_per_generation": session.m,
            "champion": {"version": ch["version"], "generation": ch["generation"],
                         "score": None if ch["score"] == float("-inf") else round(float(ch["score"]), 4)},
            "recorded_generations": list_generations()}


@app.get("/api/events")
async def events():
    q = bus.subscribe()

    async def stream():
        try:
            yield {"event": "hello", "data": json.dumps({"stage": session.stage})}
            while True:
                ev = await q.get()
                yield {"event": ev.get("phase", "message"), "data": json.dumps(ev)}
        finally:
            bus.unsubscribe(q)

    return EventSourceResponse(stream())


@app.post("/api/train/next")
async def train_next(stage: str = STAGE):
    """One press of TRAIN NEXT GENERATION (web/UI_SPEC.md 7).

    Freeze the policy -> new generation id -> new base seed -> M sandboxes
    concurrently -> collect -> ONE gradient step -> P(n+1). All M workers evaluate
    the SAME frozen policy: they are collecting experience for one agent.
    """
    if session.lock.locked():
        raise HTTPException(409, "a generation is already running")
    if stage not in ("A", "B", "C"):
        raise HTTPException(400, f"unknown stage {stage!r}")

    async with session.lock:
        if stage != session.stage:
            # Switching rungs mid-session rebuilds the trainer from that rung's own
            # config and champion; the UI can hop between A/B/C without a restart.
            session.switch(stage)
        gen = session.generation + 1
        cfg, m = session.cfg, session.m
        keys = sample_keys(session.keys["train"], m,
                           np.random.default_rng(10_000 * gen + 7))
        weights = net.pack(session.trainer.params)
        base_seed = 10_000 * gen + 7

        bus.publish(phase="generation_start", generation=gen,
                    policy_id=f"P{session.trainer.version:03d}", n_workers=m)

        async with AsyncDaytona(daytona_config()) as d:
            await ensure_snapshot(d, quiet=True)
            payload, raw, failures = await run_generation(
                d, stage=session.stage, cfg=cfg, weights_b64=weights,
                policy_version=session.trainer.version, keys=keys, base_seed=base_seed,
                on_event=lambda **kw: bus.publish(generation=gen, **kw),
                generation=gen,
            )

        stats = None
        need = int(cfg["training"]["min_successful_rollouts_frac"] * m)
        if len(raw) >= max(1, need):
            total = int(cfg["training"]["generations"])
            stats = session.trainer.update(raw, session.trainer.beta(gen, total))
            session.trainer.consider_champion(payload["mean_reward"] or -1e18, gen)
        else:
            bus.publish(phase="generation_skipped", generation=gen,
                        reason=f"only {len(raw)}/{m} rollouts returned")

        session.generation = gen
        payload["update"] = stats
        payload["next_policy_id"] = f"P{session.trainer.version:03d}"
        bus.publish(phase="generation_complete", generation=gen,
                    mean_reward=payload["mean_reward"], best_reward=payload["best_reward"],
                    success_rate=payload["success_rate"],
                    champion_worker=payload["champion_worker"],
                    next_policy_id=payload["next_policy_id"])
        return payload


@app.get("/api/generations")
async def generations():
    return {"generations": list_generations()}


@app.get("/api/generations/{n}")
async def generation(n: int):
    try:
        return load_generation(n)
    except FileNotFoundError:
        raise HTTPException(404, f"generation {n} has not been recorded")


@app.get("/api/necessity")
async def necessity():
    path = results_dir(session.stage) / "necessity.json"
    if not path.exists():
        raise HTTPException(404, "necessity study has not run yet (step A9)")
    return json.loads(path.read_text())


# Serve the live UI from `web/`. Mounted LAST so it cannot shadow the /api routes:
# Starlette matches in declaration order, and a mount at "/" swallows everything.
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")
