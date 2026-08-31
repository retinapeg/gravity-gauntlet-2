"""Shared controller-side settings: credentials, paths, the snapshot name.

Every controller module that talks to Daytona goes through `daytona_config()`, so
the API key is read in exactly one place and `.env` is loaded exactly once
(plan.md 4). The snapshot name lives here too because the builder, the rollout
and the fan-out all have to agree on it.
"""
from __future__ import annotations

import os
from pathlib import Path

from daytona import DaytonaConfig
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
GG_DIR = REPO_ROOT / "gg"

# Rebuilt as `...-v2` at Stage E, when the action space becomes 3D (plan.md 4.1).
SNAPSHOT = "gravity-gauntlet-worker-v1"


def daytona_config() -> DaytonaConfig:
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        raise SystemExit("DAYTONA_API_KEY is not set - check your .env file")
    url = os.environ.get("DAYTONA_API_URL")
    return DaytonaConfig(api_key=key, api_url=url) if url else DaytonaConfig(api_key=key)


# stage letter -> stages/<dir>. The stage folder holds parameters and keys, never code.
STAGE_DIRS = {
    "A": "stage_a_fixed_star", "B": "stage_b_fixed_planets", "C": "stage_c_orbiting",
    "D": "stage_d_stochastic", "E": "stage_e_3d", "F": "stage_f_pomdp",
}


def stage_dir(stage: str) -> Path:
    if stage not in STAGE_DIRS:
        raise SystemExit(f"unknown stage {stage!r}")
    return REPO_ROOT / "stages" / STAGE_DIRS[stage]


def load_stage_config(stage: str) -> dict:
    import json
    return json.loads((stage_dir(stage) / "config.json").read_text())


def load_stage_keys(stage: str) -> dict:
    """Stage keys, with the held-out sets normalised.

    Stage C splits held-out into two axes on purpose - unseen PHASE on a known layout
    and unseen LAYOUT entirely - because a policy that generalises across phase but not
    across layout has learned the choreography rather than the physics. Callers that
    only want "the held-out set" get the union under `eval_held_out`; callers that want
    the distinction keep the named sets.
    """
    import json
    keys = json.loads((stage_dir(stage) / "keys.json").read_text())
    named = {k: v for k, v in keys.items()
             if k.startswith("eval_held_out") and k != "eval_held_out" and isinstance(v, list)}
    keys["eval_sets"] = named or ({"eval_held_out": keys["eval_held_out"]}
                                  if "eval_held_out" in keys else {})
    if "eval_held_out" not in keys:
        keys["eval_held_out"] = [k for v in named.values() for k in v]
    return keys


def results_dir(stage: str) -> Path:
    d = stage_dir(stage) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
