"""Generation recording (plan.md 4, web/UI_SPEC.md 8).

Each completed generation is written to `runs/generations/gen_<n>.json` as it
finishes. This is what makes replay mode free: the same renderer reads a live SSE
stream or a stored file, and the stored data is genuine - recorded from real
sandboxes, never fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path

from controller.config import REPO_ROOT

GENERATIONS_DIR = REPO_ROOT / "runs" / "generations"


def generation_path(n: int, root: Path | None = None) -> Path:
    return (root or GENERATIONS_DIR) / f"gen_{n}.json"


def write_generation(payload: dict, root: Path | None = None) -> Path:
    path = generation_path(payload["generation"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)                       # atomic: a reader never sees half a file
    return path


def load_generation(n: int, root: Path | None = None) -> dict:
    return json.loads(generation_path(n, root).read_text())


def list_generations(root: Path | None = None) -> list[int]:
    d = root or GENERATIONS_DIR
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("gen_*.json"))
