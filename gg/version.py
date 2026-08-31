"""Source fingerprint of the `gg` package.

The snapshot bakes a COPY of `gg/` into the image. `obs_spec_hash` catches drift in
`spec.py`, but nothing caught a stale `physics.py` or `policy.py` - so a forgotten
snapshot rebuild would quietly train against old code and the numbers would be
wrong in a way no gate would notice. This hash closes that hole: the worker reports
it, and the controller refuses to trust a rollout whose worker does not match.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_hash() -> str:
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:8]
