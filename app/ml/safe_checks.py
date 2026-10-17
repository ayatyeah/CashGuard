"""Propose a persistent-key check after the cashier leaves; no video-specific coordinates."""
from __future__ import annotations

from .candidates import Cand, _persons
from .moments import safe_zone, _box_overlap


def propose_key(scene, candidates):
    leaves = [c for c in candidates if c.kind == "leave"]
    if not leaves:
        return []
    box = safe_zone(scene, pad=0.3)
    if box is None:
        return []
    leave = leaves[-1]
    free = [i for i in range(len(scene.obs)) if scene.times[i] > leave.t + 0.5
            and not _box_overlap(box, _persons(scene.obs[i], 0.3), thr=0.5)]
    if len(free) < 2:
        return []
    last = free[-1]
    prior = [i for i in free if scene.times[i] <= scene.times[last] - 1.0]
    if not prior:
        return []
    idx = [prior[-1], last]
    return [Cand("safe_key", leave.t, leave.i, [leave.i, *idx], {},
                 type_id="safe_key_left", extra={"key_box": box, "key_idx": idx})]
