"""Offline OWLv2 scores for next_visit candidates → data/owl/<name>.json ({"t": owl_item})."""
from __future__ import annotations
import json, os, sys
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, load_labels, candidates_for  # noqa
from app.ml.candidates import _persons  # noqa
from app.ml.ovd import Owl  # noqa
out_dir = os.path.join(ROOT, "data", "owl"); os.makedirs(out_dir, exist_ok=True)
owl = None
for name, lab in sorted(load_labels().items()):
    if "base" in lab: continue
    dst = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(dst): continue
    d, sc, cands = candidates_for(name)
    res = {}
    for c in cands:
        if c.kind != "next_visit" or c.extra.get("desk_box") is None: continue
        owl = owl or Owl(threads=8)
        obs = d["obs"]
        def gf(i): return cv2.imread(os.path.join(CACHE, "frames", name, f"{obs[i].frame_id:05d}.jpg"))
        def sc_(idxs): return max((owl.item_score(gf(i), c.extra["desk_box"], _persons(obs[i], 0.3)) for i in idxs), default=0.0)
        g, p = sc_(c.extra["gap_idx"]), sc_(c.extra["pre_idx"])
        res[str(round(c.t, 1))] = max(0.0, g - p)
        print(name, round(c.t, 1), "gap", round(g, 3), "pre", round(p, 3), flush=True)
    json.dump(res, open(dst, "w"))
