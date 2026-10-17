"""Probe: DINOv2 zone-anomaly scores on all cached videos for three situations
(desk between two visits, desk after the cashier left, safe zone after the cashier left it).
Prints per-video scores with labels so a threshold can be judged. usage: python scripts/probe_anomaly.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, load_labels, load_video  # noqa: E402
from app.ml.anomaly import DinoPatch, anomaly_score, crop_and_mask  # noqa: E402
from app.ml.candidates import _persons, hand_zone, static_regions, generate  # noqa: E402
from app.ml.events import analyze_scene  # noqa: E402
from app.ml.moments import safe_zone, visit_runs, _box_overlap  # noqa: E402


def feats(dino, frames, obs, idxs, box):
    out = []
    for i in idxs:
        c, m = crop_and_mask(frames[i], box, _persons(obs[i], 0.3), 1.0)
        if c.size and m.mean() < 0.9:
            out.append(dino.features(c, m))
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def main():
    labels = load_labels()
    dino = DinoPatch(threads=6, size=224)
    rows = []
    for name in sorted(labels):
        if "base" in labels[name]:
            continue
        d, _ = load_video(name, with_frames=False)
        obs = d["obs"]
        frames = [cv2.imread(os.path.join(CACHE, "frames", name, f"{o.frame_id:05d}.jpg")) for o in obs]
        sc = analyze_scene(obs, d["meta"]["width"], d["meta"]["height"])
        times = sc.times
        types = [t for _, t in labels[name]["types"]]
        seated = [i for i in range(len(obs)) if sc.present[i]]
        # --- desk between visits
        runs = visit_runs(sc)
        desk = hand_zone(sc, seated)
        for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
            if b0 - a1 < 3.0 or a1 - a0 < 8.0 or desk is None:
                continue
            bank_i = [i for i in range(len(obs)) if a0 - 2.0 <= times[i] <= a0 + 2.0]
            test_i = [i for i in range(len(obs)) if a1 + 1.0 <= times[i] <= b0 - 0.5][-4:]
            s = anomaly_score(feats(dino, frames, obs, test_i, desk), feats(dino, frames, obs, bank_i, desk))
            rows.append(("between", name, labels[name]["label"], "cash_left_on_desk" in types, round(s, 3)))
        # --- desk after leave + safe after leaving it
        half = [cv2.resize(f, None, fx=0.5, fy=0.5) for f in frames]
        cands = generate(sc, half)
        for c in cands:
            if c.kind == "leave" and c.f.get("max_d", 0) > 0.9:
                box = c.extra.get("desk_box")
                if box is None:
                    continue
                t_last = c.t - c.f.get("tidy", 0)
                bank_i = [i for i in seated if times[i] < t_last][-6:]
                late = [i for i in range(len(obs)) if times[i] >= c.t + 1.5][-4:]
                s = anomaly_score(feats(dino, frames, obs, late, box), feats(dino, frames, obs, bank_i, box))
                rows.append(("desk_after_leave", name, labels[name]["label"], "left_cash_unattended" in types, round(s, 3)))
            if c.kind == "safe":
                sz = safe_zone(sc)
                stand = [i for tr in sc.tracks if tr.tid in sc.cashier_ids for k, i in enumerate(tr.idx)
                         if tr.info["d"][k] > 0.25 and tr.h[k] / sc.ref_h > 1.3]
                t0 = times[min(stand)] if stand else 0
                bank_i = [i for i in range(len(obs)) if times[i] < t0 and not _box_overlap(sz, _persons(obs[i], 0.3), 0.5)][-4:] or [0]
                late = [i for i in range(len(obs)) if times[i] > c.t and not _box_overlap(sz, _persons(obs[i], 0.3), 0.5)][-4:]
                s = anomaly_score(feats(dino, frames, obs, late, sz), feats(dino, frames, obs, bank_i, sz))
                rows.append(("safe", name, labels[name]["label"], "safe_left_open" in types, round(s, 3)))
    for kind in ("between", "desk_after_leave", "safe"):
        print(f"== {kind}")
        for r in sorted((r for r in rows if r[0] == kind), key=lambda r: -r[4]):
            print(f"   {r[1]:5s} viol={r[2]} POS={int(r[3])} score={r[4]}")


if __name__ == "__main__":
    main()
