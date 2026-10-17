"""Dense-pose probe for hand-to-hand transfers: re-run YOLOX+RTMPose at 8 fps inside visitor windows and report
wrist-contact episodes (cashier wrist ↔ visitor wrist < 0.3 ref_h) with the direction of approach.
usage: python scripts/probe_handover.py NAME [NAME...]
"""
from __future__ import annotations
import os, sys, pickle
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, load_labels  # noqa
from app.ml.perception import Perception, iter_frames  # noqa
from app.ml.events import analyze_scene  # noqa

per = Perception(threads=8, crop_model=None)
for name in sys.argv[1:]:
    d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
    sc = analyze_scene(d["obs"], d["meta"]["width"], d["meta"]["height"])
    vis = [t for t in sc.tracks if t.role == "visitor"]
    if not vis or sc.seat is None:
        print(name, "no visitor"); continue
    win = [(max(0, t.t[0] - 1), t.t[-1] + 1) for t in vis]
    split = "train" if os.path.exists(f"data/train/videos/{name}.mov") else "test"
    path = f"data/{split}/videos/{name}.mov"
    seat, ref = sc.seat, sc.ref_h
    series = []  # (t, dmin, cashier_wrist_speed_toward, visitor_wrist_speed_toward)
    prev = {}
    state = {}
    for fid, t, img in iter_frames(path, 8.0):
        if not any(a <= t <= b for a, b in win):
            continue
        o = per.frame(fid, t, img, state)
        pb = o.boxes[o.classes == 0]; ps = o.scores[o.classes == 0]
        if len(pb) < 2:
            continue
        ctr = (pb[:, :2] + pb[:, 2:]) / 2
        dseat = np.linalg.norm(ctr - seat, axis=1) / ref
        ci = int(np.argmin(dseat))
        others = [j for j in range(len(pb)) if j != ci and ps[j] > 0.5 and dseat[j] < 2.0]
        if not others:
            continue
        vi = min(others, key=lambda j: dseat[j])
        kc, kv = o.kpts[ci], o.kpts[vi]; sc_c, sc_v = o.kpt_scores[ci], o.kpt_scores[vi]
        wc = kc[[9, 10]][sc_c[[9, 10]] > 0.3]; wv = kv[[9, 10]][sc_v[[9, 10]] > 0.3]
        if len(wc) == 0 or len(wv) == 0:
            continue
        dm = min(np.linalg.norm(a - b) for a in wc for b in wv) / ref
        # approach: wrist displacement toward the partner's torso since the previous dense frame
        tc = kc[[5, 6]].mean(0); tv = kv[[5, 6]].mean(0)
        def toward(w_now, w_prev, target):
            if w_prev is None: return 0.0
            v = w_now - w_prev; u = target - w_prev
            return float(np.dot(v, u) / (np.linalg.norm(u) + 1e-6)) / ref
        wcn = wc[np.argmin([min(np.linalg.norm(a - b) for b in wv) for a in wc])]
        wvn = wv[np.argmin([min(np.linalg.norm(a - b) for b in wc) for a in wv])]
        series.append((t, dm, toward(wcn, prev.get("c"), tv), toward(wvn, prev.get("v"), tc)))
        prev = {"c": wcn, "v": wvn}
    # episodes
    ep = []; cur = None
    for t, dm, ac, av in series:
        if dm < 0.3:
            if cur is None: cur = {"t0": t, "t1": t, "ac": 0.0, "av": 0.0}
            cur["t1"] = t
        else:
            if cur is not None:
                # who approached: sum of "toward" in the 1 s before contact
                pre = [(a, b) for (tt, _, a, b) in series if cur["t0"] - 1.0 <= tt < cur["t0"]]
                cur["ac"] = sum(a for a, _ in pre); cur["av"] = sum(b for _, b in pre)
                ep.append(cur); cur = None
    if cur: ep.append(cur)
    lab = load_labels().get(name, {})
    print(f"{name:5s} viol={lab.get('label')} types={[t for _, t in lab.get('types', [])]}")
    for e in ep:
        if e["t1"] - e["t0"] >= 0.2:
            who = "V→C" if e["av"] > e["ac"] + 0.05 else ("C→V" if e["ac"] > e["av"] + 0.05 else "?")
            print(f"    contact {e['t0']:5.1f}–{e['t1']:5.1f}s  approach cashier={e['ac']:+.2f} visitor={e['av']:+.2f}  → {who}")
