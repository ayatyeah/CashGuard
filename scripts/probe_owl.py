"""Probe OWLv2 (Apache-2.0) on zoomed desk crops: "banknotes on the desk" between two client visits
(positives Х-4, Х-38; negatives Х-5, Х-19, Х-39) and on the desk after the cashier left.
Vocabulary ≤ 4 classes per crop, crop upscaled to ~960 px. usage: python scripts/probe_owl.py
"""
from __future__ import annotations
import os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, load_labels, load_video  # noqa
from app.ml.candidates import _persons, hand_zone, generate  # noqa
from app.ml.events import analyze_scene  # noqa
from app.ml.moments import visit_runs  # noqa
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(8)
MID = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(MID); model = Owlv2ForObjectDetection.from_pretrained(MID).eval()
Q = ["a stack of banknotes", "an identity card", "a passport", "a paper document"]


def detect(crop):
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    s = max(im.size)
    with torch.no_grad():
        inp = proc(text=[Q], images=im, return_tensors="pt"); out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, threshold=0.1, target_sizes=torch.tensor([[s, s]]))[0]
    best = {}
    for sc, lab in zip(res["scores"], res["labels"]):
        lab = int(lab) if not isinstance(lab, str) else Q.index(lab)
        best[Q[lab]] = max(best.get(Q[lab], 0.0), float(sc))
    return best


def masked_crop(frame, box, persons):
    x1, y1, x2, y2 = [int(v) for v in box]
    c = frame[max(0, y1):y2, max(0, x1):x2].copy()
    for b in persons:
        bx1, by1, bx2, by2 = [int(v) for v in b]
        cv2.rectangle(c, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (120, 120, 120), -1)
    h, w = c.shape[:2]
    f = 960 / max(h, w)
    return cv2.resize(c, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC) if f > 1 else c


labels = load_labels()
for name in sorted(labels):
    if "base" in labels[name]:
        continue
    d, _ = load_video(name, with_frames=False); obs = d["obs"]
    sc = analyze_scene(obs, d["meta"]["width"], d["meta"]["height"])
    runs = visit_runs(sc)
    seated = [i for i in range(len(obs)) if sc.present[i]]
    desk = hand_zone(sc, seated)
    if desk is None or len(runs) < 2:
        continue
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        if b0 - a1 < 3.0 or a1 - a0 < 8.0:
            continue
        gap = [i for i in range(len(obs)) if a1 + 1.0 <= sc.times[i] <= b0 - 0.5][-3:]
        pre = [i for i in range(len(obs)) if a0 - 2.0 <= sc.times[i] <= a0 + 1.0][:2]
        t0 = time.time()
        def score(idxs):
            out = {}
            for i in idxs:
                fr = cv2.imread(os.path.join(CACHE, "frames", name, f"{obs[i].frame_id:05d}.jpg"))
                r = detect(masked_crop(fr, desk, _persons(obs[i], 0.3)))
                for k, v in r.items(): out[k] = max(out.get(k, 0), v)
            return out
        g, p = score(gap), score(pre)
        print(f"{name:5s} viol={labels[name]['label']} between {a1:.0f}-{b0:.0f}s  gap={ {k: round(v,2) for k,v in g.items()} }  pre={ {k: round(v,2) for k,v in p.items()} }  {time.time()-t0:.0f}s", flush=True)
