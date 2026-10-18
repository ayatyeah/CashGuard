"""Fast evaluation on cached perception + cached VLM answers, through the same pipeline.run as MLAnalyzer.

Perception is deterministic, so cached observations equal a fresh run. Two VLM regimes:
  --vlm-budget none   all proposed moments answered (≈ Apple M3 / no time limit)
  --vlm-budget laptop moments beyond the time budget of this laptop are dropped, in the analyzer's order
usage: python scripts/evaluate_cached.py --split test [--vlm-budget laptop]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_labels, load_video  # noqa: E402
from app.ml import pipeline  # noqa: E402

SEC_PER_MOMENT = 30.0      # Qwen3-VL-2B bf16 on Ryzen 5 5600H
PERCEPTION_RT = 1.0        # perception ≈ 1× video duration on the same CPU


def _owl_cache(name):
    """Offline OWLv2 desk-item scores (scripts/run_owl.py) so next_visit_owl is evaluated like in production."""
    p = os.path.join(ROOT, "data", "owl", f"{name}.json")
    return {float(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--vlm-budget", default="none", choices=["none", "laptop", "off"])
    ap.add_argument("--budget", type=float, default=270.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every k-th cached frame: 2 ≈ 1 fps, 3 ≈ 0.67 fps (low-frame-rate robustness)")
    a = ap.parse_args()
    labels = load_labels()
    models = pipeline.Models()

    class Clip:  # probes from cached CLIP embeddings (same text bank as runtime)
        def __init__(self):
            from app.ml.semantic import ClipEmbedder
            self.e = ClipEmbedder(threads=4)

        def probe_bank(self):
            return self.e.probe_bank()

    clip = Clip()
    rows = []
    for name in sorted(n for n, l in labels.items() if l["split"] == a.split):
        d, frames = load_video(name)
        if a.stride > 1:
            d["obs"], frames = d["obs"][::a.stride], frames[::a.stride]
            if d.get("clip_wa") is not None:
                d["clip_wa"] = d["clip_wa"][::a.stride]
        moms = None
        vpath = os.path.join(ROOT, "data", "vlm", f"{name}.json")
        if a.vlm_budget != "off" and os.path.exists(vpath):
            moms = json.load(open(vpath))
            if a.vlm_budget == "laptop":
                left = a.budget - PERCEPTION_RT * d["meta"]["duration"] - 20.0
                n_ok = max(0, int(left // SEC_PER_MOMENT))
                for k, m in enumerate(moms):
                    if k >= n_ok:
                        m["answer"], m["skipped"] = {}, True
        events, info = pipeline.run(d["obs"], frames, d["meta"]["width"], d["meta"]["height"], models,
                                    clip=clip, clip_wa=d.get("clip_wa"), vlm_moments=moms or [], owl_cache=_owl_cache(name))
        lab = labels[name]
        ev = [(e.type_id, round(e.t, 1), round(e.p, 2)) for e in events]
        pred = int(bool(ev))
        err = min((abs(e[1] - t) for e in ev for t in lab["ts"]), default=None) if (pred and lab["ts"]) else None
        rows.append({"video": name, "label": lab["label"], "pred": pred, "events": ev, "t_err": err,
                     "vlm_skipped": info.get("vlm_skipped", 0), "vlm_moments": info.get("vlm_moments", 0)})
        print(f"{name:5s} y={lab['label']} pred={pred} {'OK ' if pred == lab['label'] else 'ERR'} {ev} "
              f"ts={lab['ts']} err={err} vlm={info.get('vlm_moments', 0) - info.get('vlm_skipped', 0)}/"
              f"{info.get('vlm_moments', 0)}", flush=True)
    y = np.array([r["label"] for r in rows]); p = np.array([r["pred"] for r in rows])
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    errs = [r["t_err"] for r in rows if r["t_err"] is not None]
    s = {"split": a.split, "vlm": a.vlm_budget, "acc": (tp + tn) / len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
         "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
         "t_err_median": float(np.median(errs)) if errs else None,
         "t_err_le5": float(np.mean(np.array(errs) <= 5)) if errs else None}
    print(json.dumps(s, indent=1))
    json.dump({**s, "rows": rows}, open(os.path.join(ROOT, "data", f"eval_cached_{a.split}_{a.vlm_budget}.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
