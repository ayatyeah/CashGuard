"""End-to-end evaluation of MLAnalyzer on a split (default: the held-out TEST split).

Runs the full analyzer on every video (perception from scratch, not the training cache), compares with the
organisers' labels and writes data/eval_<split>.json.  usage: python scripts/evaluate.py [--split test]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_labels  # noqa: E402
from app.analyzers.ml import MLAnalyzer  # noqa: E402
from app.ml.perception import video_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--budget", type=float, default=270.0)
    ap.add_argument("--videos", nargs="+", help="Evaluate these names across both dataset directories")
    ap.add_argument("--output", help="JSON report destination")
    a = ap.parse_args()
    labels = load_labels()
    an = MLAnalyzer()
    vdir = os.path.join(ROOT, "data", a.split, "videos")
    paths = [os.path.join(vdir, f) for f in sorted(os.listdir(vdir))]
    if a.videos:
        paths = []
        for name in a.videos:
            matches = [os.path.join(ROOT, "data", split, "videos", name + ".mov")
                       for split in ("train", "test")]
            matches = [p for p in matches if os.path.isfile(p)]
            if len(matches) != 1 or name not in labels:
                raise ValueError(f"Unknown or ambiguous video: {name}")
            paths.append(matches[0])
    rows = []
    for path in paths:
        f = os.path.basename(path)
        name = os.path.splitext(f)[0]
        if name not in labels:
            continue
        meta = video_meta(path)
        t0 = time.time()
        res = an.analyze(path, meta, a.budget)
        dt = time.time() - t0
        ev = [(d.type_id, round(d.frame_id / meta["fps"], 1), d.confidence) for d in res.detections]
        lab = labels[name]
        pred = int(bool(ev))
        err = min((abs(e[1] - t) for e in ev for t in lab["ts"]), default=None) if (pred and lab["ts"]) else None
        rows.append({"video": name, "label": lab["label"], "pred": pred, "events": ev, "ts": lab["ts"],
                     "t_err": err, "sec": round(dt, 1), "duration": round(meta["duration"], 1),
                     "warnings": res.warnings, "timings": res.timings})
        print(f"{name:5s} y={lab['label']} pred={pred} {'OK ' if pred == lab['label'] else 'ERR'} "
              f"{ev} ts={lab['ts']} err={err} time={dt:.0f}s/{meta['duration']:.0f}s", flush=True)
    y = np.array([r["label"] for r in rows]); p = np.array([r["pred"] for r in rows])
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    errs = [r["t_err"] for r in rows if r["t_err"] is not None]
    summary = {"split": "selected" if a.videos else a.split,
               "evaluation_scope": "Development videos; not an untouched test set",
               "owl_loaded": an.owl is not None, "vlm_loaded": an.vlm is not None,
               "n": len(rows), "acc": float((tp + tn) / len(rows)), "tp": tp, "fp": fp, "fn": fn,
               "tn": tn, "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
               "t_err_median": float(np.median(errs)) if errs else None,
               "sec_per_video_max": max(r["sec"] for r in rows), "rows": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    json.dump(summary, open(a.output or os.path.join(ROOT, "data", f"eval_{a.split}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
