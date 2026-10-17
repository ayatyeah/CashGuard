"""Train the decision-stump scorer on the TRAIN split and report leave-one-video-out (LOVO) metrics.

Positives for a stump: candidates of its kind in videos annotated with its violation type
(configs/train_annotations.json, plus a few verified extra events). Negatives: candidates of that kind in
videos without violations and in videos annotated with a *different* type of the same kind.
Output: models/scorer.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, candidates_for, load_labels  # noqa: E402
from app.ml import scorer as S  # noqa: E402

# true events visible in train videos besides the organisers' timestamp (docs/train_timelines.md)
EXTRA_TYPES = {"Х-30": ["door_left_open"], "Х-33": ["unauthorized_person"], "Х-17": ["unauthorized_person"]}


def collect():
    from app.ml.semantic import ClipEmbedder
    clip = ClipEmbedder(threads=6)
    bank = clip.probe_bank()
    labels = load_labels()
    rows = []
    for name, lab in sorted(labels.items()):
        if lab["split"] != "train":
            continue
        _, _, cands = candidates_for(name, bank)
        types = [t for _, t in lab["types"]] + EXTRA_TYPES.get(name, [])
        for c in cands:
            rows.append({"video": name, "kind": c.kind, "t": c.t, "f": dict(c.f), "vlabel": lab["label"],
                         "types": types, "ts": lab["ts"]})
    return rows, labels


def train(rows):
    model = {"stumps": {}}
    for name, (kind, feat, type_id, gates) in S.STUMPS.items():
        pos, neg = [], []
        for r in rows:
            if r["kind"] != kind or feat not in r["f"] or not S._gate_ok(r["f"], gates):
                continue
            if type_id in r["types"]:
                near = any(abs(r["t"] - t) <= 8.0 for t in r["ts"]) or type_id in EXTRA_TYPES.get(r["video"], [])
                if near:
                    pos.append(r["f"][feat])
            elif r["vlabel"] == 0 or (kind == "leave" and r["types"]):
                # normal videos; for "leave" also leaves annotated with the other leave type
                neg.append(r["f"][feat])
        st = S.fit_stump(pos, neg)
        if st:
            model["stumps"][name] = st
    return model


class C:  # minimal candidate view for scoring stored rows
    def __init__(self, r):
        self.kind, self.f, self.t = r["kind"], r["f"], r["t"]


def predict(rows, model):
    out = {}
    for r in rows:
        for type_id, p in S.score(C(r), model):
            out.setdefault(r["video"], []).append((type_id, round(r["t"], 1), round(p, 2)))
    return out


def report(pred, labels, videos, title):
    tp = fp = tn = fn = 0
    dts, type_ok = [], []
    print(f"\n{title}")
    for v in videos:
        lab = labels[v]["label"]
        ev = pred.get(v, [])
        y = int(bool(ev))
        tp += y and lab; fp += y and not lab; tn += (not y) and (not lab); fn += (not y) and lab
        if y and lab and labels[v]["ts"]:
            dts.append(min(abs(e[1] - t) for e in ev for t in labels[v]["ts"]))
            ann = {t for _, t in labels[v]["types"]}
            if ann:
                type_ok.append(any(e[0] in ann for e in ev))
        print(f"  {v:5s} y={lab} pred={y} {'OK ' if y == lab else 'ERR'} {ev} ts={labels[v]['ts']}")
    n = tp + fp + tn + fn
    print(f"verdict acc={(tp + tn) / n:.3f} TP={tp} FP={fp} TN={tn} FN={fn} "
          f"recall={tp / max(1, tp + fn):.2f} precision={tp / max(1, tp + fp):.2f}")
    if dts:
        dts = np.array(dts)
        print(f"timecode |err|: median={np.median(dts):.1f}s mean={dts.mean():.1f}s  <=2s {np.mean(dts <= 2):.2f}  "
              f"<=5s {np.mean(dts <= 5):.2f};  type correct on detected: {np.mean(type_ok):.2f}")
    return (tp + tn) / n


def main():
    rows, labels = collect()
    videos = sorted(n for n, l in labels.items() if l["split"] == "train")
    lovo = {}
    for v in videos:
        m = train([r for r in rows if r["video"] != v])
        lovo.update({k: e for k, e in predict([r for r in rows if r["video"] == v], m).items()})
    report(lovo, labels, videos, "LOVO (each video scored by a model trained without it):")
    model = train(rows)
    report(predict(rows, model), labels, videos, "Final model on train (resubstitution):")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump(model, open(os.path.join(ROOT, "models", "scorer.json"), "w"), indent=1)
    print("\nstumps:", json.dumps(model["stumps"], indent=None))
    json.dump(rows, open(os.path.join(ROOT, "data", "train_rows.json"), "w"))


if __name__ == "__main__":
    main()
