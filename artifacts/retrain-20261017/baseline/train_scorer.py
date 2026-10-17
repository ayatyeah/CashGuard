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
from app.ml import moments as M  # noqa: E402
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
        _, sc, cands = candidates_for(name, bank)
        vdir = os.environ.get("FORTE_VLM_DIR") or os.path.join("data", "vlm_confirmed" if "--vlm-confirmed" in sys.argv else "vlm")
        vpath = os.path.join(ROOT, vdir, f"{name}.json")
        if os.path.exists(vpath):
            cands += M.to_candidates(sc, json.load(open(vpath)))
        opath = os.path.join(ROOT, "data", "owl", f"{name}.json")
        if os.path.exists(opath):
            cache = {float(k): v for k, v in json.load(open(opath)).items()}
            for c in cands:
                if c.kind == "next_visit" and round(c.t, 1) in cache:
                    c.f["owl_item"] = float(cache[round(c.t, 1)])
        types = [t for _, t in lab["types"]] + EXTRA_TYPES.get(name, [])
        for c in cands:
            rows.append({"video": name, "kind": c.kind, "t": c.t, "f": dict(c.f), "vlabel": lab["label"],
                         "types": types, "ts": lab["ts"]})
    return rows, labels


def train(rows, enabled=None):
    model = {"stumps": {}}
    for name, (kind, feat, type_id, gates) in S.STUMPS.items():
        pos, neg = [], []
        for r in rows:
            if r["kind"] != kind or feat not in r["f"] or not S._gate_ok(r["f"], gates):
                continue
            if type_id in r["types"]:
                near = (kind.startswith("vlm_") or any(abs(r["t"] - t) <= 8.0 for t in r["ts"])
                        or type_id in EXTRA_TYPES.get(r["video"], []))
                if near:
                    pos.append(r["f"][feat])
            elif r["vlabel"] == 0 or (kind == "leave" and r["types"]):
                # normal videos; for "leave" also leaves annotated with the other leave type
                neg.append(r["f"][feat])
        st = S.fit_stump(pos, neg)
        if st:
            st["enabled"] = enabled is None or name in enabled
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


GROUPS: dict = {}  # video -> CV group (cabinet or the video itself)


def lovo_predict(rows, videos, enabled):
    """Cross-validated predictions: each group (cabinet by default) is scored by a model trained without it."""
    out = {}
    for g in sorted({GROUPS.get(v, v) for v in videos}):
        held = {v for v in videos if GROUPS.get(v, v) == g}
        m = train([r for r in rows if r["video"] not in held], enabled)
        out.update({k: e for k, e in predict([r for r in rows if r["video"] in held], m).items()})
    return out


def lovo_score(pred, labels, videos):
    """Higher is better: -(2·FN + FP), then -FP (a miss costs ≈ 2× a false alarm in the jury's scoring)."""
    fn = fp = 0
    for v in videos:
        y = int(bool(pred.get(v)))
        fn += (not y) and labels[v]["label"]
        fp += y and not labels[v]["label"]
    return -(2 * fn + fp), -fp


def accepts_cv(score, best):
    """Neither total errors nor false positives may increase."""
    return score[0] >= best[0] and score[1] >= best[1]


def main():
    # VLM features are evaluated but shipped only with --with-vlm: on the held-out test they added false
    # positives (teller_drinks, client_gives_bag) and found no new violations (docs/ml_model.md)
    with_vlm = "--with-vlm" in sys.argv
    rows, labels = collect()
    videos = sorted(n for n, l in labels.items() if l["split"] == "train")
    by_video = "--cv-video" in sys.argv
    # augmented copies always share the group of their original (never train on a copy of a held-out video)
    GROUPS.update({v: (labels[v].get("base", v) if by_video else labels[v].get("cabinet", labels[v].get("base", v)))
                   for v in videos})
    originals = [v for v in videos if "base" not in labels[v]]  # metrics on real videos only
    print("CV:", "leave-one-video-out" if by_video else "leave-one-CABINET-out",
          f"({len(set(GROUPS.values()))} groups, {len(videos)} videos, {len(originals)} originals)")
    EXPERIMENTAL = {"next_visit_desk", "safe_open", "next_visit_owl"}  # new, weakly supported features: kept only if CV does not get worse
    base = {k for k in S.STUMPS if not k.startswith("vlm_") and k not in EXPERIMENTAL and k not in S.ZONE_EXPERIMENTAL}
    best = lovo_score(lovo_predict(rows, videos, base), labels, originals)
    print("LOVO base (CV only): loss(2FN+FP)", -best[0], "FP", -best[1])
    # greedy forward selection of VLM features by LOVO (keep only if it does not hurt)
    enabled = set(base)
    for k in sorted(EXPERIMENTAL) + [k for k in S.STUMPS if k.startswith("vlm_")]:
        sc = lovo_score(lovo_predict(rows, videos, enabled | {k}), labels, originals)
        # keep if it does not hurt LOVO: single-example types cannot show a LOVO gain, but extra false
        # positives on the normal videos would show up here
        print(f"  + {k:34s} loss {-sc[0]:2d} FP {-sc[1]}", "KEEP" if accepts_cv(sc, best) else "")
        if accepts_cv(sc, best):
            best, enabled = sc, enabled | {k}
    if not with_vlm:
        enabled = {k for k in enabled if not k.startswith("vlm_")}
    print("enabled:", sorted(enabled))
    lovo = lovo_predict(rows, videos, enabled)
    report(lovo, labels, originals, "CROSS-VALIDATED (each group scored by a model trained without it):")
    model = train(rows, enabled)
    report(predict(rows, model), labels, originals, "Final model on train (resubstitution):")
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump(model, open(os.path.join(ROOT, "models", "scorer.json"), "w"), indent=1)
    print("\nstumps:", json.dumps(model["stumps"], indent=None))
    json.dump(rows, open(os.path.join(ROOT, "data", "train_rows.json"), "w"))


if __name__ == "__main__":
    main()
