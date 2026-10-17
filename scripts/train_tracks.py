"""Train the person-track classifier: visitor inside the cash room vs. passer-by behind the glass / other.

Labels: configs/visitors_train.json (intervals when a visitor is inside). Train split only.
Output: models/track_model.json (standardised logistic regression, no sklearn needed at runtime).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.events import TRACK_FEATS, analyze_scene  # noqa: E402
from common import CACHE, ROOT, load_labels  # noqa: E402


def collect():
    vis = json.load(open(os.path.join(ROOT, "configs", "visitors_train.json")))
    labels = load_labels()
    rows = []
    for name, l in sorted(labels.items()):
        if l["split"] != "train" or l.get("base", name) in vis.get("_exclude", []):
            continue
        d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
        sc = analyze_scene(d["obs"], d["meta"]["width"], d["meta"]["height"], track_model=None)
        iv = vis.get(l.get("base", name), [])
        for tr in sc.tracks:
            if tr.role == "cashier" or len(tr.t) < 3:
                continue
            inside = np.zeros(len(tr.t), bool)
            for a, b in iv:
                inside |= (tr.t >= a) & (tr.t <= b)
            # a visitor track must also come close to the desk; far tracks in a visitor interval are passers-by
            y = int(inside.mean() > 0.5 and tr.feat["dmin"] < 1.2 and tr.feat["frac_close"] > 0.25)
            rows.append((name, tr.tid, y, [tr.feat[k] for k in TRACK_FEATS]))
    return rows


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    return {"feats": TRACK_FEATS, "mean": mu.tolist(), "std": sd.tolist(), "coef": m.coef_[0].tolist(),
            "intercept": float(m.intercept_[0])}


def predict(model, X):
    z = ((X - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-z))


def main():
    rows = collect()
    names = np.array([r[0] for r in rows]); y = np.array([r[2] for r in rows]); X = np.array([r[3] for r in rows])
    print(f"tracks: {len(y)}  visitors: {y.sum()}")
    # leave-one-video-out
    p = np.zeros(len(y))
    for n in sorted(set(names)):
        tr, te = names != n, names == n
        if y[tr].min() == y[tr].max():
            continue
        p[te] = predict(fit(X[tr], y[tr]), X[te])
    pred = p >= 0.5
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    print(f"LOVO: acc={np.mean(pred == y):.3f} TP={tp} FP={fp} FN={fn}")
    for n, yy, pp, x in zip(names, y, p, X):
        if (pp >= 0.5) != yy:
            print(f"   err {n} y={yy} p={pp:.2f} " + " ".join(f"{k}={v:.2f}" for k, v in zip(TRACK_FEATS, x)))
    model = fit(X, y)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump(model, open(os.path.join(ROOT, "models", "track_model.json"), "w"), indent=1)
    print("coef:", {k: round(c, 2) for k, c in zip(TRACK_FEATS, model["coef"])})


if __name__ == "__main__":
    main()
