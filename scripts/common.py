"""Shared helpers for training / evaluation scripts (dataset cache → scene → candidates)."""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from app.ml.candidates import generate  # noqa: E402
from app.ml.events import analyze_scene  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")


def norm_name(n: str) -> str:
    return n.replace("Х", "X").strip()


def parse_ts(s: str) -> float | None:
    if not s:
        return None
    m = re.findall(r"(\d+)\s*мин", s)
    sec = re.findall(r"(\d+)\s*сек", s)
    return (int(m[0]) * 60 if m else 0) + (int(sec[0]) if sec else 0)


def load_labels() -> dict:
    """{cache_name: {"label": 0/1, "ts": [seconds], "split": train/test, "types": [...]}}"""
    ann = json.load(open(os.path.join(ROOT, "configs", "train_annotations.json")))
    out = {}
    for split in ("train", "test"):
        p = os.path.join(ROOT, "data", split, "labels.csv")
        for r in csv.DictReader(open(p)):
            name = r["file"]
            ts = parse_ts(r["timestamp"])
            e = out.setdefault(name, {"label": 0, "ts": [], "split": split, "types": []})
            if r["label"].strip() == "нарушение":
                e["label"] = 1
                if ts is not None:
                    e["ts"].append(float(ts))
    # duplicates: Х-47 ≡ Х-4, Х-48 ≡ Х-5 — merge timestamps into the original, drop the copy
    for dup, orig in (("Х-47", "Х-4"), ("Х-48", "Х-5")):
        if dup in out and orig in out:
            out[orig]["ts"] = sorted(set(out[orig]["ts"] + out[dup]["ts"]))
            del out[dup]
    for name, a in ann.items():
        if name.startswith("_"):
            continue
        if name in out:
            out[name]["types"] = [(e["t"], e["type_id"]) for e in a["events"]]
    return out


def load_video(name: str, with_frames: bool = True, frame_scale: float = 0.5):
    d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
    frames = []
    if with_frames:
        for o in d["obs"]:
            img = cv2.imread(os.path.join(CACHE, "frames", name, f"{o.frame_id:05d}.jpg"))
            if frame_scale != 1.0:
                img = cv2.resize(img, None, fx=frame_scale, fy=frame_scale, interpolation=cv2.INTER_AREA)
            frames.append(img)
    return d, frames


def probes_for(d: dict, bank: dict | None):
    if bank is None or "clip_wa" not in d:
        return {}
    from app.ml.semantic import probe_scores
    return probe_scores(d["clip_wa"], bank)


def candidates_for(name: str, bank=None):
    d, frames = load_video(name)
    sc = analyze_scene(d["obs"], d["meta"]["width"], d["meta"]["height"])
    return d, sc, generate(sc, frames, probes_for(d, bank))
