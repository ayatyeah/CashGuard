"""Build weak-label proposals for a small supervised dataset (image + question + JSON answer) for LoRA fine-tuning of the VLM
(mlx-vlm on a Mac) from our cached frames and the per-video ground truth in docs/train_timelines.md.

Pairs come from the same key moments the runtime asks about (teller at work, client visit, desk after leaving,
safe zone, between visits), but video-level truth must be reviewed on every crop before training.
Split is by cabinet (configs/cabinets.json): --val-cabinets C,E  → val, the rest → train.

usage: python scripts/make_vlm_sft.py --out data/sft [--val-cabinets C,E] [--aug 0] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import os
import pickle
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, load_labels  # noqa: E402
from app.ml import moments as M  # noqa: E402
from app.ml import vlm as V  # noqa: E402
from app.ml.candidates import crop_scaled, generate  # noqa: E402
from app.ml.events import analyze_scene  # noqa: E402

# ---- ground truth per video and moment kind (from docs/train_timelines.md). None = unknown → skip the key.
TELLER = {  # keys: teller_holds_phone, teller_drinks, cup_or_food_on_desk, handbag_or_coat_on_desk
    "Х-46": {"teller_drinks": True, "cup_or_food_on_desk": True, "handbag_or_coat_on_desk": False, "teller_holds_phone": False},
    "Х-40": {"handbag_or_coat_on_desk": True, "teller_drinks": False, "cup_or_food_on_desk": False, "teller_holds_phone": False},
}
TELLER_NEG = ["Х-43", "Х-45", "Х-44", "Х-8", "Х-21", "Х-3", "Х-35", "Х-39", "Х-5", "Х-19", "Х-14", "Х-25", "Х-27", "Х-31"]
DESK = {  # after the cashier left: cash_left_on_desk, documents_or_notebook_left, bag_left_on_desk, laptop_screen_on, desk_is_tidy
    "Х-26": {"cash_left_on_desk": True, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": False},
    "Х-24": {"cash_left_on_desk": False, "documents_or_notebook_left": True, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": False},
    "Х-15": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": True, "laptop_screen_on": False, "desk_is_tidy": False},
    "Х-28": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": True, "desk_is_tidy": True},
    "Х-11": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": True, "desk_is_tidy": True},
    "Х-25": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
    "Х-27": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
    "Х-12": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
    "Х-23": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
    "Х-29": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
    "Х-22": {"cash_left_on_desk": False, "documents_or_notebook_left": False, "bag_left_on_desk": False, "laptop_screen_on": False, "desk_is_tidy": True},
}
SAFE = {  # safe_door_open, key_in_safe_lock
    "Х-20": {"safe_door_open": True, "key_in_safe_lock": False},
    "Х-22": {"safe_door_open": False, "key_in_safe_lock": True},
    **{n: {"safe_door_open": False, "key_in_safe_lock": False} for n in ["Х-21", "Х-23", "Х-29", "Х-25", "Х-27", "Х-8", "Х-19", "Х-35", "Х-39", "Х-14", "Х-12", "Х-11", "Х-15", "Х-24"]},
}
VISITOR = {  # client_shows_id_card, client_gives_cash, client_gives_bag
    "Х-3": {"client_shows_id_card": True, "client_gives_cash": True, "client_gives_bag": False},
    "Х-5": {"client_shows_id_card": True, "client_gives_cash": True, "client_gives_bag": False},
    "Х-19": {"client_shows_id_card": True, "client_gives_cash": True, "client_gives_bag": False},
    "Х-35": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "Х-37": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "Х-39": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "Х-34": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "X-2": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "Х-38": {"client_shows_id_card": False, "client_gives_cash": True, "client_gives_bag": False},
    "Х-17": {"client_shows_id_card": False, "client_gives_cash": False, "client_gives_bag": True},
    "Х-16": {"client_shows_id_card": True, "client_gives_cash": False, "client_gives_bag": True},
    "X-1": {"client_shows_id_card": True, "client_gives_cash": True, "client_gives_bag": False},
}
BETWEEN = {"Х-4": {"cash_on_desk": True}, "Х-38": {"cash_on_desk": True},
           "Х-5": {"cash_on_desk": False}, "Х-19": {"cash_on_desk": False}, "Х-39": {"cash_on_desk": False}}


CONTEXT = {
    "teller": "The image shows a bank teller at her desk (several moments).",
    "visitor": "The image shows moments of a client visit at a bank teller desk, in time order from left to right.",
    "desk": "The image shows a bank teller desk after the teller walked away.",
    "safe": "The image shows a metal safe (cash cabinet) in a bank office, several moments.",
    "between": "The image shows a bank teller at her desk between two client visits.",
}


def answer_json(keys: list[str], truth: dict):
    """Only keys with known ground truth go into the question and the answer."""
    known = {k: truth[k] for k in keys if isinstance(truth.get(k), bool)}
    if not known:
        return None, None
    return list(known), json.dumps(known)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    a = rng.uniform(0.8, 1.2); b = rng.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=a, beta=b)
    h, w = out.shape[:2]
    s = rng.uniform(0.9, 1.0)
    x0, y0 = int((1 - s) * w * rng.random()), int((1 - s) * h * rng.random())
    return out[y0:y0 + int(s * h), x0:x0 + int(s * w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--val-cabinets", default="C,E")
    ap.add_argument("--aug", type=int, default=0, help="augmented copies per pair (train split only)")
    ap.add_argument("--dry-run", action="store_true", help="validate/generate row counts without writing images or JSONL")
    a = ap.parse_args()
    if a.aug < 0:
        ap.error("--aug must be nonnegative")
    rng = random.Random(0)
    labels = load_labels()
    val_cab = {c.strip() for c in a.val_cabinets.split(",") if c.strip()}
    known_cab = {v.get("cabinet") for v in labels.values()} - {None}
    if not val_cab or not val_cab <= known_cab:
        ap.error(f"unknown or empty validation cabinets: {val_cab}")
    if any(not v.get("cabinet") for v in labels.values()):
        ap.error("every video must have a cabinet; refusing possible train/val leakage")
    if not a.dry_run:
        if Path(a.out).exists() and any(Path(a.out).iterdir()):
            ap.error("output must be empty; stale images/rows can contaminate splits")
        os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    manifest = []
    image_splits = {}
    rows = {"train": [], "val": []}
    stats = {}
    for name, lab in sorted(labels.items()):
        if "base" in lab or lab.get("split") != "train":
            continue
        d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
        obs = d["obs"]
        frames = [cv2.imread(os.path.join(CACHE, "frames", name, f"{o.frame_id:05d}.jpg")) for o in obs]
        if any(f is None for f in frames):
            raise ValueError(f"missing/corrupt cached frame: {name}")
        sc = analyze_scene(obs, d["meta"]["width"], d["meta"]["height"])
        leaves = [c for c in generate(sc, [cv2.resize(f, None, fx=0.5, fy=0.5) for f in frames]) if c.kind == "leave"]
        moms = M.propose(sc, frames, leaves)
        split = "val" if lab.get("cabinet") in val_cab else "train"
        for mi, m in enumerate(moms):
            kind = m["kind"]
            neg_t = {"teller_holds_phone": False, "teller_drinks": False, "cup_or_food_on_desk": False,
                     "handbag_or_coat_on_desk": False}  # cash_on_desk unknown while counting → not asked
            truth = {"teller": TELLER.get(name, neg_t if name in TELLER_NEG else {}),
                     "between": BETWEEN.get(name, {}), "desk": DESK.get(name, {}), "safe": SAFE.get(name, {}),
                     "visitor": VISITOR.get(name, {})}[kind]
            all_keys = ["cash_on_desk"] if kind == "between" else getattr(V, m["q"])[0]
            keys, ans = answer_json(all_keys, truth)
            if ans is None:
                continue
            prompt = V._json_prompt(CONTEXT[kind], keys)
            imgs = [crop_scaled(frames[i], m["crop"], 1.0) for i in m["idx"]]
            variants = [(V.compose(imgs), "orig")]
            if split == "train":
                for k in range(a.aug):
                    variants.append((V.compose([augment(im, rng) for im in imgs]), f"aug{k}"))
            for img, tag in variants:
                fn = f"{name.replace('Х', 'X')}_{kind}_{mi}_{tag}.jpg"
                digest = hashlib.sha256(img.tobytes()).hexdigest()
                previous = image_splits.setdefault(digest, split)
                if previous != split:
                    raise ValueError("identical image appears across train/validation")
                if not a.dry_run:
                    if not cv2.imwrite(os.path.join(a.out, "images", fn), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError(f"cannot write {fn}")
                manifest.append({"video": name, "cabinet": lab["cabinet"], "split": split,
                                 "kind": kind, "image": fn, "sha256_pixels": digest,
                                 "frame_ids": [int(obs[i].frame_id) for i in m["idx"]],
                                 "label_status": "weak_video_labels_review_crop_before_training"})
                rows[split].append({"images": [f"images/{fn}"],
                                    "messages": [{"role": "system", "content": V.SYSTEM},
                                                 {"role": "user", "content": prompt},
                                                 {"role": "assistant", "content": ans}]})
                stats[(split, kind)] = stats.get((split, kind), 0) + 1
    if a.dry_run:
        print("DRY RUN: no files written; labels remain weak until crop review")
        print({s: len(rs) for s, rs in rows.items()})
        return
    for split, rs in rows.items():
        with open(os.path.join(a.out, f"{'validation' if split == 'val' else split}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out, "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print({f"{s}/{k}": n for (s, k), n in sorted(stats.items())})
    print(f"train {len(rows['train'])}  val {len(rows['val'])}  → {a.out}")


if __name__ == "__main__":
    main()
