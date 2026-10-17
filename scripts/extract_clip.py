"""Add CLIP embeddings (full frame + work-area crop) to cached perception pickles.

usage: python scripts/extract_clip.py --cache data/cache
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.semantic import ClipEmbedder, crop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    emb = ClipEmbedder(threads=4)
    for p in sorted(glob.glob(os.path.join(a.cache, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        if "clip_full" in d:
            continue
        name = os.path.basename(p)[:-4]
        t0 = time.time()
        full, wa = [], []
        for o in d["obs"]:
            img = cv2.imread(os.path.join(a.cache, "frames", name, f"{o.frame_id:05d}.jpg"))
            full.append(img)
            wa.append(crop(img, o.extra.get("work_area")))
        d["clip_full"] = np.concatenate([emb.images(full[i:i + 32]) for i in range(0, len(full), 32)])
        d["clip_wa"] = np.concatenate([emb.images(wa[i:i + 32]) for i in range(0, len(wa), 32)])
        pickle.dump(d, open(p, "wb"))
        print(f"{name}: {len(full)} frames {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
