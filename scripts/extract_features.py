"""Run perception on dataset videos and cache per-frame observations (pickle) for training / evaluation.

usage: python scripts/extract_features.py data/train/videos data/test/videos --out data/cache --fps 3
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.perception import Perception, video_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--det", default="yolox_m.onnx")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = Perception(threads=a.threads, det_model=a.det)
    for d in a.dirs:
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".mov"):
                continue
            name = os.path.splitext(f)[0]
            dst = os.path.join(a.out, f"{name}.pkl")
            if os.path.exists(dst):
                continue
            t0 = time.time()
            path = os.path.join(d, f)
            fdir = os.path.join(a.out, "frames", name)
            os.makedirs(fdir, exist_ok=True)

            def save(o, img, fdir=fdir):
                cv2.imwrite(os.path.join(fdir, f"{o.frame_id:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

            obs = per.run(path, fps=a.fps, on_frame=save)
            with open(dst, "wb") as fh:
                pickle.dump({"meta": video_meta(path), "fps": a.fps, "obs": obs, "split": os.path.basename(os.path.dirname(d))}, fh)
            print(f"{name}: {len(obs)} frames, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
