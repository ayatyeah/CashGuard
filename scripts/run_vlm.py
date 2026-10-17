"""Ask the local VLM about key moments of cached videos; save answers to data/vlm/<name>.json (for training).

usage: python scripts/run_vlm.py [--split train] [--names X-2 Х-4 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, load_labels, load_video  # noqa: E402
from app.ml import moments as M  # noqa: E402
from app.ml.candidates import generate  # noqa: E402
from app.ml.events import analyze_scene  # noqa: E402
from app.ml.vlm import VLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--out", default=os.path.join("data", "vlm"), help="output dir for answers (e.g. artifacts/vlm_8b)")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    labels = load_labels()
    names = a.names or sorted(n for n, l in labels.items() if l["split"] == a.split)
    vlm = VLM()
    for name in names:
        dst = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(dst):
            continue
        d, frames = load_video(name)
        sc = analyze_scene(d["obs"], d["meta"]["width"], d["meta"]["height"])
        leaves = [c for c in generate(sc, frames) if c.kind == "leave"]
        moms = M.propose(sc, frames, leaves)

        def get_frame(i, d=d, name=name):
            return cv2.imread(os.path.join(CACHE, "frames", name, f"{d['obs'][i].frame_id:05d}.jpg"))

        t0 = time.time()
        M.ask_all(vlm, moms, get_frame)
        for m in moms:
            m["crop"] = [float(x) for x in m["crop"]]
        json.dump(moms, open(dst, "w"), ensure_ascii=False, indent=1)
        print(f"{name}: {len(moms)} moments, {time.time() - t0:.0f}s  " +
              " | ".join(f"{m['kind']}@{m['t']:.0f}:{','.join(k for k, v in m['answer'].items() if v)}" for m in moms),
              flush=True)


if __name__ == "__main__":
    main()
