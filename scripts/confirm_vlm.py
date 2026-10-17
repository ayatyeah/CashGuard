"""Self-consistency ensemble for cached VLM answers: re-ask positives on perturbed views, keep 2-of-3 votes.

reads data/vlm/<name>.json, writes data/vlm_confirmed/<name>.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT  # noqa: E402
from app.ml import moments as M  # noqa: E402
from app.ml.vlm import VLM  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("data", "vlm"))
    ap.add_argument("--dst", default=os.path.join("data", "vlm_confirmed"))
    a = ap.parse_args()
    src, dst = os.path.join(ROOT, a.src), os.path.join(ROOT, a.dst)
    os.makedirs(dst, exist_ok=True)
    vlm = None
    for f in sorted(os.listdir(src)):
        name = f[:-5]
        out = os.path.join(dst, f)
        if os.path.exists(out):
            continue
        moms = json.load(open(os.path.join(src, f)))
        d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
        need = any((m.get("answer") or {}).get(k) for m in moms
                   for k in ("teller_drinks", "handbag_or_coat_on_desk", "client_gives_bag"))
        if need:
            vlm = vlm or VLM()
            t0 = time.time()

            def get_frame(i, d=d, name=name):
                return cv2.imread(os.path.join(CACHE, "frames", name, f"{d['obs'][i].frame_id:05d}.jpg"))

            M.confirm_positives(vlm, moms, get_frame, d["meta"]["width"], d["meta"]["height"], len(d["obs"]))
            print(f"{name}: {[m.get('votes') for m in moms if m.get('votes')]} {time.time() - t0:.0f}s", flush=True)
        json.dump(moms, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
