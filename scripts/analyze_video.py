"""Analyze one video with MLAnalyzer and print the result in the jury API format (ТЗ §10).

usage: python scripts/analyze_video.py path/to/video.mov [--budget 270] [--out result.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.analyzers.ml import MLAnalyzer  # noqa: E402
from app.ml.perception import video_meta  # noqa: E402
from app.ml import pipeline  # noqa: E402


def frame_b64(path: str, frame_id: int, max_side: int = 1280) -> str:
    """JPEG (base64) of a frame; seeking near the end of some files fails, so step back a few frames."""
    cap = cv2.VideoCapture(path)
    img = None
    for f in (frame_id, frame_id - 1, frame_id - 2, frame_id - 3, frame_id - 6, frame_id - 12):
        if f < 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok and im is not None:
            img = im
            break
    cap.release()
    if img is None:
        return ""
    s = max_side / max(img.shape[:2])
    if s < 1:  # keep the stdout JSON small (Go backend limit 32 MB)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode() if ok else ""


def to_api(path: str, res, norms: dict, fps: float) -> dict:
    violations = []
    for d in res.detections:
        violations.append({
            "type": pipeline.title(d.type_id, norms),
            "type_id": d.type_id,
            "timecode_sec": round(d.frame_id / fps, 1),
            "description": d.description,
            "confidence": d.confidence,
            "evidence_frames": [{"timestamp_sec": round(f / fps, 1), "image_base64": frame_b64(path, f)}
                                for f in d.evidence_frame_ids[:3]],
        })
    return {"status": "done", "verdict": "violation" if violations else "no_violation", "violations": violations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--budget", type=float, default=270.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    an = MLAnalyzer()
    meta = video_meta(a.video)
    res = an.analyze(a.video, meta, a.budget)
    out = to_api(a.video, res, an.models.norms, meta["fps"])
    out["_timings"], out["_warnings"] = res.timings, res.warnings
    s = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    short = json.loads(s)
    for v in short["violations"]:
        for e in v["evidence_frames"]:
            e["image_base64"] = e["image_base64"][:16] + "..."
    print(json.dumps(short, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
