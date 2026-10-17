"""Launcher for the Go backend (ANALYZER_COMMAND contract, see README «Подключение локальной модели»).

    cashguard_analyzer.py <absolute-video-path> [budget-seconds]
The Go backend passes the budget left until the job deadline (counted from upload); without it ANALYZER_BUDGET_SEC is used.
stdout: the completed result JSON only; all logs go to stderr; exit code 0 on success.
"""
from __future__ import annotations

import json
import os
import sys
import time

T0 = time.time()

# keep stdout clean: everything printed by libraries (incl. C extensions) goes to stderr
_REAL_STDOUT = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: cashguard_analyzer.py <video> [budget-seconds]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    from app.analyzers.ml import MLAnalyzer
    from app.ml.perception import video_meta
    from scripts.analyze_video import frame_b64
    from app.ml import pipeline

    an = MLAnalyzer()
    meta = video_meta(path)
    budget = float(sys.argv[2] if len(sys.argv) == 3 else os.environ.get("ANALYZER_BUDGET_SEC", "240"))
    budget = max(25.0, budget - (time.time() - T0))  # model loading above counts against the same deadline
    print(f"budget {budget:.0f}s after {time.time() - T0:.0f}s of startup", file=sys.stderr)
    res = an.analyze(path, meta, budget)
    fps = meta["fps"] or 30.0
    violations = []
    for d in res.detections:
        frames = [{"timestamp_sec": round(f / fps, 1), "image_base64": frame_b64(path, f)}
                  for f in d.evidence_frame_ids[:3]]
        frames = [f for f in frames if f["image_base64"]]
        if not frames:
            continue
        v = {"type": pipeline.title(d.type_id, an.models.norms), "timecode_sec": round(d.frame_id / fps, 1),
             "description": d.description, "evidence_frames": frames}
        n = an.models.norms.get(d.type_id) or {}
        if n.get("source_verified") and n.get("document"):
            v["rule"] = {"title": n["document"], "section": n["clause"], "url": n.get("url", "")}
        violations.append(v)
    out = {"status": "done", "verdict": "violation" if violations else "no_violation", "violations": violations}
    print(f"timings={res.timings} warnings={res.warnings}", file=sys.stderr)
    os.write(_REAL_STDOUT, json.dumps(out, ensure_ascii=False, allow_nan=False).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
