"""MLAnalyzer — local CV pipeline (STACK.md §4 interface).

perception (YOLOX-M + YOLOX-S crop + RTMPose, ONNX) → CLIP ViT-B/32 work-area embeddings →
scene analysis (cashier / visitors / passers-by, trained track model) → candidate events →
trained scorer (models/scorer.json) → detections with descriptions and KZ norm references.
All models run locally on CPU; no network access is needed after the weights are downloaded.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from app.analyzers.base import AnalysisResult, Detection
from app.ml import pipeline
from app.ml.perception import Perception, iter_frames, video_meta
from app.ml.semantic import ClipEmbedder, crop

# seconds of CPU per sampled frame (perception + CLIP); measured on a Ryzen 5 5600H (6 cores, CPU only), Apple M3 is faster
COST_PER_FRAME = float(os.environ.get("FORTE_COST_PER_FRAME", "0.6"))


class MLAnalyzer:
    def __init__(self, threads: int | None = None):
        threads = threads or max(2, (os.cpu_count() or 4) - 2)
        self.perception = Perception(threads=threads)
        self.clip = ClipEmbedder(threads=threads)
        self.models = pipeline.Models()

    def _fps(self, duration: float, budget_sec: float) -> tuple[float, str]:
        # spend at most ~70% of the budget on perception
        fps = 0.7 * budget_sec / max(duration * COST_PER_FRAME, 1e-3)
        if fps >= 2.0:
            return 2.0, "full"
        return max(0.5, fps), "cheap"

    def analyze(self, video_path: str, meta: dict, budget_sec: float) -> AnalysisResult:
        t0 = time.time()
        vm = video_meta(video_path)
        duration = float(meta.get("duration") or vm["duration"])
        fps, mode = self._fps(duration, budget_sec)
        warnings = [] if mode == "full" else [f"мало времени: кадры анализируются с частотой {fps:.1f}/с"]

        obs, frames, wa_crops = [], [], []
        state: dict = {}
        W = H = None
        deadline = t0 + 0.8 * budget_sec
        last_t = 0.0
        for fid, t, img in iter_frames(video_path, fps):
            if W is None:
                H, W = img.shape[:2]
            o = self.perception.frame(fid, t, img, state)
            obs.append(o)
            frames.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            wa_crops.append(crop(img, o.extra.get("work_area")))
            last_t = t
            if time.time() > deadline:
                warnings.append(f"анализ остановлен по времени на {t:.0f} с из {duration:.0f} с")
                break
        t_perc = time.time() - t0
        if not obs:
            raise RuntimeError("видео не декодируется")
        clip_wa = np.concatenate([self.clip.images(wa_crops[i:i + 32]) for i in range(0, len(wa_crops), 32)])
        t_clip = time.time() - t0 - t_perc
        events, info = pipeline.run(obs, frames, W, H, self.models, clip=self.clip, clip_wa=clip_wa)

        dets = []
        for e in events:
            ev_ids = [obs[i].frame_id for i in e.evidence] or [obs[e.i].frame_id]
            dets.append(Detection(type_id=e.type_id, frame_id=int(obs[e.i].frame_id),
                                  evidence_frame_ids=[int(x) for x in ev_ids[:3]],
                                  description=pipeline.describe(e, self.models.norms), confidence=round(e.p, 3)))
        if not info["cashier_found"]:
            warnings.append("рабочее место кассира не найдено — вывод ненадёжен")
        return AnalysisResult(detections=dets, covered=[(0.0, float(last_t))], warnings=warnings,
                              timings={"perception": round(t_perc, 2), "clip": round(t_clip, 2),
                                       "total": round(time.time() - t0, 2), "frames": len(obs), "fps": fps},
                              mode=mode)
