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

# seconds of CPU per sampled frame (perception + CLIP); measured on a Ryzen 5 5600H (6 cores, CPU only).
# In Docker on Apple M3 8 GB it is ~1.1 s, so src/backend/Dockerfile.ml sets FORTE_COST_PER_FRAME=1.2.
COST_PER_FRAME = float(os.environ.get("FORTE_COST_PER_FRAME", "0.6"))


class MLAnalyzer:
    def __init__(self, threads: int | None = None):
        # in containers os.cpu_count() is the host's core count, not the CPU quota → allow an explicit override
        threads = threads or int(os.environ.get("FORTE_THREADS", "0")) or max(2, (os.cpu_count() or 4) - 2)
        self.perception = Perception(threads=threads)
        try:
            self.clip = ClipEmbedder(threads=threads)
        except Exception as e:  # CLIP weights missing → analysis without the CLIP probes
            print(f"CLIP disabled: {e}", file=__import__("sys").stderr)
            self.clip = None
        self.models = pipeline.Models()
        self.owl = None
        if os.environ.get("FORTE_USE_OWL", "0") == "1" and any(
                k in {"next_visit_owl", "safe_key_owl"} and st.get("enabled", True)
                for k, st in self.models.scorer.get("stumps", {}).items()):
            try:
                from app.ml.ovd import Owl
                self.owl = Owl(threads=threads)
            except Exception as e:
                print(f"OWL disabled: {e}", file=__import__("sys").stderr)
        self.vlm = None
        if os.environ.get("FORTE_USE_VLM", "0") == "1" and pipeline.uses_vlm(self.models):
            try:
                from app.ml.vlm import VLM
                self.vlm = VLM(threads=threads)
            except Exception as e:  # weights missing / backend unavailable → CV-only mode
                print(f"VLM disabled: {e}", file=__import__("sys").stderr)
        # Without a VLM the post-perception stages (CLIP, OWL, evidence) took ≤ 10 s of a 260 s budget on Railway,
        # so perception gets 75 % instead of 60 %: long videos are sampled denser (leave detection degrades below
        # 1 frame/s: 6 → 10 errors on the 40 cached videos when thinned to 1 fps).
        if self.vlm is None:
            self.PERCEPTION_SHARE = 0.75

    PERCEPTION_SHARE = 0.6   # share of the budget for perception; the rest: CLIP, VLM, evidence frames
    VLM_MIN_LEFT = 90.0      # run the VLM only if at least this many seconds remain after perception

    def _fps(self, duration: float, budget_sec: float) -> tuple[float, str]:
        fps = self.PERCEPTION_SHARE * budget_sec / max(duration * COST_PER_FRAME, 1e-3)
        if fps >= 2.0:
            return 2.0, "full"
        return max(0.5, fps), "cheap"

    @staticmethod
    def shared_budget(t0: float, budget_sec: float) -> float:
        """The server lowers a running job's deadline when more videos arrive (FORTE_DEADLINE_FILE holds the absolute
        unix deadline). Returns the budget measured from t0, never larger than the original one."""
        p = os.environ.get("FORTE_DEADLINE_FILE")
        if not p:
            return budget_sec
        try:
            dl = float(open(p).read().strip())
        except (OSError, ValueError):
            return budget_sec
        return min(budget_sec, max(1.0, dl - 15.0 - t0))  # same 15 s reserve for evidence/saving as the server

    def analyze(self, video_path: str, meta: dict, budget_sec: float) -> AnalysisResult:
        t0 = time.time()
        budget_sec = self.shared_budget(t0, budget_sec)
        vm = video_meta(video_path)
        duration = float(meta.get("duration") or vm["duration"])
        fps, mode = self._fps(duration, budget_sec)
        warnings = [] if mode == "full" else [f"мало времени: кадры анализируются с частотой {fps:.1f}/с"]

        obs, frames, wa_crops = [], [], []
        state: dict = {}
        W = H = None
        deadline = t0 + 0.8 * budget_sec
        last_t = 0.0
        truncated = False
        # adaptive sampling: after a few frames re-estimate the real per-frame cost on this machine and
        # thin the sampling so that perception finishes within its share of the budget
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("видео не декодируется")
        src_fps = vm["fps"]
        step = max(1, int(round(src_fps / fps)))
        i = 0
        t_perc_start = time.time()
        while True:
            if not cap.grab():
                break
            if i % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    h0, w0 = img.shape[:2]
                    s = 960 / max(h0, w0)
                    if s < 1:
                        img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
                    if W is None:
                        H, W = img.shape[:2]
                    t = i / src_fps
                    o = self.perception.frame(i, t, img, state)
                    obs.append(o)
                    frames.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
                    wa_crops.append(cv2.resize(crop(img, o.extra.get("work_area")), (224, 224)))
                    last_t = t
                    if len(obs) >= 6 and len(obs) % 6 == 0:
                        nb = self.shared_budget(t0, budget_sec)
                        if nb < budget_sec:  # more videos arrived: shrink to this job's share
                            budget_sec = nb
                            deadline = t0 + 0.8 * budget_sec
                            if "мало времени: частота кадров снижена" not in warnings:
                                warnings.append("мало времени: частота кадров снижена")
                        # re-plan: thin the sampling over the REST of the video rather than cut its end off
                        per = (time.time() - t_perc_start) / len(obs)
                        remaining = max(0.0, duration - t)
                        allowed = self.PERCEPTION_SHARE * budget_sec - (time.time() - t0)
                        need = remaining * (src_fps / step) * per
                        if need > allowed and allowed > 0:
                            new_step = int(np.ceil(step * need / allowed))
                            if new_step > step:
                                step = min(new_step, int(src_fps * 4))  # floor 0.25 fps
                                if mode != "cheap":
                                    mode = "cheap"
                                    warnings.append("мало времени: частота кадров снижена")
                    if time.time() > deadline:
                        warnings.append(f"анализ остановлен по времени на {t:.0f} с из {duration:.0f} с")
                        truncated = True
                        break
            i += 1
        cap.release()
        t_perc = time.time() - t0
        if not obs:
            raise RuntimeError("видео не декодируется")
        clip_wa = (np.concatenate([self.clip.images(wa_crops[i:i + 32]) for i in range(0, len(wa_crops), 32)])
                   if self.clip is not None else None)
        t_clip = time.time() - t0 - t_perc
        fid = [o.frame_id for o in obs]

        def get_frame(i):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid[i])
            ok, img = cap.read()
            cap.release()
            if not ok or img is None:
                return cv2.resize(frames[i], (W, H))
            # crops are in perception coordinates (frames may have been downscaled to max_side=960)
            return img if img.shape[:2] == (H, W) else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        budget_sec = self.shared_budget(t0, budget_sec)
        vlm_budget = max(0.0, budget_sec - (time.time() - t0) - 20.0)
        if vlm_budget < self.VLM_MIN_LEFT:
            vlm_budget = 0.0  # not enough time for the VLM → CV-only result (never a timeout)
        events, info = pipeline.run(obs, frames, W, H, self.models, clip=self.clip, clip_wa=clip_wa,
                                    vlm=self.vlm, get_frame=get_frame, vlm_budget=vlm_budget,
                                    truncated=truncated, owl=self.owl,
                                    inference_budget=max(0.0, budget_sec - (time.time() - t0) - 20.0))
        if info.get("owl_skipped"):
            warnings.append(f"OWL: не завершено проверок предметов: {info['owl_skipped']} (нет времени)")
        if info.get("vlm_skipped"):
            warnings.append(f"VLM: пропущено {info['vlm_skipped']} из {info['vlm_moments']} моментов (нет времени)")

        dets = []
        for e in events:
            ev_ids = [obs[i].frame_id for i in e.evidence] or [obs[e.i].frame_id]
            if obs[e.i].frame_id not in ev_ids and len(ev_ids) < 3:  # the timecode frame itself is evidence too
                ev_ids = sorted(ev_ids + [obs[e.i].frame_id])
            dets.append(Detection(type_id=e.type_id, frame_id=int(obs[e.i].frame_id),
                                  evidence_frame_ids=[int(x) for x in ev_ids[:3]],
                                  description=pipeline.describe(e, self.models.norms), confidence=round(e.p, 3)))
        if not info["cashier_found"]:
            warnings.append("рабочее место кассира не найдено — вывод ненадёжен")
        return AnalysisResult(detections=dets, covered=[(0.0, float(last_t))], warnings=warnings,
                              timings={"perception": round(t_perc, 2), "clip": round(t_clip, 2),
                                       "budget": round(budget_sec, 1), "step": step,
                                       "vlm_moments": info.get("vlm_moments", 0),
                                       "owl_key_candidates": info.get("owl_key_candidates", 0),
                                       "owl_key_max": info.get("owl_key_max", 0.0),
                                       "total": round(time.time() - t0, 2), "frames": len(obs), "fps": fps},
                              mode=mode)
