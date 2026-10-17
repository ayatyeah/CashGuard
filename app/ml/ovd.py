"""Open-vocabulary detection (OWLv2, Apache-2.0) on zoomed, person-masked desk crops.
Used only in event windows (desk between two client visits) where the question is "is a small item — a cash bundle
or a card — lying on the desk that was not there before the first client?".
"""
from __future__ import annotations

import os

import cv2
import numpy as np

OWL_ID = os.environ.get("FORTE_OWL", "google/owlv2-base-patch16-ensemble")
ITEMS = ["a stack of banknotes", "an identity card", "a passport"]   # "paper document" fires on normal desks
DISTRACTORS = ["a paper document"]


class Owl:
    def __init__(self, threads: int = 6):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        torch.set_num_threads(threads)
        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(OWL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).eval()
        self.queries = ITEMS + DISTRACTORS

    def detect(self, crop: np.ndarray, threshold: float = 0.1, queries=None) -> dict:
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        s = max(im.size)
        queries = self.queries if queries is None else queries
        with self.torch.no_grad():
            inp = self.proc(text=[queries], images=im, return_tensors="pt")
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, threshold=threshold, target_sizes=self.torch.tensor([[s, s]]))[0]
        best: dict = {}
        for sc, lab in zip(res["scores"], res["labels"]):
            name = lab if isinstance(lab, str) else queries[int(lab)]
            best[name] = max(best.get(name, 0.0), float(sc))
        return best

    def key_score(self, frame, box):
        from .candidates import crop_scaled
        crop = crop_scaled(frame, box, 1.0)
        if crop.size == 0:
            return 0.0
        queries = ["a key in a lock", "a metal key", "a safe door handle", "a metal cabinet door", "a padlock"]
        return self.detect(crop, threshold=0.01, queries=queries).get("a metal key", 0.0)

    def item_score(self, frame: np.ndarray, box, person_boxes, scale: float = 1.0) -> float:
        """Max score over ITEMS on the desk crop (persons painted over, crop upscaled to ~960 px)."""
        x1, y1, x2, y2 = (np.asarray(box) * scale).astype(int)
        c = frame[max(0, y1):y2, max(0, x1):x2].copy()
        if c.size == 0:
            return 0.0
        for b in person_boxes:
            bx1, by1, bx2, by2 = (np.asarray(b) * scale).astype(int)
            cv2.rectangle(c, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (120, 120, 120), -1)
        h, w = c.shape[:2]
        f = 960 / max(h, w)
        if f > 1:
            c = cv2.resize(c, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
        r = self.detect(c)
        return max((r.get(k, 0.0) for k in ITEMS), default=0.0)
