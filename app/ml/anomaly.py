"""Zone-state anomaly on DINOv2-small patch features (PatchCore / AnomalyDINO style, Apache-2.0).

Idea: we only know what "normal" looks like (desk before the client arrives, safe before it is opened, normal end
states from other videos). A region is anomalous if its patch features are far from every normal patch.
Score = mean of the top-q% nearest-neighbour cosine distances over the region's patches (person patches masked).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

DINO_ID = os.environ.get("FORTE_DINO", "facebook/dinov2-small")
PATCH = 14


class DinoPatch:
    def __init__(self, threads: int = 4, size: int = 224):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModel.from_pretrained(DINO_ID).eval()
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def features(self, bgr: np.ndarray, mask: np.ndarray | None = None):
        """→ (N,384) L2-normalised patch features of the crop, patches overlapping the mask (True=exclude) dropped."""
        s = self.size
        img = cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None])
        with self.torch.no_grad():
            out = self.model(pixel_values=t).last_hidden_state[0]
        n_reg = out.shape[0] - (s // PATCH) ** 2  # CLS (+ registers)
        f = out[n_reg:].numpy().astype(np.float32)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-6
        if mask is not None:
            g = s // PATCH
            m = cv2.resize(mask.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA) > 0.3
            f = f[~m.ravel()]
        return f


def anomaly_score(test: np.ndarray, bank: np.ndarray, top_q: float = 0.05) -> float:
    """Mean of the top-q fraction of nearest-neighbour cosine distances (robust to a few noisy patches)."""
    if len(test) == 0 or len(bank) == 0:
        return float("nan")
    sim = test @ bank.T
    d = 1.0 - sim.max(axis=1)
    k = max(1, int(np.ceil(top_q * len(d))))
    return float(np.sort(d)[-k:].mean())


def crop_and_mask(frame: np.ndarray, box, person_boxes, scale: float):
    x1, y1, x2, y2 = (np.asarray(box) * scale).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    c = frame[y1:y2, x1:x2]
    m = np.zeros(c.shape[:2], bool)
    for b in person_boxes:
        bx1, by1, bx2, by2 = (np.asarray(b) * scale).astype(int)
        w, h = bx2 - bx1, by2 - by1
        bx1, by1, bx2, by2 = bx1 - int(0.15 * w) - x1, by1 - int(0.1 * h) - y1, bx2 + int(0.15 * w) - x1, by2 + int(0.1 * h) - y1
        m[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = True
    return c, m
