"""Minimal person tracker (IoU + centre distance, Hungarian assignment). Camera is static, motion is slow."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-6)


@dataclass
class Track:
    tid: int
    obs: list = field(default_factory=list)   # list of (frame_idx, det_idx, box, score)
    last_box: np.ndarray | None = None
    last_i: int = -1
    misses: int = 0


class Tracker:
    def __init__(self, max_miss: int = 6, min_iou: float = 0.1, max_center: float = 0.25):
        self.max_miss, self.min_iou, self.max_center = max_miss, min_iou, max_center
        self.active: list[Track] = []
        self.done: list[Track] = []
        self.next_id = 0

    def step(self, fi: int, boxes: np.ndarray, scores: np.ndarray, diag: float) -> list[int]:
        """Returns track id per detection."""
        ids = [-1] * len(boxes)
        if self.active and len(boxes):
            tb = np.stack([t.last_box for t in self.active])
            cost = 1 - iou(tb, boxes)
            tc = (tb[:, :2] + tb[:, 2:]) / 2; dc = (boxes[:, :2] + boxes[:, 2:]) / 2
            dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / diag
            cost = cost + dist
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if (1 - cost[i, j] + dist[i, j]) >= self.min_iou or dist[i, j] < self.max_center * 0.4:
                    if dist[i, j] > self.max_center:
                        continue
                    t = self.active[i]
                    t.obs.append((fi, j, boxes[j], float(scores[j]))); t.last_box = boxes[j]; t.last_i = fi; t.misses = 0
                    ids[j] = t.tid
        for j in range(len(boxes)):
            if ids[j] < 0:
                t = Track(self.next_id, [(fi, j, boxes[j], float(scores[j]))], boxes[j], fi)
                self.next_id += 1
                self.active.append(t)
                ids[j] = t.tid
        keep = []
        for t in self.active:
            if t.last_i != fi:
                t.misses += 1
            (keep if t.misses <= self.max_miss else self.done).append(t)
        self.active = keep
        return ids

    def finish(self) -> list[Track]:
        return sorted(self.done + self.active, key=lambda t: t.tid)
