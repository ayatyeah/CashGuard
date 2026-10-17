"""Experimental appearance residuals; these do not identify cash or keys.

No learned weights or annotations enter extraction. Missing/occluded comparisons
return no features, never a fabricated zero measurement.
"""
import cv2
import numpy as np


def residual_features(reference, targets, masks, ref_px):
    if len(targets) < 3 or ref_px <= 0:
        return {}
    common = np.logical_and.reduce(masks).astype(np.uint8)
    # Census = blur radius 1 + neighbour radius 1; ZNCC has radius 3.
    def visible(radius):
        return cv2.erode(common, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    valid = visible(2)
    zncc_valid = visible(3)
    if valid.sum() < 50:
        return {}

    def census(a):
        a = cv2.GaussianBlur(a, (3, 3), 0)
        return np.stack([a > np.roll(np.roll(a, dy, 0), dx, 1)
                         for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                                        (0, 1), (1, -1), (1, 0), (1, 1))])

    def local_correlation(a, b):
        a, b = a.astype(np.float32), b.astype(np.float32)
        mean = lambda x: cv2.blur(x, (7, 7))
        ma, mb = mean(a), mean(b)
        va = np.maximum(mean(a * a) - ma * ma, 0)
        vb = np.maximum(mean(b * b) - mb * mb, 0)
        denom = np.sqrt(va * vb)
        corr = np.ones_like(a)
        np.divide(mean(a * b) - ma * mb, denom, out=corr, where=denom > 1)
        return np.clip(1 - corr, 0, 2)

    extra = {}
    if zncc_valid.sum() >= 50:
        extra["zncc_change"] = float(np.median([
            local_correlation(reference, b)[zncc_valid].mean() for b in targets]))
    base = census(reference)
    maps = [(base != census(b)).mean(axis=0) > 0.375 for b in targets]
    persistent = np.sum(maps, axis=0) >= 3
    changed = (persistent & valid).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    compact = [area / ref_px ** 2 for x, y, w, h, area in stats[1:]
               if 4 <= area and w <= .3 * ref_px and h <= .3 * ref_px
               and area / (w * h) >= .35]
    return {**extra, "census_change": float(np.median([m[valid].mean() for m in maps])),
            "persistent_change": float(persistent[valid].mean()),
            "small_residual": float(max(compact, default=0))}
