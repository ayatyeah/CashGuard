"""app/ml/candidates.py pure helpers: change_score, _overlap, entry_index (fake track / observations)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.ml import candidates as CD
from app.ml.events import PTrack


# ---------------------------------------------------------------- change_score
def _halves(h=120, w=160, left=30, right=220):
    img = np.full((h, w), left, np.uint8)
    img[:, w // 2:] = right
    return img


def test_change_score_identical_is_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    mask = np.ones(a.shape, bool)
    assert CD.change_score(a, a.copy(), mask) == 0.0


def test_change_score_global_brightness_shift_is_ignored():
    a = _halves()
    b = np.clip(a.astype(int) + 25, 0, 255).astype(np.uint8)
    assert CD.change_score(a, b, np.ones(a.shape, bool)) < 0.01


def test_change_score_very_different_is_high():
    a = _halves()
    b = a[:, ::-1].copy()  # bright and dark halves swapped (e.g. door opened)
    s = CD.change_score(a, b, np.ones(a.shape, bool))
    assert 0.9 < s <= 1.0


def test_change_score_respects_mask():
    a = _halves()
    b = a.copy()
    b[:, :40] = 255 - b[:, :40]          # change only the leftmost quarter
    mask = np.zeros(a.shape, bool)
    mask[:, 60:] = True                  # look elsewhere (beyond the blur radius)
    assert CD.change_score(a, b, mask) == 0.0
    # a uniformly changed region alone would be normalised away (mean subtraction), so look at changed +
    # unchanged pixels together
    mask_changed = np.zeros(a.shape, bool)
    mask_changed[:, 20:60] = True
    assert CD.change_score(a, b, mask_changed) > 0.5


@pytest.mark.parametrize("n", [0, 1, 49])
def test_change_score_tiny_mask_is_nan(n):
    a = _halves()
    mask = np.zeros(a.shape, bool)
    mask.flat[:n] = True
    assert math.isnan(CD.change_score(a, a[:, ::-1].copy(), mask))


# ---------------------------------------------------------------- _overlap
B = np.array([0.0, 0.0, 10.0, 10.0])


def test_overlap_empty():
    assert CD._overlap(B, np.zeros((0, 4))) is False


@pytest.mark.parametrize("other, expected", [
    ([0, 0, 10, 10], True),          # identical
    ([2, 2, 8, 8], True),            # 36 % of B covered (> 15 %)
    ([20, 20, 30, 30], False),       # disjoint
    ([9, 0, 20, 10], False),         # 10 % of B covered
    ([8, 0, 20, 10], True),          # 20 % of B covered
    ([10, 0, 20, 10], False),        # touching edge
])
def test_overlap_fraction_of_first_box(other, expected):
    assert CD._overlap(B, np.array([other], float)) is expected


def test_overlap_is_relative_to_first_box_and_any_of_many():
    big = np.array([0.0, 0.0, 100.0, 100.0])
    small = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert CD._overlap(big, small) is False      # 1 % of the big box
    assert CD._overlap(small[0], big[None]) is True
    many = np.array([[50, 50, 60, 60], [200, 200, 210, 210], [0, 0, 10, 10]], float)
    assert CD._overlap(B, many) is True


# ---------------------------------------------------------------- entry_index
def make_track(t0, n, scores, kp=0.9, move_px=0.0, h=200.0, w=80.0, fps=2.0):
    """Track sampled at `fps`; centre moves `move_px` per sample along x. obs contain the person box plus a
    far-away distractor so the nearest-box matching in entry_index is exercised."""
    t = t0 + np.arange(n) / fps
    cx = 300.0 + move_px * np.arange(n)
    cy = np.full(n, 400.0)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    scores = np.broadcast_to(np.asarray(scores, float), (n,))
    obs = []
    for k in range(n):
        distractor = [1000, 50, 1080, 250]
        obs.append(SimpleNamespace(boxes=np.array([distractor, boxes[k]], float),
                                   scores=np.array([0.99, scores[k]])))
    kps = np.broadcast_to(np.asarray(kp, float), (n,))[:, None].repeat(17, 1)
    tr = PTrack(tid=1, idx=np.arange(n), t=t, boxes=boxes, kpts=np.zeros((n, 17, 2)), kps=kps)
    return tr, obs


def test_entry_index_behind_glass_then_inside():
    scores = [0.5, 0.6, 0.55, 0.95, 0.96, 0.97, 0.95, 0.9]
    tr, obs = make_track(5.0, len(scores), scores)
    assert CD.entry_index(tr, obs) == 3


def test_entry_index_needs_two_consecutive_confident_frames():
    scores = [0.5, 0.95, 0.5, 0.95, 0.5, 0.95, 0.5]
    tr, obs = make_track(5.0, len(scores), scores)
    assert CD.entry_index(tr, obs) is None


def test_entry_index_low_keypoint_visibility_is_not_inside():
    tr, obs = make_track(5.0, 8, 0.95, kp=0.3)
    assert CD.entry_index(tr, obs) is None


def test_entry_index_never_confident():
    tr, obs = make_track(5.0, 8, 0.6)
    assert CD.entry_index(tr, obs) is None


def test_entry_index_present_from_start_and_static_is_not_an_entry():
    tr, obs = make_track(0.0, 12, 0.95, move_px=0.0)
    assert CD.entry_index(tr, obs) is None


def test_entry_index_present_from_start_but_walks_in():
    # 40 px per sample at 2 fps → 320 px in 4 s = 1.6 heights ≥ 0.6
    tr, obs = make_track(0.0, 12, 0.95, move_px=40.0)
    assert CD.entry_index(tr, obs) == 0
