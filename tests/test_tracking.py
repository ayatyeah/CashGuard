"""app/ml/tracking.py: IoU helper and Tracker on synthetic moving boxes."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.tracking import Tracker, iou

W, H = 1280, 720
DIAG = float(np.hypot(W, H))


def box(cx, cy, w=100.0, h=220.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def step(tr, fi, boxes):
    b = np.array(boxes, float).reshape(-1, 4)
    return tr.step(fi, b, np.full(len(b), 0.9), DIAG)


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)
    assert iou(a, b).shape == (1, 3)


def test_slowly_moving_box_keeps_id():
    tr = Tracker()
    ids = [step(tr, fi, [box(300 + 6 * fi, 400 + 1 * fi)]) for fi in range(40)]
    assert all(i == [0] for i in ids)
    tracks = tr.finish()
    assert len(tracks) == 1 and len(tracks[0].obs) == 40


def test_new_far_box_gets_new_id_and_old_id_is_kept():
    tr = Tracker()
    for fi in range(10):
        assert step(tr, fi, [box(300 + 5 * fi, 400)]) == [0]
    seen = {0}
    for fi in range(10, 20):
        ids = step(tr, fi, [box(300 + 5 * fi, 400), box(1100, 300)])  # far box appears at frame 10
        assert ids[0] == 0
        seen.add(ids[1])
    assert seen == {0, 1}
    assert [t.tid for t in tr.finish()] == [0, 1]


def test_detection_order_does_not_matter():
    tr = Tracker()
    step(tr, 0, [box(300, 400), box(1000, 400)])
    for fi in range(1, 10):
        a, b = box(300 + 4 * fi, 400), box(1000 - 4 * fi, 400)
        ids = step(tr, fi, [b, a] if fi % 2 else [a, b])
        assert ids == ([1, 0] if fi % 2 else [0, 1])


def test_teleport_beyond_max_center_is_a_new_track():
    tr = Tracker()
    step(tr, 0, [box(200, 400)])
    assert step(tr, 1, [box(200 + 0.3 * DIAG, 400)]) == [1]   # > max_center (0.25 diag)


@pytest.mark.parametrize("gap, same", [(3, True), (6, True), (7, False)])
def test_short_occlusion_keeps_id_long_one_does_not(gap, same):
    tr = Tracker(max_miss=6)
    for fi in range(5):
        step(tr, fi, [box(400, 400)])
    for fi in range(5, 5 + gap):
        assert step(tr, fi, np.zeros((0, 4))) == []
    ids = step(tr, 5 + gap, [box(405, 400)])
    assert (ids == [0]) is same


def test_finish_returns_active_and_done_sorted():
    tr = Tracker(max_miss=1)
    step(tr, 0, [box(200, 400), box(900, 400)])
    for fi in range(1, 5):
        step(tr, fi, [box(900, 400)])           # track 0 disappears → moved to done
    out = tr.finish()
    assert [t.tid for t in out] == [0, 1]
    assert len(out[0].obs) == 1 and len(out[1].obs) == 5
    fi, j, b, s = out[1].obs[-1]
    assert (fi, j, s) == (4, 0, pytest.approx(0.9))
