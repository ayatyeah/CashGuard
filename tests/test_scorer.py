"""app/ml/scorer.py: decision-stump fitting and candidate scoring (gates, disabled stumps, fixed rules)."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from app.ml import scorer as S


def _p(st, x):
    return 1.0 / (1.0 + np.exp(-(x - st["thr"]) / st["scale"]))


def cand(kind, **f):
    return SimpleNamespace(kind=kind, f=dict(f))


# ---------------------------------------------------------------- fit_stump
def test_fit_stump_no_positives_returns_none():
    assert S.fit_stump([], [0.1, 0.2]) is None
    assert S.fit_stump([], []) is None


def test_fit_stump_separable_max_margin():
    pos, neg = [3.0, 4.0, 5.0], [0.0, 1.0]
    st = S.fit_stump(pos, neg)
    assert st["separable"] is True
    assert st["thr"] == pytest.approx(2.0)          # midpoint of the gap (1, 3)
    assert st["scale"] == pytest.approx(0.5)        # gap / 4
    assert (st["n_pos"], st["n_neg"]) == (3, 2)
    assert all(_p(st, x) > 0.5 for x in pos)
    assert all(_p(st, x) < 0.5 for x in neg)
    json.dumps(st)  # must be serialisable into models/scorer.json


def test_fit_stump_non_separable_best_accuracy_threshold():
    pos, neg = [1.0, 3.0, 5.0, 6.0], [0.0, 2.0, 4.0]
    st = S.fit_stump(pos, neg)
    assert st["separable"] is False
    assert st["thr"] in set(pos + neg)
    assert st["scale"] > 0
    pos_a, neg_a = np.array(pos), np.array(neg)
    acc = (pos_a >= st["thr"]).mean() + (neg_a < st["thr"]).mean()
    best = max((pos_a >= t).mean() + (neg_a < t).mean() for t in pos + neg)
    assert acc == pytest.approx(best)
    json.dumps(st)


def test_fit_stump_identical_values_does_not_crash():
    st = S.fit_stump([1.0, 1.0], [1.0])
    assert st["separable"] is False and st["scale"] > 0 and np.isfinite(st["thr"])


def test_fit_stump_no_negatives_accepts_all_positives():
    pos = [2.0, 3.0, 10.0]
    st = S.fit_stump(pos, [])
    assert st["separable"] is True and st["n_neg"] == 0
    assert st["thr"] < min(pos)
    assert all(_p(st, x) >= 0.5 for x in pos)


@pytest.mark.xfail(strict=True, reason=(
    "latent BUG app/ml/scorer.py:69 — with no negatives thr = 0.9*min(pos), which is ABOVE min(pos) when the "
    "feature is negative, so that positive scores < 0.5. Harmless today (all stump features are >= 0)."))
def test_fit_stump_no_negatives_negative_feature():
    st = S.fit_stump([-1.0, 2.0], [])
    assert _p(st, -1.0) >= 0.5


# ---------------------------------------------------------------- score
MODEL = {"stumps": {
    "leave_screen": {"thr": 1.5, "scale": 0.1},
    "leave_items": {"thr": 0.5, "scale": 0.1},
    "entry_door": {"thr": 0.13, "scale": 0.01},
    "crowd_standing": {"thr": 0.65, "scale": 0.1},
    "vlm_teller_holds_phone": {"thr": 0.5, "scale": 0.1},
}}
LEAVE_GATES = {"max_d": 1.2, "end_gap": 1.0}


def types(out):
    return sorted(t for t, _ in out)


def test_score_leave_screen_fires_with_gates_open():
    out = S.score(cand("leave", screen=2.0, unsecured=0.0, **LEAVE_GATES), MODEL)
    assert types(out) == ["workstation_unlocked"]
    assert all(0.5 <= p <= 1.0 and isinstance(p, float) for _, p in out)


def test_score_two_stumps_on_one_candidate():
    out = S.score(cand("leave", screen=2.0, unsecured=1.0, **LEAVE_GATES), MODEL)
    assert types(out) == ["left_cash_unattended", "workstation_unlocked"]


def test_score_below_threshold_is_dropped():
    assert S.score(cand("leave", screen=1.0, unsecured=0.1, **LEAVE_GATES), MODEL) == []


@pytest.mark.parametrize("gates", [
    {"max_d": 0.5, "end_gap": 1.0},     # cashier did not walk far
    {"max_d": 1.2, "end_gap": 0.1},     # track ends right at the end of the video
    {"end_gap": 1.0},                   # missing gate feature counts as 0
    {},
])
def test_score_gates_block(gates):
    assert S.score(cand("leave", screen=5.0, unsecured=5.0, **gates), MODEL) == []


def test_score_disabled_stump_is_ignored():
    model = json.loads(json.dumps(MODEL))
    model["stumps"]["leave_screen"]["enabled"] = False
    out = S.score(cand("leave", screen=2.0, unsecured=1.0, **LEAVE_GATES), model)
    assert types(out) == ["left_cash_unattended"]
    model["stumps"]["leave_screen"]["enabled"] = True
    assert "workstation_unlocked" in types(S.score(cand("leave", screen=2.0, **LEAVE_GATES), model))


def test_score_requires_matching_kind_feature_and_stump():
    assert S.score(cand("entry", screen=5.0, **LEAVE_GATES), MODEL) == []      # wrong kind
    assert S.score(cand("leave", **LEAVE_GATES), MODEL) == []                  # feature missing
    assert S.score(cand("leave", screen=5.0, **LEAVE_GATES), {}) == []         # no stumps in model
    assert S.score(cand("leave", screen=5.0, **LEAVE_GATES), {"kinds": {}}) == []  # pipeline default model


def test_score_entry_and_vlm_stumps():
    assert types(S.score(cand("entry", door_change=0.3), MODEL)) == ["door_left_open"]
    assert S.score(cand("entry", door_change=0.05), MODEL) == []
    assert types(S.score(cand("vlm_teller_holds_phone", count=2.0, frac=0.5), MODEL)) == ["phone_use"]
    # vlm kind with no fitted stump in the model → nothing
    assert S.score(cand("vlm_teller_drinks", count=5.0), MODEL) == []


def test_score_rules_crowd():
    # fixed rule: >=3 persons for >=2 s and no "standing" feature (n_inside run candidate)
    out = S.score(cand("crowd", n_max=3.0, dur=2.5), {})
    assert out == [("unauthorized_person", 0.75)]
    assert S.score(cand("crowd", n_max=2.0, dur=10.0), {}) == []
    # visitor-track crowd candidates carry "standing": the rule is skipped, only the stump decides
    assert S.score(cand("crowd", n_max=3.0, dur=5.0, standing=0.2), MODEL) == []
    assert types(S.score(cand("crowd", n_max=3.0, dur=5.0, standing=0.9), MODEL)) == ["unauthorized_person"]


def test_score_rules_short_visit_and_phone():
    assert S.score(cand("short_visit", dwell=10.0, max_d=1.0), {}) == [("count_not_in_client_presence", 0.7)]
    assert S.score(cand("short_visit", dwell=30.0, max_d=1.0), {}) == []
    assert S.score(cand("short_visit", dwell=10.0, max_d=2.0), {}) == []
    assert S.score(cand("phone", prob=0.9, persist=5.0, clip=0.7), {}) == [("phone_use", 0.7)]
    assert S.score(cand("phone", prob=0.9, persist=5.0, clip=0.1), {}) == []


def test_stump_definitions_are_consistent():
    from app.ml.candidates import FEATS, KINDS
    for name, (kind, feat, type_id, gates) in S.STUMPS.items():
        assert type_id
        if not kind.startswith("vlm_"):
            assert kind in KINDS, name
            assert feat in FEATS, name
            assert all(g in FEATS for g in gates), name


def test_trained_scorer_json_matches_stump_names():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "scorer.json")
    if not os.path.exists(path):
        pytest.skip("models/scorer.json not present")
    with open(path) as fh:
        model = json.load(fh)
    for name, st in model.get("stumps", {}).items():
        assert name in S.STUMPS, f"unknown stump {name} in scorer.json"
        assert np.isfinite(st["thr"]) and st["scale"] > 0, name
