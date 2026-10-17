from types import SimpleNamespace
import numpy as np
import pytest
from app.ml import safe_checks as K
from app.ml.candidates import Cand


def scene(monkeypatch):
    monkeypatch.setattr(K, 'safe_zone', lambda *a, **kw: np.array([0.,0.,10.,10.]))
    obs = [SimpleNamespace(boxes=np.empty((0,4)), classes=np.array([]), scores=np.array([])) for _ in range(6)]
    return SimpleNamespace(obs=obs,times=np.array([0.,1.,2.,3.,4.,5.]))


def test_key_check_requires_departure(monkeypatch):
    s=scene(monkeypatch)
    assert K.propose_key(s,[]) == []


def test_key_check_uses_two_free_frames_after_departure(monkeypatch):
    s=scene(monkeypatch);leave=Cand('leave',2.,2,[2])
    c,=K.propose_key(s,[leave])
    assert c.t == 2. and c.i == 2
    assert c.extra['key_idx'] == [4,5]
    assert c.evidence == [2,4,5]
    assert c.f == {}  # geometry alone must never claim there is a key


def test_key_check_rejects_occluded_safe(monkeypatch):
    s=scene(monkeypatch)
    for o in s.obs:
        o.boxes=np.array([[0.,0.,10.,10.]])
        o.classes=np.array([0]);o.scores=np.array([1.])
    assert K.propose_key(s,[Cand('leave',2.,2,[2])]) == []


def test_key_check_requires_one_second_between_views(monkeypatch):
    s=scene(monkeypatch);s.times=np.array([0.,1.,2.,2.7,2.8,2.9])
    assert K.propose_key(s,[Cand('leave',2.,2,[2])]) == []


def test_key_check_needs_a_safe_region(monkeypatch):
    s=scene(monkeypatch)
    monkeypatch.setattr(K,'safe_zone',lambda *a,**kw:None)
    assert K.propose_key(s,[Cand('leave',2.,2,[2])]) == []
