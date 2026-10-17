from types import SimpleNamespace

import numpy as np

from app.ml import pipeline as P


def setup(monkeypatch):
    c = SimpleNamespace(kind='next_visit', t=10., i=0, evidence=[0], f={},
                        extra={'gap_idx': [0], 'pre_idx': [1], 'desk_box': [0, 0, 4, 4]})
    monkeypatch.setattr(P, 'analyze_scene', lambda *a, **k: SimpleNamespace(seat=[0, 0]))
    monkeypatch.setattr(P, 'generate', lambda *a: [c])
    obs = [SimpleNamespace(boxes=np.empty((0, 4)), classes=np.array([]), scores=np.array([]))] * 2
    model = SimpleNamespace(track={}, scorer={'stumps': {'next_visit_owl':
                            {'thr': .1, 'scale': .1, 'enabled': True}}})
    return c, obs, model


def test_owl_window_requires_complete_comparison(monkeypatch):
    c, obs, model = setup(monkeypatch)
    calls = []
    owl = SimpleNamespace(item_score=lambda *args: calls.append(args) or .8)
    events, info = P.run(obs, [], 4, 4, model, owl=owl, get_frame=lambda i: i,
                         inference_budget=0)
    assert not calls and not events
    assert 'owl_item' not in c.f
    assert info['owl_skipped'] == 1


def test_owl_scores_difference_and_reduces_vlm_budget(monkeypatch):
    c, obs, model = setup(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(P.time, 'monotonic', lambda: clock[0])
    def score(frame, *args):
        clock[0] += 5.
        return [.8, .2][frame]
    from app.ml import moments as M
    budgets = []
    model.scorer['stumps']['vlm_teller_drinks'] = {'enabled': True}
    monkeypatch.setattr(M, 'propose', lambda *a: [])
    monkeypatch.setattr(M, 'ask_all', lambda *a, **kw: budgets.append(kw['budget_sec']))
    monkeypatch.setattr(M, 'confirm_positives', lambda *a, **kw: None)
    events, info = P.run(obs, [], 4, 4, model, owl=SimpleNamespace(item_score=score),
                         vlm=object(), get_frame=lambda i: i, inference_budget=60, vlm_budget=60)
    assert budgets == [50.]
    assert len(events) == 1
    assert abs(c.f['owl_item'] - .6) < 1e-6
    assert info['owl_skipped'] == 0


def test_partial_owl_window_does_not_emit_false_alarm(monkeypatch):
    c, obs, model = setup(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(P.time, 'monotonic', lambda: clock[0])
    def score(*args):
        clock[0] += 10.
        return .9
    events, info = P.run(obs, [], 4, 4, model, owl=SimpleNamespace(item_score=score),
                         get_frame=lambda i: i, inference_budget=16)
    assert not events and 'owl_item' not in c.f
    assert info['owl_skipped'] == 1


def test_persistent_key_requires_both_views(monkeypatch):
    from app.ml import safe_checks as K
    c, obs, model = setup(monkeypatch)
    c.kind = 'safe_key'
    c.extra = {'key_idx': [0,1], 'key_box': [0,0,4,4]}
    monkeypatch.setattr(P,'generate',lambda *a: [])
    monkeypatch.setattr(K,'propose_key',lambda *a: [c])
    model.scorer['stumps']['safe_key_owl'] = {'thr': .1, 'scale': .1, 'enabled': True}
    for values, expected in [([.8,.9],1),([.8,.01],0)]:
        c.f = {}
        events, info = P.run(obs,[],4,4,model,
                            owl=SimpleNamespace(key_score=lambda frame,*a: values[frame]),
                            get_frame=lambda i:i,inference_budget=60)
        assert len(events) == expected
        assert c.f['owl_key'] == min(values)
        assert info['owl_skipped'] == 0
