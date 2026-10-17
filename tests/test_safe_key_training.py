import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_scorer import train


def row(name,score,types,label):
    return dict(video=name,kind='safe_key',t=10.,f={'owl_key':score},types=types,vlabel=label,ts=[10.])


def test_single_positive_keeps_conservative_floor():
    model=train([row('positive',.06,['safe_key_left'],1),row('normal',.01,[],0)])
    assert model['stumps']['safe_key_owl']['thr'] >= .054


def test_other_violation_types_are_negative_key_examples():
    model=train([row('positive',.06,['safe_key_left'],1),row('door',.05,['door_left_open'],1)])
    assert model['stumps']['safe_key_owl']['n_neg'] == 1
    assert model['stumps']['safe_key_owl']['thr'] > .05


def test_no_positive_does_not_train_key_detector():
    model=train([row('normal',.01,[],0)])
    assert 'safe_key_owl' not in model['stumps']
