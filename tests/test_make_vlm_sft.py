"""Guard against training unknown answers as confidently false."""
from scripts.make_vlm_sft import answer_json
import json


def test_unknown_ground_truth_is_not_negative():
    keys, answer = answer_json(['yes', 'no', 'unknown', 'missing'],
                              {'yes': True, 'no': False, 'unknown': None})
    assert keys == ['yes', 'no']
    assert json.loads(answer) == {'yes': True, 'no': False}


def test_all_unknown_skips_row():
    assert answer_json(['unknown'], {'unknown': None}) == (None, None)
