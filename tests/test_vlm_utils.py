"""app/ml/vlm.py pure helpers: parse_json (VLM answer parsing) and compose (frame strip for the VLM)."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml import vlm as V


# ---------------------------------------------------------------- parse_json
@pytest.mark.parametrize("text, expected", [
    ('{"teller_holds_phone": true, "cash_on_desk": false}', {"teller_holds_phone": True, "cash_on_desk": False}),
    ('```json\n{"teller_holds_phone": true, "cash_on_desk": false}\n```',
     {"teller_holds_phone": True, "cash_on_desk": False}),
    ('```json\n{\n  "teller_drinks": false,\n  "desk_is_tidy": true\n}\n```',
     {"teller_drinks": False, "desk_is_tidy": True}),
    ("{'teller_holds_phone': true, 'cash_on_desk': false}", {"teller_holds_phone": True, "cash_on_desk": False}),
    ('{"teller_holds_phone": True, "cash_on_desk": False}', {"teller_holds_phone": True, "cash_on_desk": False}),
    ('{"a": true, "b": false} The teller is clearly visible.', {"a": True, "b": False}),
    # trailing text that itself contains braces: greedy match breaks json.loads, regex fallback recovers
    ('{"a": true} Note: the {phone} is on the desk.', {"a": True}),
    ('Answer:\n{"a": true}\n{"b": false}', {"a": True, "b": False}),
    # string answers are mapped to booleans
    ('{"a": "yes", "b": "No", "c": "true", "d": " TRUE "}', {"a": True, "b": False, "c": True, "d": True}),
    # non-string values are kept as-is (ask_flags applies bool())
    ('{"a": 1, "b": 0, "c": null}', {"a": 1, "b": 0, "c": None}),
])
def test_parse_json_typical_outputs(text, expected):
    assert V.parse_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "I cannot tell.", "{garbage}", "{}", "[true, false]",
                                  "{'a': maybe}", "```json\n```"])
def test_parse_json_garbage_returns_empty_dict(text):
    assert V.parse_json(text) == {}


def test_parse_json_python_dict_repr():
    # regression: single quotes AND True/False together used to return {}
    assert V.parse_json("{'teller_holds_phone': True, 'cash_on_desk': False}") == {
        "teller_holds_phone": True, "cash_on_desk": False}


def test_ask_flags_keeps_only_requested_keys_as_bools(monkeypatch):
    vlm = object.__new__(V.VLM)  # no __init__ → no model loading
    monkeypatch.setattr(vlm, "ask", lambda images, q, max_new_tokens=80: '{"a": "yes", "b": 0, "zzz": true}',
                        raising=False)
    assert V.VLM.ask_flags(vlm, [], (["a", "b", "c"], "prompt")) == {"a": True, "b": False}


def test_question_bank_prompts_mention_every_key():
    for keys, prompt in (V.Q_TELLER, V.Q_VISITOR, V.Q_DESK):
        assert keys and len(set(keys)) == len(keys)
        for k in keys:
            assert f'"{k}"' in prompt


# ---------------------------------------------------------------- compose
def _frame(h, w, value=100):
    return np.full((h, w, 3), value, np.uint8)


@pytest.mark.parametrize("sizes, max_side", [
    ([(720, 1280), (360, 640), (480, 480)], 640),
    ([(1080, 1920), (1080, 1920), (1080, 1920)], 640),
    ([(200, 100), (400, 300)], 640),           # small: no upscaling
    ([(1000, 200)], 256),                      # tall single frame
    ([(240, 320), (241, 333), (239, 199), (500, 700)], 300),
])
def test_compose_bounds_and_channels(sizes, max_side):
    frames = [_frame(h, w, 40 * k) for k, (h, w) in enumerate(sizes)]
    out = V.compose(frames, max_side=max_side)
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8
    assert max(out.shape[:2]) <= max_side
    assert min(out.shape[:2]) > 0


def test_compose_keeps_order_and_height_without_downscale():
    a, b = _frame(100, 50, 10), _frame(200, 100, 200)
    out = V.compose([a, b], max_side=10_000)
    # common height = min height (100); b is resized to 100x50 → strip 100x100
    assert out.shape == (100, 100, 3)
    assert out[:, :50].mean() == pytest.approx(10, abs=1)
    assert out[:, 50:].mean() == pytest.approx(200, abs=1)


def test_compose_downscale_keeps_aspect():
    out = V.compose([_frame(720, 1280), _frame(720, 1280)], max_side=640)
    h, w = out.shape[:2]
    assert w == 640
    assert w / h == pytest.approx(2560 / 720, rel=0.02)
