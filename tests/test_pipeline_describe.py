"""app/ml/pipeline.py: describe() / title() with the real norms from app/rules/norms.yaml."""
from __future__ import annotations

import pytest

from app.ml import pipeline as PL

SUFFIX = " Результат — подсказка для проверки человеком."
INTERP = " (связанное требование)"


def ev(type_id, t=42.4, **f):
    return PL.Event(type_id=type_id, t=t, i=0, evidence=[0], p=0.9, kind="test", f=dict(f))


@pytest.mark.parametrize("type_id", sorted(PL.TEMPLATES))
def test_every_template_renders_with_minimal_features(type_id, norms):
    for f in ({}, {"count": 1.0}):
        text = PL.describe(ev(type_id, **f), norms)
        assert isinstance(text, str) and text.endswith(SUFFIX)
        assert "{" not in text and "}" not in text
        assert "42" in text  # t formatted with {t:.0f}


@pytest.mark.parametrize("type_id", sorted(PL.TEMPLATES))
def test_norm_reference_appended_when_source_verified(type_id, norms):
    n = norms.get(type_id) or {}
    text = PL.describe(ev(type_id), norms)
    if n.get("source_verified") and n.get("document"):
        assert f"{n['document']}, {n['clause']}." in text
        if n.get("interpretive"):
            assert f"Норматив{INTERP}: " in text
        else:
            assert "Норматив: " in text and f"Норматив{INTERP}" not in text
    else:
        assert "Норматив" not in text


def test_interpretive_and_direct_norms_both_present_in_yaml(norms):
    """Sanity: the parametrized test above really covers both branches."""
    verified = {k: v for k, v in norms.items() if k in PL.TEMPLATES and v.get("source_verified")}
    assert any(v.get("interpretive") for v in verified.values())
    assert any(not v.get("interpretive") for v in verified.values())


def test_specific_marking(norms):
    assert f"Норматив{INTERP}:" in PL.describe(ev("door_left_open"), norms)          # interpretive: true
    direct = PL.describe(ev("left_cash_unattended"), norms)                          # interpretive: false
    assert "Норматив: " in direct and INTERP not in direct


@pytest.mark.parametrize("norm, has_ref", [
    ({"source_verified": False, "document": "Doc", "clause": "p. 1"}, False),
    ({"source_verified": True, "document": "", "clause": "p. 1"}, False),
    ({"source_verified": True, "document": "Doc", "clause": "p. 1"}, True),
    ({"source_verified": True, "document": "Doc", "clause": "p. 1", "interpretive": True}, True),
])
def test_norm_reference_rules_on_synthetic_norms(norm, has_ref):
    text = PL.describe(ev("phone_use"), {"phone_use": norm})
    assert ("Doc, p. 1." in text) is has_ref
    assert (INTERP in text) is bool(has_ref and norm.get("interpretive"))


def test_unknown_type_and_empty_norms():
    text = PL.describe(ev("something_new"), {})
    assert text.startswith("Около 42 с") and text.endswith(SUFFIX)
    assert "Норматив" not in text
    assert PL.title("something_new", {}) == "something_new"


def test_feature_values_are_rendered(norms):
    text = PL.describe(ev("workstation_unlocked", t=10.0, screen=2.345), norms)
    assert "2.3 раза" in text
    text = PL.describe(ev("count_not_in_client_presence", t=10.0, dwell=14.6), norms)
    assert "через 15 с" in text
    # NaN / int / bool feature values must not break formatting
    PL.describe(ev("workstation_unlocked", screen=float("nan"), standing=True, n_max=3), norms)


def test_every_emittable_type_has_template_title_and_norm(norms):
    """Every type the system can emit must have a description template and a titled norm entry
    (otherwise the UI falls back to a raw type id / generic text)."""
    from app.ml import scorer as S
    from app.ml.candidates import DEFAULT_TYPE
    from app.ml.moments import VLM_TYPES

    emitted = ({t for _, _, t, _ in S.STUMPS.values()} | {t for _, t, _ in S.RULES.values()}
               | set(DEFAULT_TYPE.values()) | set(VLM_TYPES.values()))
    for t in emitted:
        assert t in PL.TEMPLATES, t
        assert norms.get(t, {}).get("title"), t
        assert PL.title(t, norms) == norms[t]["title"]


def test_verified_norms_have_fields_used_by_describe_and_analyzer(norms):
    """describe() and scripts/cashguard_analyzer.py index n['clause'] when source_verified + document."""
    for k, n in norms.items():
        if n.get("source_verified") and n.get("document"):
            assert n.get("clause"), k
            assert str(n.get("url", "")).startswith("https://"), k
