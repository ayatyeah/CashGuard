"""Trained scoring layer: candidate features → (type_id, probability) pairs.

With ~30 training videos and 1–5 examples per violation type, each violation type is decided by one strong,
camera-independent feature with a *learned* threshold (max-margin decision stump, scripts/train_scorer.py).
Kinds that have no positive training examples use conservative fixed rules.
"""
from __future__ import annotations

import numpy as np

# stump name → (candidate kind, feature, type_id, gates {feature: min value})
STUMPS = {
    "leave_screen": ("leave", "screen", "workstation_unlocked", {"max_d": 0.9, "end_gap": 0.5}),
    "leave_items": ("leave", "unsecured", "left_cash_unattended", {"max_d": 0.9, "end_gap": 0.5}),
    # door state is measured only when the person who entered walked away from the door to the workplace
    # (a visitor sits right by the glass wall/door and contaminates the measurement)
    "entry_door": ("entry", "door_change", "door_left_open", {"standing": 0.5}),
    # "persist" = min/max apparent height during the visit: ≈1 when the visitor never sits down
    "crowd_standing": ("crowd", "persist", "unauthorized_person", {"brief": 1.0}),
    "drink_clip": ("drink", "clip", "food_drink", {}),
    "next_visit_desk": ("next_visit", "desk_change", "cash_left_on_desk", {}),
    "safe_open": ("safe", "safe_change", "safe_left_open", {"dur": 3.0}),
    # OWLv2 on the desk crop between two client visits: item (banknotes / card) appeared after the first client came
    "next_visit_owl": ("next_visit", "owl_item", "cash_left_on_desk", {}),
    "safe_key_owl": ("safe_key", "owl_key", "safe_key_left", {}),
}

# Opt-in experimental features are excluded from the default training set.
ZONE_EXPERIMENTAL = {
    "proto_safe_zncc": ("safe", "zncc_change", "safe_left_open", {"dur": 3.0}),
    "proto_desk_persistent": ("next_visit", "persistent_change", "cash_left_on_desk", {}),
    "proto_safe_small": ("safe", "small_residual", "safe_key_left", {"dur": 3.0}),
}
STUMPS.update(ZONE_EXPERIMENTAL)

# VLM answers (app/ml/moments.py): "vlm_<key>" candidates, feature "count" = number of "yes" answers
for _k, _t in {
    "teller_holds_phone": "phone_use", "teller_drinks": "food_drink", "cup_or_food_on_desk": "food_drink",
    "handbag_or_coat_on_desk": "personal_items", "client_gives_bag": "collection_bag_no_check",
    "cash_stacks_mixed_on_desk": "cash_not_counted", "cash_left_on_desk": "left_cash_unattended",
    "documents_or_notebook_left": "left_cash_unattended", "bag_left_on_desk": "left_cash_unattended",
}.items():
    STUMPS[f"vlm_{_k}"] = (f"vlm_{_k}", "count", _t, {})

# kinds / patterns without positive training examples → conservative rules (p in 0..1)
RULES = {
    "crowd_n": ("crowd", "unauthorized_person",
                lambda f: 0.75 if (f.get("n_max", 0) >= 3 and f.get("dur", 0) >= 2.0 and "standing" not in f) else 0.0),
    # the visitor must actually walk away (track ends ≥ 0.25 ref_h farther than its closest point): a fragment of a
    # seated client's track ends at the desk and must not read as "left before the operation finished"
    "short_visit": ("short_visit", "count_not_in_client_presence",
                    lambda f: 0.7 if (f.get("dwell", 99) < 20 and f.get("max_d", 9) < 1.5
                                      and f.get("dend", 9.0) - f.get("max_d", 0.0) >= 0.25) else 0.0),
    "left_client": ("leave", "cashier_left_client_waiting",
                    lambda f: 0.75 if (f.get("client_remains", 0) >= 4.0 and f.get("max_d", 0) > 0.9
                                       and f.get("end_gap", 0) >= 0.5) else 0.0),
    "phone": ("phone", "phone_use",
              lambda f: 0.7 if (f.get("prob", 0) >= 0.85 and f.get("persist", 0) >= 4 and f.get("clip", 0) >= 0.6) else 0.0),
}


def _gate_ok(f: dict, gates: dict) -> bool:
    return all(f.get(k, 0.0) >= v for k, v in gates.items())


def score(cand, model: dict) -> list[tuple[str, float]]:
    """All (type_id, p) this candidate supports with p >= 0.5."""
    out = []
    for name, (kind, feat, type_id, gates) in STUMPS.items():
        st = model.get("stumps", {}).get(name)
        if kind != cand.kind or st is None or feat not in cand.f or not _gate_ok(cand.f, gates):
            continue
        if not st.get("enabled", True):
            continue
        p = 1.0 / (1.0 + np.exp(-(cand.f[feat] - st["thr"]) / st["scale"]))
        if p >= 0.5:
            out.append((type_id, float(p)))
    for name, (kind, type_id, rule) in RULES.items():
        if kind == cand.kind:
            p = float(rule(cand.f))
            if p >= 0.5:
                out.append((type_id, p))
    return out


def fit_stump(pos: list[float], neg: list[float]) -> dict | None:
    if not pos:
        return None
    pos, neg = np.array(pos, float), np.array(neg, float)
    if len(neg) == 0:
        thr = float(pos.min()) * 0.9
        return {"thr": thr, "scale": max(abs(thr) * 0.05, 1e-3), "separable": True, "n_pos": len(pos), "n_neg": 0}
    if pos.min() > neg.max():  # separable: max-margin threshold
        thr = (pos.min() + neg.max()) / 2
        gap = pos.min() - neg.max()
        return {"thr": float(thr), "scale": float(max(gap / 4, 1e-3)), "separable": True,
                "n_pos": len(pos), "n_neg": len(neg)}
    cands = np.unique(np.concatenate([pos, neg]))
    best = max(cands, key=lambda t: (pos >= t).mean() + (neg < t).mean())
    sd = np.concatenate([pos, neg]).std() + 1e-6
    return {"thr": float(best), "scale": float(sd / 4), "separable": False, "n_pos": len(pos), "n_neg": len(neg)}
