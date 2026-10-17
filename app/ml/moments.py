"""Key moments for VLM questions, proposed from the scene (camera-agnostic) + VLM answers → candidates."""
from __future__ import annotations

import numpy as np

from . import vlm as V
from .candidates import Cand, _persons, crop_scaled, static_regions
from .events import Scene

MAX_TELLER_Q = 5


def _union(boxes, W, H, pad=0.25):
    b = np.array(boxes, float)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = x2 - x1, y2 - y1
    return np.array([max(0, x1 - pad * w), max(0, y1 - pad * h), min(W, x2 + pad * w), min(H, y2 + pad * h)])


def _box_at(tr, i):
    k = np.where(tr.idx == i)[0]
    return tr.boxes[k[0]] if len(k) else None


def safe_zone(scene: Scene, pad: float = 0.6):
    """Where the cashier works standing away from her seat (the safe / cabinet): wrists while standing still."""
    pts = []
    for tr in scene.tracks:
        if tr.tid not in scene.cashier_ids or len(tr.t) < 3:
            continue
        d = tr.info["d"]
        rel = tr.h / scene.ref_h
        c = tr.center
        v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(tr.t).clip(1e-3) / scene.ref_h]
        from .candidates import entry_index
        k_in = entry_index(tr, scene.obs)
        t_ok = tr.t[k_in] + 5.0 if k_in is not None else -1.0  # standing right after entering = at the door
        for k in range(len(tr.t)):
            if d[k] > 0.25 and rel[k] > 1.3 and v[k] < 0.35 and tr.t[k] > t_ok:
                w, s = tr.kpts[k, [9, 10]], tr.kps[k, [9, 10]]
                pts += list(w[s > 0.3])
    if len(pts) < 4:
        return None
    p = np.array(pts)
    med = np.median(p, axis=0)
    keep = np.linalg.norm(p - med, axis=1) < 1.0 * scene.ref_h  # one cluster (the safe), drop outliers
    if keep.sum() < 4:
        return None
    p = p[keep]
    r = pad * scene.ref_h
    x1, y1 = p.min(0) - r; x2, y2 = p.max(0) + r
    return np.array([max(0, x1), max(0, y1), min(scene.W, x2), min(scene.H, y2)], np.float32)


def _box_overlap(box, boxes, thr=0.05):
    if boxes is None or len(boxes) == 0:
        return False
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return bool((inter / ((box[2] - box[0]) * (box[3] - box[1]) + 1e-6) > thr).any())


def visit_runs(scene: Scene, near_d: float = 0.85, min_len: float = 3.0) -> list[tuple[float, float]]:
    """Visits = runs of a visitor sitting/standing at the desk (consecutive clients may share one track)."""
    out = []
    for tr in scene.tracks:
        if tr.role != "visitor":
            continue
        near = tr.info["d"] < near_d
        k = 0
        while k < len(near):
            if not near[k]:
                k += 1
                continue
            e = k
            while e + 1 < len(near) and (near[e + 1] or (e + 2 < len(near) and near[e + 2] and tr.t[e + 2] - tr.t[e] < 2.0)):
                e += 1
            if tr.t[e] - tr.t[k] >= min_len:
                out.append((float(tr.t[k]), float(tr.t[e])))
            k = e + 1
    return sorted(out)


def propose(scene: Scene, frames: list, leave_cands: list) -> list[dict]:
    """→ [{"kind": teller|visitor|desk, "q": question, "idx": [obs indices], "crop": box, "t": time}]"""
    obs, times, W, H = scene.obs, scene.times, scene.W, scene.H
    cashier = [t for t in scene.tracks if t.role == "cashier"]
    visitors = [t for t in scene.tracks if t.role == "visitor"]
    out = []
    # --- teller at work: frames where the cashier sits at the workplace, spread over the video
    present = np.where(scene.present)[0]
    if len(present) >= 2:
        n_q = int(min(MAX_TELLER_Q, max(1, round((times[present[-1]] - times[present[0]]) / 12.0))))
        chunks = np.array_split(present, n_q)
        for ch in chunks:
            if len(ch) < 2:
                continue
            idx = [int(ch[len(ch) // 3]), int(ch[(2 * len(ch)) // 3])]
            boxes = [b for i in idx for tr in cashier for b in [_box_at(tr, i)] if b is not None]
            if not boxes:
                continue
            out.append({"kind": "teller", "q": "Q_TELLER", "idx": idx, "crop": _union(boxes, W, H, 0.45),
                        "t": float(times[idx[0]])})
    # --- visits: 4 frames from arrival to departure
    for tr in visitors:
        near = tr.idx[tr.info["d"] < 1.6]
        if len(near) < 4:
            continue
        idx = [int(near[k]) for k in np.linspace(0, len(near) - 1, 4).astype(int)]
        boxes = [b for i in idx for t in cashier + [tr] for b in [_box_at(t, i)] if b is not None]
        dwell = float(tr.t[-1] - tr.t[0])
        real = tr.feat["kpt"] > 0.5 and tr.feat["score"] > 0.6
        out.append({"kind": "visitor", "q": "Q_VISITOR", "idx": idx, "crop": _union(boxes, W, H, 0.2),
                    "t": float(times[idx[0]]), "tid": tr.tid,
                    "short_visit": bool(real and dwell < 15.0 and tr.t[-1] < times[-1] - 1.0)})
    # --- safe / cabinet after the cashier walked away from it (door left open, key left in the lock)
    sz = safe_zone(scene)
    if sz is not None:
        from .candidates import _persons
        free = [i for i in range(len(obs)) if not _box_overlap(sz, _persons(obs[i], 0.3))]
        # only moments after the cashier has been at the safe at least once
        at_safe = [i for tr in scene.tracks if tr.tid in scene.cashier_ids
                   for k, i in enumerate(tr.idx) if tr.info["d"][k] > 0.25 and (tr.h[k] / scene.ref_h) > 1.3]
        if at_safe:
            t0 = times[min(at_safe)]
            free = [i for i in free if times[i] > t0 + 1.0]
        if len(free) >= 2:
            idx = [free[len(free) // 2], free[-1]]
            out.append({"kind": "safe", "q": "Q_SAFE", "idx": idx, "crop": sz, "t": float(times[idx[0]])})
    # --- between two client visits: the desk should be clear (previous client's cash locked away)
    runs = visit_runs(scene)
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        if b0 - a1 < 3.0 or a1 - a0 < 8.0:
            continue
        gap = [i for i in range(len(obs)) if a1 + 1.0 <= times[i] <= b0 - 0.5 and scene.present[i]]
        if len(gap) < 2:
            continue
        idx = [gap[len(gap) // 2], gap[-1]]
        boxes = [b for i in idx for tr in cashier for b in [_box_at(tr, i)] if b is not None]
        if not boxes:
            continue
        out.append({"kind": "between", "q": "Q_TELLER", "idx": idx, "crop": _union(boxes, W, H, 0.45),
                    "t": float(max(a1, b0 - 3.0))})
    # --- desk after the cashier left
    for c in leave_cands:
        box = c.extra.get("desk_box")
        late = [i for i in range(len(obs)) if times[i] >= c.t + 2.0]
        if box is None or not late:
            continue
        idx = [late[len(late) // 2], late[-1]]
        out.append({"kind": "desk", "q": "Q_DESK", "idx": idx, "crop": box, "t": c.t})
    return out


def _jitter_crop(box, W, H, scale):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * scale, (y2 - y1) * scale
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


def confirm_positives(vlm, moments: list[dict], get_frame, W: int, H: int, n_obs: int,
                      keys=("teller_drinks", "handbag_or_coat_on_desk", "client_gives_bag"),
                      budget_sec: float | None = None) -> list[dict]:
    """Self-consistency ensemble: re-ask positive answers on 2 perturbed views (tighter/looser crop, shifted
    frames); keep a "yes" only if at least 2 of 3 answers agree. Only positives are re-asked → cheap."""
    import sys
    import time
    t0 = time.time()
    for m in moments:
        a = m.get("answer") or {}
        pos = [k for k in keys if a.get(k)]
        if not pos:
            continue
        votes = {k: 1 for k in pos}
        variants = [(0.8, 1), (1.25, -1)]
        for sc, shift in variants:
            if budget_sec is not None and time.time() - t0 > budget_sec:
                break
            idx = [min(n_obs - 1, max(0, i + shift)) for i in m["idx"]]
            crop = _jitter_crop(m["crop"], W, H, sc)
            try:
                imgs = [crop_scaled(get_frame(i), crop, 1.0) for i in idx]
                b = vlm.ask_flags(imgs, getattr(V, m["q"]))
            except Exception as e:
                print(f"VLM confirm failed: {e}", file=sys.stderr)
                continue
            for k in pos:
                votes[k] += int(bool(b.get(k)))
        m["votes"] = votes
        for k in pos:
            if votes[k] < 2:
                a[k] = False
    return moments


def ask_all(vlm, moments: list[dict], get_frame, budget_sec: float | None = None) -> list[dict]:
    """get_frame(obs_index) -> full-resolution BGR frame (crops are in original pixel coordinates)."""
    import os
    import sys
    import time
    t0 = time.time()
    est = float(os.environ.get("FORTE_VLM_CALL_SEC", "35"))  # expected seconds per VLM call (updated online)
    for m in moments:
        if budget_sec is not None and (time.time() - t0) + est > budget_sec:
            m["answer"], m["skipped"] = {}, True
            continue
        t1 = time.time()
        try:
            imgs = [crop_scaled(get_frame(i), m["crop"], 1.0) for i in m["idx"]]
            m["answer"] = vlm.ask_flags(imgs, getattr(V, m["q"]))
        except Exception as e:  # one failed question must not lose the CV result
            print(f"VLM moment failed: {e}", file=sys.stderr)
            m["answer"], m["skipped"] = {}, True
        dt = time.time() - t1
        est = max(0.5 * est + 0.5 * dt, dt)
    return moments


# VLM answer key → violation type it supports (other keys are context only)
VLM_TYPES = {
    "teller_holds_phone": "phone_use",
    "teller_drinks": "food_drink",
    "cup_or_food_on_desk": "food_drink",
    "handbag_or_coat_on_desk": "personal_items",
    "client_gives_bag": "collection_bag_no_check",
    "cash_stacks_mixed_on_desk": "cash_not_counted",
    "cash_left_on_desk": "left_cash_unattended",
    "documents_or_notebook_left": "left_cash_unattended",
    "bag_left_on_desk": "left_cash_unattended",
}


def to_candidates(scene: Scene, moments: list[dict]) -> list[Cand]:
    """One candidate per VLM key that was answered "yes" at least once.
    kind = "vlm_<key>", t = first moment with "yes", f = {count, frac} over moments of that kind."""
    out = []
    by_kind: dict = {}
    for m in moments:
        if m.get("answer"):
            by_kind.setdefault(m["kind"], []).append(m)
    by_tid = {tr.tid: tr for tr in scene.tracks}
    for m in moments:  # cached answers (data/vlm/*.json) predate the flag → derive it from the track
        if m["kind"] == "visitor" and "short_visit" not in m and m.get("tid") in by_tid:
            tr = by_tid[m["tid"]]
            real = tr.feat["kpt"] > 0.5 and tr.feat["score"] > 0.6
            m["short_visit"] = bool(real and (tr.t[-1] - tr.t[0]) < 15.0 and tr.t[-1] < scene.times[-1] - 1.0)
    short = {m.get("tid") for m in moments if m["kind"] == "visitor" and m.get("short_visit")}
    for mk, ms in by_kind.items():
        keys = sorted({k for m in ms for k in m["answer"]})
        for k in keys:
            if k not in VLM_TYPES:
                continue
            yes = [m for m in ms if m["answer"].get(k)]
            if k == "client_gives_bag":  # a proper handover (ID, seal check, the courier stays) is normal
                yes = [m for m in yes if m.get("tid") in short]
            if not yes:
                continue
            first = min(yes, key=lambda m: m["t"])
            i0 = int(first["idx"][0])
            out.append(Cand(f"vlm_{k}", float(scene.times[i0]), i0, [int(x) for x in first["idx"]][:3],
                            {"count": float(len(yes)), "frac": len(yes) / len(ms)}, type_id=VLM_TYPES[k]))
    return out
