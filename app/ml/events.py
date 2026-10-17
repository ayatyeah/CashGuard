"""Camera-agnostic scene analysis on top of per-frame perception.

Everything is expressed relative to the cashier's seat (longest stationary person position) and to the
cashier's apparent height, so it does not depend on where the camera is mounted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perception import COCO, FrameObs
from .tracking import Tracker

C = {n: i for i, n in enumerate(COCO)}
SMALL_ITEMS = [C[n] for n in ("cup", "bottle", "wine glass", "bowl", "cell phone", "book", "remote", "handbag",
                              "backpack", "suitcase", "banana", "apple", "sandwich", "donut", "cake", "scissors")]
DRINK = [C[n] for n in ("cup", "bottle", "wine glass", "bowl")]
BAGS = [C[n] for n in ("handbag", "backpack", "suitcase")]


@dataclass
class PTrack:
    tid: int
    idx: np.ndarray        # sampled-frame indices
    t: np.ndarray          # seconds
    boxes: np.ndarray      # (n,4)
    kpts: np.ndarray       # (n,17,2)
    kps: np.ndarray        # (n,17)
    role: str = "other"    # cashier | visitor | outside

    @property
    def foot(self):
        return np.stack([(self.boxes[:, 0] + self.boxes[:, 2]) / 2, self.boxes[:, 3]], 1)

    @property
    def center(self):
        return (self.boxes[:, :2] + self.boxes[:, 2:]) / 2

    @property
    def h(self):
        return self.boxes[:, 3] - self.boxes[:, 1]


@dataclass
class Scene:
    obs: list
    W: int
    H: int
    tracks: list
    seat: np.ndarray | None          # (x,y) centre of cashier at work
    ref_h: float                     # cashier height at seat, px
    cashier_ids: set
    present: np.ndarray              # (T,) cashier at workplace
    n_inside: np.ndarray             # (T,) persons inside room (cashier + visitors)
    times: np.ndarray
    info: dict = field(default_factory=dict)


def build_tracks(obs: list[FrameObs], W: int, H: int, min_score: float = 0.35) -> list[PTrack]:
    tr = Tracker(max_miss=4)
    diag = float(np.hypot(W, H))
    per_frame = []
    for i, o in enumerate(obs):
        pidx = np.where(o.classes == 0)[0]
        keep = o.scores[pidx] > min_score
        sel = pidx[keep]
        sel_kp = np.where(keep)[0]
        dup = np.zeros(len(sel), bool)
        for a in range(len(sel)):
            for b in range(len(sel)):
                if a != b and o.scores[sel[b]] > o.scores[sel[a]]:
                    A, B = o.boxes[sel[a]], o.boxes[sel[b]]
                    iw = max(0.0, min(A[2], B[2]) - max(A[0], B[0])); ih = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
                    if iw * ih > 0.85 * (A[2] - A[0]) * (A[3] - A[1]):
                        dup[a] = True
        sel, sel_kp = sel[~dup], sel_kp[~dup]
        ids = tr.step(i, o.boxes[sel], o.scores[sel], diag)
        # kpts are stored in the order of person detections
        per_frame.append((sel, sel_kp, ids))
    tracks = []
    for t in tr.finish():
        if len(t.obs) < 2:
            continue
        idx, boxes, kp, ks = [], [], [], []
        for fi, j, box, _ in t.obs:
            sel, kp_rows, _ = per_frame[fi]
            o = obs[fi]
            idx.append(fi); boxes.append(box)
            r = kp_rows[j]
            if r < len(o.kpts):
                kp.append(o.kpts[r]); ks.append(o.kpt_scores[r])
            else:
                kp.append(np.zeros((17, 2), np.float32)); ks.append(np.zeros(17, np.float32))
        idx = np.array(idx)
        tracks.append(PTrack(t.tid, idx, np.array([obs[i].t for i in idx]), np.array(boxes), np.array(kp), np.array(ks)))
    return tracks


def _anchor(tr: PTrack) -> np.ndarray:
    """Hip midpoint when visible (arms move while working, hips do not), else box centre."""
    c = tr.center.copy()
    ok = (tr.kps[:, [11, 12]] > 0.4).all(1)
    if ok.mean() < 0.6:
        return c
    hips = tr.kpts[:, [11, 12]].mean(1)
    last = hips[np.argmax(ok)]
    for i in range(len(c)):  # hold last visible hip position through short occlusions
        if ok[i]:
            last = hips[i]
        c[i] = last
    return c


def _stationary_segments(tr: PTrack, ref: float, max_speed: float = 0.25, min_len: float = 1.5):
    """Segments where the track anchor moves < max_speed * ref px/s."""
    c = _anchor(tr)
    if len(c) < 2:
        return []
    dt = np.diff(tr.t).clip(1e-3)
    v = np.linalg.norm(np.diff(c, axis=0), axis=1) / dt / max(ref, 1)
    slow = np.r_[v[0], v] < max_speed
    segs, s = [], None
    for i, f in enumerate(slow):
        if f and s is None:
            s = i
        if (not f or i == len(slow) - 1) and s is not None:
            e = i if f else i - 1
            if tr.t[e] - tr.t[s] >= min_len:
                segs.append((s, e))
            s = None
    return segs


TRACK_FEATS = ["dmin", "dmed", "dend", "relh_med", "relh_min", "relh_max", "score", "kpt", "dur", "still",
               "foot_rel", "x_rel", "start_rel", "end_rel", "frac_close"]


def track_features(tr: PTrack, obs, seat, seat_box, ref_h, dur_video) -> dict:
    d = np.linalg.norm(tr.center - seat, axis=1) / ref_h
    sc = []
    for i, b in zip(tr.idx, tr.boxes):
        o = obs[i]
        j = int(np.argmin(np.abs(o.boxes - b).sum(1))) if len(o.boxes) else -1
        sc.append(float(o.scores[j]) if j >= 0 else 0.0)
    rel = tr.h / ref_h
    if len(tr.t) > 1:
        v = np.linalg.norm(np.diff(tr.center, axis=0), axis=1) / np.diff(tr.t).clip(1e-3) / ref_h
        still = float((v < 0.25).mean())
    else:
        still = 0.0
    return {
        "dmin": float(d.min()), "dmed": float(np.median(d)), "dend": float(d[-1]),
        "relh_med": float(np.median(rel)), "relh_min": float(rel.min()), "relh_max": float(rel.max()),
        "score": float(np.mean(sc)), "kpt": float(tr.kps.mean()), "dur": float(tr.t[-1] - tr.t[0]), "still": still,
        "foot_rel": float((np.median(tr.boxes[:, 3]) - seat_box[3]) / ref_h),
        "x_rel": float(abs(np.median(tr.center[:, 0]) - seat[0]) / ref_h),
        "start_rel": float(tr.t[0] / max(dur_video, 1)), "end_rel": float(tr.t[-1] / max(dur_video, 1)),
        "frac_close": float((d < 1.2).mean()),
    }


def visitor_prob(f: dict, model: dict | None) -> float:
    """Trained logistic model (scripts/train.py → models/track_model.json); heuristic fallback."""
    if model:
        x = (np.array([f[k] for k in model["feats"]]) - np.array(model["mean"])) / np.array(model["std"])
        z = float(x @ np.array(model["coef"]) + model["intercept"])
        p = 1.0 / (1.0 + np.exp(-z))
        # people seen through the glass wall or in reflections are detected with low confidence
        return p if f["score"] > 0.6 else min(p, 0.3)
    return 1.0 if (f["dmin"] < 1.4 and f["relh_med"] > 0.45 and f["dur"] >= 2.0 and f["still"] > 0.4) else 0.0


def load_json_model(name: str) -> dict | None:
    import json
    import os
    from .perception import MODELS_DIR
    p = os.path.join(MODELS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def analyze_scene(obs: list[FrameObs], W: int, H: int, track_model: dict | None | str = "default") -> Scene:
    if track_model == "default":
        track_model = load_json_model("track_model.json")
    tracks = build_tracks(obs, W, H)
    times = np.array([o.t for o in obs])
    T = len(obs)
    if not tracks:
        return Scene(obs, W, H, [], None, H * 0.2, set(), np.zeros(T, bool), np.zeros(T, int), times)

    # --- cashier: present for most of the video, mostly still, large (in the room, near the camera)
    T_video = max(float(times[-1]), 1.0)
    best = None
    for tr in tracks:
        presence = tr.t[-1] - tr.t[0]
        segs = _stationary_segments(tr, float(np.median(tr.h)))
        still = sum(tr.t[e] - tr.t[s] for s, e in segs) / max(presence, 1e-3)
        score = presence * (0.5 + 0.5 * min(still, 1.0)) * (np.median(tr.h) / H) ** 0.5
        if best is None or score > best[0]:
            best = (score, tr, segs)
    _, ctr, segs = best
    # --- seat: the cashier's longest *seated* stationary episode (standing at the safe is down-weighted)
    p10, p90 = np.percentile(ctr.h, 10), np.percentile(ctr.h, 90)
    stands = p90 > 1.5 * p10
    s, e, bs = 0, len(ctr.t) - 1, -1.0
    for a, b in segs:
        hs = float(np.median(ctr.h[a:b + 1]))
        seated = (not stands) or hs < 0.8 * p90
        sc_ = (ctr.t[b] - ctr.t[a]) * (1.0 if seated else 0.2)
        if sc_ > bs:
            s, e, bs = a, b, sc_
    seat = np.median(ctr.center[s:e + 1], axis=0)
    ref_h = float(np.median(ctr.h[s:e + 1]))

    # --- roles
    def dist(tr):
        return np.linalg.norm(tr.center - seat, axis=1) / ref_h

    cashier_ids = set()
    seat_box = np.median(ctr.boxes[s:e + 1], axis=0)
    for tr in tracks:
        d = dist(tr)
        tr.info = {"d": d}
        tr.feat = track_features(tr, obs, seat, seat_box, ref_h, times[-1])
        if tr.tid == ctr.tid:
            tr.role = "cashier"
            cashier_ids.add(tr.tid)
    # fragments of the cashier: overlap the seat in time-gaps of the main cashier track and stay near it
    main_t = set(np.round(ctr.t, 2))
    for tr in tracks:
        if tr.role == "cashier":
            continue
        overlap = np.mean([round(x, 2) in main_t for x in tr.t])
        if overlap < 0.2 and tr.info["d"].mean() < 0.7 and tr.feat["relh_med"] > 0.6:
            tr.role = "cashier"
            cashier_ids.add(tr.tid)
    for tr in tracks:
        if tr.role == "cashier":
            continue
        p = visitor_prob(tr.feat, track_model)
        tr.info["p_visitor"] = p
        tr.role = "visitor" if p >= 0.5 else "outside"

    present = np.zeros(T, bool)
    n_inside = np.zeros(T, int)
    for tr in tracks:
        if tr.role == "cashier":
            d = tr.info["d"]
            present[tr.idx[d < 1.0]] = True
        if tr.role in ("cashier", "visitor"):
            inside = tr.info["d"] < 2.2
            np.add.at(n_inside, tr.idx[inside], 1)
    # bridge 1-frame gaps
    for i in range(1, T - 1):
        if not present[i] and present[i - 1] and present[i + 1]:
            present[i] = True
    return Scene(obs, W, H, tracks, seat, ref_h, cashier_ids, present, n_inside, times,
                 {"cashier_tid": ctr.tid})
