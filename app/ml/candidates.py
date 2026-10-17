"""Candidate events + feature vectors. A trained classifier (scorer.py) turns features into violation probability.

Candidate kinds (→ default type_id from app/rules/norms.yaml):
  leave        cashier leaves the workplace and does not come back  → left_cash_unattended / workstation_unlocked
  entry        a person enters; door state after the entry          → door_left_open
  crowd        extra person in the room during work                 → unauthorized_person
  short_visit  visitor leaves shortly after approaching the desk    → count_not_in_client_presence
  drink        cashier drinks / cup near cash                       → food_drink
  phone        smartphone in the cashier's hands                    → phone_use
  bag          bag/clothes on the cash desk                         → personal_items
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import cv2
import numpy as np

from .events import BAGS, DRINK, Scene, C, _anchor
from .perception import COCO

KINDS = ["leave", "entry", "crowd", "short_visit", "drink", "phone", "bag", "next_visit", "safe", "safe_key"]
DEFAULT_TYPE = {
    "leave": "left_cash_unattended", "entry": "door_left_open", "crowd": "unauthorized_person",
    "short_visit": "count_not_in_client_presence", "drink": "food_drink", "phone": "phone_use", "bag": "personal_items",
    "next_visit": "cash_left_on_desk", "safe": "safe_left_open", "safe_key": "safe_key_left",
}
FEATS = ["rem", "max_d", "end_gap", "end_score", "screen", "items", "desk_probe", "light_drop", "door_change",
         "n_max", "dur", "dwell", "standing", "prob", "count", "near_hand", "persist", "clip", "door_min", "tidy", "unsecured", "desk_change", "desk_blob", "safe_change",
         "persistent_change", "zncc_change", "small_residual", "client_remains", "owl_item", "owl_key"]


@dataclass
class Cand:
    kind: str
    t: float                       # event time (s)
    i: int                         # index into scene.obs
    evidence: list                 # indices into scene.obs (1-3)
    f: dict = field(default_factory=dict)
    type_id: str = ""
    extra: dict = field(default_factory=dict)

    def vec(self) -> np.ndarray:
        oh = [1.0 if self.kind == k else 0.0 for k in KINDS]
        return np.array(oh + [float(self.f.get(k, 0.0)) for k in FEATS], np.float32)


def _idx(scene: Scene, t: float) -> int:
    return int(np.argmin(np.abs(scene.times - t)))


def _overlap(box, boxes) -> bool:
    if len(boxes) == 0:
        return False
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return bool((inter / ((box[2] - box[0]) * (box[3] - box[1]) + 1e-6) > 0.15).any())


def _persons(o, thr=0.3):
    return o.boxes[(o.classes == 0) & (o.scores > thr)]


def screen_level(frame: np.ndarray, o, scale: float) -> float:
    """Relative brightness of the upper part of the most confident laptop box (screen), 0 if no free laptop."""
    m = (o.classes == C["laptop"]) & (o.scores > 0.5)
    if not m.any():
        return 0.0
    b = o.boxes[m][np.argmax(o.scores[m])]
    if _overlap(b, _persons(o)):
        return float("nan")
    x1, y1, x2, y2 = (b * scale).astype(int)
    roi = frame[y1:y1 + max(2, int((y2 - y1) * 0.6)), x1:x2]
    if roi.size == 0:
        return 0.0
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    scene_v = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].mean()) + 1.0
    return float(np.percentile(v, 75) / scene_v)


def desk_items(o, desk) -> int:
    """Small movable objects (crop detector) on the cashier's desk that are not held by a person."""
    cd = o.extra.get("crop_det")
    if cd is None or desk is None:
        return 0
    cb, cs, cc = cd
    pb = _persons(o)
    n = 0
    for b, s, c in zip(cb, cs, cc):
        name = COCO[c]
        if name in ("cell phone", "book", "remote", "handbag", "backpack", "suitcase", "cup", "bottle", "scissors") and s > 0.2:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if desk[0] <= cx <= desk[2] and desk[1] <= cy <= desk[3] and not _overlap(b, pb):
                n += 1
    return n


def static_regions(scene: Scene, seated_idx: list[int]):
    """Desk region (cashier at work + laptop, expanded) and laptop box, estimated while the cashier sits."""
    if not seated_idx:
        return None, None
    lap, cash = [], []
    cids = scene.cashier_ids
    for tr in scene.tracks:
        if tr.tid in cids:
            for i, b in zip(tr.idx, tr.boxes):
                if i in seated_idx:
                    cash.append(b)
    for i in seated_idx:
        o = scene.obs[i]
        m = (o.classes == C["laptop"]) & (o.scores > 0.5)
        if m.any():
            lap.append(o.boxes[m][np.argmax(o.scores[m])])
    laptop = np.median(lap, axis=0) if len(lap) >= 2 else None
    if not cash:
        return None, laptop
    cb = np.median(cash, axis=0)
    boxes = [cb] + ([laptop] if laptop is not None else [])
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    desk = np.array([max(0, x1 - 0.3 * w), max(0, y1 + 0.2 * h), min(scene.W, x2 + 0.3 * w),
                     min(scene.H, y2 + 0.35 * h)], np.float32)
    return desk, laptop


def hand_zone(scene: Scene, seated_idx: list[int], pad: float = 0.35):
    """Where the seated cashier's hands are (= the desk surface she works on), independent of the camera side."""
    pts = []
    sidx = set(seated_idx)
    for tr in scene.tracks:
        if tr.tid not in scene.cashier_ids:
            continue
        for k, i in enumerate(tr.idx):
            if i in sidx:
                w, s = tr.kpts[k, [9, 10]], tr.kps[k, [9, 10]]
                pts += list(w[s > 0.3])
    if len(pts) < 5:
        return None
    p = np.array(pts)
    x1, y1 = np.percentile(p, 5, axis=0); x2, y2 = np.percentile(p, 95, axis=0)
    r = pad * scene.ref_h
    return np.array([max(0, x1 - r), max(0, y1 - r), min(scene.W, x2 + r), min(scene.H, y2 + r)], np.float32)


def fixed_screen_level(frame: np.ndarray, o, laptop, scale: float) -> float:
    """Brightness of the (static) laptop screen area relative to the whole frame; NaN if occluded."""
    if laptop is None:
        return 0.0
    if _overlap(laptop, _persons(o)):
        return float("nan")
    x1, y1, x2, y2 = (laptop * scale).astype(int)
    roi = frame[max(0, y1):max(0, y1) + max(2, int((y2 - y1) * 0.6)), max(0, x1):x2]
    if roi.size == 0:
        return 0.0
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    scene_v = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].mean()) + 1.0
    return float(np.percentile(v, 75) / scene_v)


def crop_scaled(frame, box, scale):
    x1, y1, x2, y2 = (np.asarray(box) * scale).astype(int)
    c = frame[max(0, y1):y2, max(0, x1):x2]
    return c if c.size else frame


def change_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray, normalize_contrast: bool = False) -> float:
    """Fraction of masked pixels that changed between two gray frames (illumination-normalised;
    with normalize_contrast also contrast-normalised — for scenes where the lights may be switched off)."""
    if mask.sum() < 50:
        return float("nan")
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32); b = cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32)
    a = a - a[mask].mean(); b = b - b[mask].mean()
    if normalize_contrast:
        sa, sb = a[mask].std() + 1e-3, b[mask].std() + 1e-3
        s = (sa + sb) / 2
        a, b = a / sa * s, b / sb * s
    d = np.abs(a - b)
    return float((d[mask] > 18).mean())


def new_blob(a: np.ndarray, b: np.ndarray, mask: np.ndarray, ref_px: float) -> float:
    """Largest compact changed blob (e.g. a cash bundle put down) between two gray frames, as area / ref_px²."""
    if mask.sum() < 50:
        return float("nan")
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32); b = cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32)
    a = a - a[mask].mean(); b = b - b[mask].mean()
    ch = ((np.abs(a - b) > 25) & mask).astype(np.uint8)
    ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ch, 8)
    best = 0.0
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < 4 or w > 0.8 * ref_px or h > 0.8 * ref_px:
            continue  # tiny noise or large structures (furniture, people)
        if area / float(w * h) < 0.35:
            continue  # not compact (edges, hair)
        best = max(best, area / ref_px ** 2)
    return best


def person_mask(shape, o, scale, pad=0.15):
    m = np.ones(shape, bool)
    for b in _persons(o, 0.2):
        w, h = b[2] - b[0], b[3] - b[1]
        x1, y1, x2, y2 = ((b + np.array([-w * pad, -h * pad, w * pad, h * pad])) * scale).astype(int)
        m[max(0, y1):y2, max(0, x1):x2] = False
    return m


def entry_index(tr, obs, hi: float = 0.8) -> int | None:
    """Index in the track where the person becomes clearly visible inside (not through the glass), if it entered."""
    sc = []
    for i, b in zip(tr.idx, tr.boxes):
        o = obs[i]
        j = int(np.argmin(np.abs(o.boxes - b).sum(1)))
        sc.append(float(o.scores[j]))
    sc = np.array(sc)
    inside = (sc > hi) & (tr.kps.mean(1) > 0.55)
    ref = float(np.median(tr.h))
    if tr.t[0] < 0.25:
        # visible from the first frame: counts as an entry only if the person walks in during the first 4 s
        m4 = tr.t <= tr.t[0] + 4.0
        disp = np.linalg.norm(tr.center[m4] - tr.center[0], axis=1).max() / max(ref, 1)
        if disp < 0.6:
            return None
    for k in range(len(sc) - 1):
        if inside[k] and inside[k + 1]:
            return k
    return None


def generate(scene: Scene, frames: list[np.ndarray], probes: dict | None = None) -> list[Cand]:
    """frames: sampled frames (BGR, any scale) aligned with scene.obs. probes: CLIP probe scores per frame (optional)."""
    obs, times = scene.obs, scene.times
    if not obs or scene.seat is None:
        return []
    dur = float(times[-1])
    scale = frames[0].shape[1] / scene.W if frames else 1.0
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cands: list[Cand] = []
    cashier = [t for t in scene.tracks if t.role == "cashier"]
    visitors = [t for t in scene.tracks if t.role == "visitor"]
    probes = probes or {}

    # ---------------- leave ----------------
    seated_t = []
    for tr in cashier:
        d = tr.info["d"]
        rel = tr.h / scene.ref_h
        seated_t += list(tr.t[(d < 0.45) & (rel < 1.35)])
    if seated_t:
        t_last = max(seated_t)
        rem = dur - t_last
        if rem >= 3.0:
            # the walk-away starts when the cashier stands up, or after the last long stop (safe, light switch)
            max_d, t_walk, end_t, t_far = 0.0, t_last + 0.5, 0.0, t_last + 0.5
            for tr in cashier:
                m = tr.t > t_last
                if not m.any():
                    continue
                d = tr.info["d"][m]; tt = tr.t[m]
                end_t = max(end_t, float(tt[-1]))
                if d.max() <= max_d:
                    continue
                max_d = float(d.max())
                k_far = int(np.argmax(d >= 0.9 * d.max()))
                c = _anchor(tr)[m][: k_far + 1]
                tw = t_last + 0.5
                if len(c) > 2:
                    v = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(tt[: k_far + 1]).clip(1e-3) / scene.ref_h
                    slow = np.r_[v < 0.3, False]
                    run_start = None
                    for q in range(len(slow)):
                        if slow[q] and run_start is None:
                            run_start = q
                        if not slow[q] and run_start is not None:
                            if tt[q] - tt[run_start] >= 2.0:
                                tw = float(tt[q])
                            run_start = None
                t_walk = max(t_walk, min(tw, float(tt[k_far])))
                t_far = float(tt[k_far])
            i0 = _idx(scene, t_last)
            seated_idx = [i for i in range(len(obs)) if times[i] <= t_last]
            desk, laptop = static_regions(scene, seated_idx)
            # evidence window: after the cashier has walked away (reached the farthest point)
            late = [i for i in range(len(obs)) if times[i] >= max(t_far, t_walk + 1.5)]
            scr = [fixed_screen_level(frames[i], obs[i], laptop, scale) for i in late]
            scr = [s for s in scr if s == s]
            items = [desk_items(obs[i], desk) for i in late]
            b0 = np.mean([o.brightness for o in obs[max(0, i0 - 4):i0 + 1]])
            b1 = np.mean([obs[i].brightness for i in late]) if late else b0
            f = {"rem": rem, "max_d": max_d, "end_gap": dur - end_t, "tidy": float(t_walk - t_last),
                 "end_score": 0.0, "screen": float(np.median(scr)) if scr else 0.0,
                 "items": float(np.mean(items)) if items else 0.0,
                 "desk_probe": 0.0,  # filled by the trained CLIP desk probe (scorer.py)
                 "light_drop": float((b0 - b1) / (b0 + 1))}
            # objects stayed on the desk AND no time was spent securing them (safe) before walking away
            f["unsecured"] = f["items"] / (1.0 + f["tidy"] / 5.0)
            # a client still at the desk after the cashier left (п. 22): visitor within 1.2 ref_h of the seat
            vis_late = 0
            for tr in visitors:
                m = (tr.t >= t_walk + 1.0) & (tr.info["d"] < 1.2)
                vis_late = max(vis_late, int(m.sum()))
            f["client_remains"] = vis_late / 2.0  # seconds at 2 fps
            iw = _idx(scene, t_walk)
            # evidence: start of the walk + the unattended desk afterwards
            free = [i for i in late if desk is None or not _overlap(desk, _persons(obs[i], 0.5))]
            ev_late = (free or late)[len(free or late) // 2] if late else None
            ev = [iw] + ([ev_late] if ev_late is not None else [])
            c = Cand("leave", float(times[iw]), iw, ev, f)
            if desk is not None and late:
                pick = (free or late)
                pick = [pick[k] for k in np.linspace(0, len(pick) - 1, min(4, len(pick))).astype(int)]
                c.extra = {"desk_crops": [crop_scaled(frames[i], desk, scale) for i in pick], "desk_box": desk}
            cands.append(c)

    # ---------------- entry / door ----------------
    for tr in cashier + visitors:
        k = entry_index(tr, obs)
        if k is None:
            continue
        t_in = float(tr.t[k])
        # door ≈ where the person crosses from "behind the glass" to "inside": box at the crossing ±1 s
        m = np.abs(tr.t - t_in) <= 1.0
        bx = tr.boxes[m]
        x1, y1 = bx[:, 0].min(), bx[:, 1].min(); x2, y2 = bx[:, 2].max(), bx[:, 3].max()
        w, h = x2 - x1, y2 - y1
        reg = np.zeros(gray[0].shape, bool)
        X1, Y1, X2, Y2 = (np.array([x1 - 0.5 * w, y1 - 0.15 * h, x2 + 0.5 * w, y2 + 0.05 * h]) * scale).astype(int)
        reg[max(0, Y1):Y2, max(0, X1):X2] = True
        i_before = _idx(scene, max(0.0, t_in - 2.5))
        pre = reg & person_mask(gray[0].shape, obs[i_before], scale)
        vals = []
        for j in range(len(obs)):
            if t_in + 4.0 <= times[j] <= t_in + 12.0:
                mk = pre & person_mask(gray[0].shape, obs[j], scale)
                if mk.sum() > 0.25 * pre.sum():
                    vals.append(change_score(gray[i_before], gray[j], mk))
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        door_min = float(np.min(vals))
        te = t_in + 3.0  # organisers' convention: the moment the person walks away leaving the door open
        ie = _idx(scene, te)
        cands.append(Cand("entry", te, ie, [_idx(scene, t_in + 1.0), ie, _idx(scene, t_in + 7.0)],
                          {"door_change": float(np.median(vals)), "door_min": door_min, "dur": float(tr.t[-1] - tr.t[0]),
                           "standing": 1.0 if tr.role == "cashier" else 0.0}))

    # ---------------- next visit: did the desk return to its state before the previous client's visit? ----
    # (cash of the previous client left on the desk when the next one comes in — the bundle is ~15 px, so a
    # masked appearance change of the desk region is used instead of an object detector). Consecutive clients
    # often sit on the same chair and end up in one track, so visits are runs of "sitting at the desk".
    if visitors:
        seated_idx = [i for i in range(len(obs)) if scene.present[i]]
        _, laptop_box = static_regions(scene, seated_idx)
        desk = hand_zone(scene, seated_idx)
        visits = []
        for tr in visitors:
            near = tr.info["d"] < 0.85
            k = 0
            while k < len(near):
                if not near[k]:
                    k += 1
                    continue
                e = k
                while e + 1 < len(near) and (near[e + 1] or (e + 2 < len(near) and near[e + 2]
                                                             and tr.t[e + 2] - tr.t[e] < 2.0)):
                    e += 1
                if tr.t[e] - tr.t[k] >= 3.0:
                    visits.append((float(tr.t[k]), float(tr.t[e])))
                k = e + 1
        visits.sort()
        if desk is not None and len(visits) >= 2:
            reg = np.zeros(gray[0].shape, bool)
            X1, Y1, X2, Y2 = (desk * scale).astype(int)
            reg[max(0, Y1):Y2, max(0, X1):X2] = True
            if laptop_box is not None:  # the screen changes legitimately (unlock, typing)
                lw, lh = laptop_box[2] - laptop_box[0], laptop_box[3] - laptop_box[1]
                L1, T1, L2, T2 = ((laptop_box + np.array([-0.15 * lw, -0.15 * lh, 0.15 * lw, 0.15 * lh])) * scale).astype(int)
                reg[max(0, T1):T2, max(0, L1):L2] = False
            for (a0, a1), (b0, b1) in zip(visits, visits[1:]):
                if b0 - a1 < 3.0 or a1 - a0 < 8.0:
                    continue
                i_a = _idx(scene, a0 + 1.0)
                gap = [j for j in range(len(obs)) if a1 + 1.0 <= times[j] <= b0 - 0.5]
                vals, blobs = [], []
                proto_targets, proto_masks = [], []
                for j in gap[-4:]:
                    mk = reg & person_mask(gray[0].shape, obs[i_a], scale) & person_mask(gray[0].shape, obs[j], scale)
                    vals.append(change_score(gray[i_a], gray[j], mk))
                    proto_targets.append(gray[j]); proto_masks.append(mk)
                    blobs.append(new_blob(gray[i_a], gray[j], mk, scene.ref_h * scale))
                vals = [v for v in vals if v == v]
                blobs = [v for v in blobs if v == v]
                if vals:
                    te = max(a1, b0 - 3.0)
                    ie = _idx(scene, te)
                    c = Cand("next_visit", te, ie, [ie, gap[-1]],
                             {"desk_change": float(np.median(vals)), "dur": float(a1 - a0),
                              "desk_blob": float(np.median(blobs)) if blobs else 0.0})
                    pre = [j for j in range(len(obs)) if a0 - 2.0 <= times[j] <= a0 + 1.0][:2]
                    c.extra = {"gap_idx": gap[-3:], "pre_idx": pre, "desk_box": desk}
                    cands.append(c)
                    if os.environ.get("FORTE_ZONE_PROTOTYPES") == "1":
                        from .zone_prototypes import residual_features
                        cands[-1].f.update(residual_features(gray[i_a], proto_targets, proto_masks, scene.ref_h * scale))

    # ---------------- safe left open: the safe zone looks different at the end than at the start ----------
    from .moments import safe_zone, _box_overlap
    sz = safe_zone(scene)
    if sz is not None:
        reg = np.zeros(gray[0].shape, bool)
        X1, Y1, X2, Y2 = (sz * scale).astype(int)
        reg[max(0, Y1):Y2, max(0, X1):X2] = True
        standing = []
        for tr in cashier:
            k_in = entry_index(tr, obs)
            t_ok = tr.t[k_in] + 5.0 if k_in is not None else -1.0
            standing += [int(i) for k, i in enumerate(tr.idx)
                         if tr.info["d"][k] > 0.25 and tr.h[k] / scene.ref_h > 1.3 and tr.t[k] > t_ok]
        standing = sorted(set(standing))
        free = [i for i in range(len(obs)) if i not in set(standing)
                and not _box_overlap(sz, _persons(obs[i], 0.3), thr=0.5)]
        if standing and len(free) >= 2:
            t_safe0 = float(times[min(standing)])
            ref_i = [i for i in free if times[i] < t_safe0] or [0]
            i0 = ref_i[-1]
            late = [i for i in free if times[i] > float(times[max(standing)])]
            if late:
                vals = []
                proto_targets, proto_masks = [], []
                for j in late[-4:]:
                    mk = reg & person_mask(gray[0].shape, obs[i0], scale) & person_mask(gray[0].shape, obs[j], scale)
                    vals.append(change_score(gray[i0], gray[j], mk, normalize_contrast=True))
                    proto_targets.append(gray[j]); proto_masks.append(mk)
                vals = [v for v in vals if v == v]
                if vals:
                    # event time: when the cashier walked away from the safe for the last time
                    t_leave = float(times[max(standing)]) + 0.5
                    ie = _idx(scene, t_leave)
                    cands.append(Cand("safe", t_leave, ie, [ie, late[-1]],
                                      {"safe_change": float(np.median(vals)), "dur": float(times[-1] - t_leave)}))
                    if os.environ.get("FORTE_ZONE_PROTOTYPES") == "1":
                        from .zone_prototypes import residual_features
                        cands[-1].f.update(residual_features(gray[i0], proto_targets, proto_masks, scene.ref_h * scale))

    # ---------------- crowd ----------------
    n = scene.n_inside
    run = 0
    for i in range(len(n)):
        run = run + 1 if n[i] >= 3 else 0
        if run == 4:  # ~2 s at 2 fps
            s = i - 3
            j = s
            while j < len(n) and n[j] >= 3:
                j += 1
            cands.append(Cand("crowd", float(times[s]), s, [s, min(j - 1, s + 2)],
                              {"n_max": float(n[s:j].max()), "dur": float(times[j - 1] - times[s])}))
    for tr in visitors:  # visitor who never sits down (clients sit at the desk; outsiders / couriers stand)
        d = tr.info["d"]
        k_in = entry_index(tr, obs)
        k0 = k_in if k_in is not None else 0
        inside = np.arange(len(tr.t)) >= k0
        near = inside & (d < 1.5)
        if near.sum() < 4:
            continue
        p90 = np.percentile(tr.h[near], 90)
        standing = float((tr.h[near] > 0.8 * p90).mean())
        sit_ratio = float(np.percentile(tr.h[near], 10) / max(p90, 1))
        i = _idx(scene, float(tr.t[near][0]) + 1.0)
        cands.append(Cand("crowd", float(tr.t[near][0]) + 1.0, i, [i, int(tr.idx[near][len(tr.idx[near]) // 2])],
                          {"n_max": float(n[tr.idx[near]].max()), "dur": float(tr.t[near][-1] - tr.t[near][0]),
                           "standing": standing, "persist": sit_ratio}))

    # ---------------- short visit ----------------
    for tr in visitors:
        if tr.t[-1] > dur - 1.0 or tr.t[0] < 0.5:
            continue
        dwell = float(tr.t[-1] - tr.t[0])
        d = tr.info["d"]
        real = tr.feat["kpt"] > 0.5 and tr.feat["score"] > 0.6   # not a reflection / fragment
        if 4.0 <= dwell < 20.0 and d.min() < 0.95 and real:
            k = int(np.argmin(d))
            i = int(tr.idx[k])
            te = max(float(tr.t[0]), float(tr.t[-1]) - 3.0)
            cands.append(Cand("short_visit", te, _idx(scene, te), [i, _idx(scene, tr.t[-1])],
                              {"dwell": dwell, "max_d": float(d.min())}))

    # ---------------- drink / phone / bag ----------------
    pr = probes.get("drinking")
    if pr is not None:
        hits = np.where((pr > 0.8) & (n >= 1))[0]
        if len(hits) >= 2:
            i = int(hits[0])
            cands.append(Cand("drink", float(times[i]), i, list(hits[:3].astype(int)),
                              {"clip": float(pr[hits].mean()), "count": float(len(hits))}))
    # phone/cup near cashier wrists, bags on desk
    ph_hits, cup_hits, bag_hits = [], [], []
    for i, o in enumerate(obs):
        cd = o.extra.get("crop_det")
        if cd is None:
            continue
        cb, cs, cc = cd
        wl = [tr.kpts[np.where(tr.idx == i)[0][0], [9, 10]] for tr in cashier if i in set(tr.idx.tolist())]
        ws = [tr.kps[np.where(tr.idx == i)[0][0], [9, 10]] for tr in cashier if i in set(tr.idx.tolist())]
        wrists = np.concatenate(wl)[np.concatenate(ws) > 0.3] if wl else np.zeros((0, 2))
        for b, s, c in zip(cb, cs, cc):
            ctr = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            near = len(wrists) and np.linalg.norm(wrists - ctr, axis=1).min() < 0.35 * scene.ref_h
            if c == C["cell phone"] and s > 0.35 and near:
                ph_hits.append((i, float(s)))
            if c in DRINK and s > 0.15:
                cup_hits.append((i, float(s), bool(near)))
            if c in BAGS and s > 0.3 and not _overlap(b, _persons(o)):
                bag_hits.append((i, float(s)))
    clip_ph, clip_dr = probes.get("phone"), probes.get("drinking")
    if ph_hits:
        idx = sorted(set(i for i, _ in ph_hits))
        span = times[idx[-1]] - times[idx[0]]
        cands.append(Cand("phone", float(times[idx[0]]), idx[0], idx[:3],
                          {"count": float(len(idx)), "persist": float(span), "prob": max(s for _, s in ph_hits),
                           "clip": float(np.max(clip_ph[idx])) if clip_ph is not None else 0.0}))
    if cup_hits:
        idx = sorted(set(i for i, _, _ in cup_hits))
        nh = [i for i, _, nr in cup_hits if nr]
        i0 = nh[0] if nh else idx[0]
        cands.append(Cand("drink", float(times[i0]), i0, ([i0] + idx[:2])[:3],
                          {"count": float(len(idx)), "near_hand": float(len(set(nh))),
                           "prob": max(s for _, s, _ in cup_hits), "persist": float(times[idx[-1]] - times[idx[0]]),
                           "clip": float(np.max(clip_dr[idx])) if clip_dr is not None else 0.0}))
    if bag_hits:
        idx = sorted(set(i for i, _ in bag_hits))
        span = times[idx[-1]] - times[idx[0]]
        cands.append(Cand("bag", float(times[idx[0]]), idx[0], idx[:3],
                          {"count": float(len(idx)), "persist": float(span), "prob": max(s for _, s in bag_hits)}))
    for c in cands:
        c.type_id = DEFAULT_TYPE[c.kind]
        if c.kind == "leave" and c.f.get("screen", 0) > 1.6 and c.f.get("items", 0) < 0.5:
            c.type_id = "workstation_unlocked"
    return cands
