"""K4 probe (docs/research/killer/K4_unknown_types.md): per-video token strings + per-visit grammar from cached scenes (2 fps, no frames, no OWL/VLM).
usage: .venv/bin/python scripts/probe_grammar.py  (run from repo root; writes scripts/probe_grammar.json)
"""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import pickle
from common import CACHE, load_labels  # noqa
from app.ml.events import analyze_scene  # noqa
from app.ml.moments import visit_runs, safe_zone  # noqa
from app.ml.candidates import hand_zone, entry_index  # noqa

labels = load_labels()
ann = json.load(open("configs/train_annotations.json"))


def frame_tokens(sc):
    """Per-frame symbols: cashier state, visitor state, hand state."""
    T = len(sc.obs); times = sc.times
    cash = [t for t in sc.tracks if t.role == "cashier"]
    vis = [t for t in sc.tracks if t.role == "visitor"]
    C = ["-"] * T; V = ["."] * T; Hd = [" "] * T
    # cashier
    for tr in cash:
        k_in = entry_index(tr, sc.obs)
        t_ok = tr.t[k_in] + 5.0 if k_in is not None else -1.0
        for k, i in enumerate(tr.idx):
            d = tr.info["d"][k]; rel = tr.h[k] / sc.ref_h
            if d < 0.45 and rel < 1.35:
                s = "S"
            elif d > 0.25 and rel > 1.3 and tr.t[k] > t_ok:
                s = "F"  # standing away from the seat (safe/cabinet) — refined below with velocity
            elif d < 2.2:
                s = "W"
            else:
                s = "-"
            if C[i] == "-" or s == "S":
                C[i] = s
    # standing-but-walking vs standing still: velocity
    for tr in cash:
        c = tr.center
        v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(tr.t).clip(1e-3) / sc.ref_h]
        for k, i in enumerate(tr.idx):
            if C[i] == "F" and v[k] >= 0.35:
                C[i] = "W"
    # visitors
    nin = np.zeros(T, int)
    for tr in vis:
        for k, i in enumerate(tr.idx):
            d = tr.info["d"][k]
            if d < 0.85:
                V[i] = "v" if V[i] != "2" else "2"
            elif d < 2.2 and V[i] == ".":
                V[i] = "n"
            if d < 2.2:
                nin[i] += 1
    for i in range(T):
        if nin[i] >= 2:
            V[i] = "2"
    # hands: contact (cashier wrist near visitor wrist) / work (wrists in hand zone, moving) / idle
    hz = hand_zone(sc, list(np.where(sc.present)[0]))
    wr = {}  # i -> cashier wrists
    for tr in cash:
        for k, i in enumerate(tr.idx):
            w, s = tr.kpts[k, [9, 10]], tr.kps[k, [9, 10]]
            wr.setdefault(i, []).extend(list(w[s > 0.3]))
    vw = {}
    for tr in vis:
        for k, i in enumerate(tr.idx):
            if tr.info["d"][k] < 1.6:
                w, s = tr.kpts[k, [9, 10]], tr.kps[k, [9, 10]]
                vw.setdefault(i, []).extend(list(w[s > 0.3]))
    prev = None
    for i in range(T):
        cw = wr.get(i, [])
        if not cw:
            prev = None; continue
        cw = np.array(cw)
        if i in vw and len(vw[i]):
            dm = min(np.linalg.norm(a - b) for a in cw for b in np.array(vw[i])) / sc.ref_h
            if dm < 0.35:
                Hd[i] = "h"; prev = cw; continue
        if hz is not None:
            inz = [(hz[0] <= p[0] <= hz[2] and hz[1] <= p[1] <= hz[3]) for p in cw]
            if any(inz):
                sp = 0.0
                if prev is not None and len(prev):
                    sp = min(np.linalg.norm(a - b) for a in cw for b in prev) / sc.ref_h / max(times[i] - times[i - 1], 1e-3)
                Hd[i] = "k" if sp > 0.12 else "q"
        prev = cw
    return C, V, Hd, nin


def episodes(sym, times, min_len=0.0):
    """Run-length episodes [(symbol, t0, t1)]."""
    out = []
    i = 0
    while i < len(sym):
        j = i
        while j + 1 < len(sym) and sym[j + 1] == sym[i]:
            j += 1
        if times[j] - times[i] + 0.5 >= min_len:
            out.append((sym[i], float(times[i]), float(times[j]) + 0.5))
        i = j + 1
    return out


def visit_grammar(sc, C, V, Hd, nin, t0, t1):
    """Event-level symbols inside [t0-2, t1+6]: IN, CONTACT, WORK, SAFE, CROWD, C_AWAY, OUT."""
    times = sc.times
    m = (times >= t0 - 2.0) & (times <= t1 + 6.0)
    idx = np.where(m)[0]
    ev = []
    # contacts: runs of h >= 1 frame; work: runs of k >= 2 frames (1 s); safe: runs of F >= 2 frames; away: C=='-' or W far
    def runs(symbols, target, min_frames):
        r = []; k = 0
        while k < len(idx):
            if symbols[idx[k]] == target:
                e = k
                while e + 1 < len(idx) and symbols[idx[e + 1]] == target:
                    e += 1
                if e - k + 1 >= min_frames:
                    r.append((float(times[idx[k]]), float(times[idx[e]])))
                k = e + 1
            else:
                k += 1
        return r
    for a, b in runs(Hd, "h", 1): ev.append((a, "CONTACT"))
    for a, b in runs(Hd, "k", 3): ev.append((a, "WORK"))
    for a, b in runs(C, "F", 2): ev.append((a, "SAFE"))
    for a, b in runs(C, "-", 3):
        if a > t0: ev.append((a, "C_AWAY"))
    for a, b in runs(V, "2", 3): ev.append((a, "CROWD"))
    ev.append((t0, "IN")); ev.append((t1, "OUT"))
    ev.sort()
    seq = [s for _, s in ev]
    # merge consecutive duplicates
    out = [seq[0]]
    for s in seq[1:]:
        if s != out[-1]:
            out.append(s)
    return out, ev


def lev(a, b):
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    d[:, 0] = range(len(a) + 1); d[0, :] = range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[-1, -1])


rows = []
names = sorted(n for n in labels if n in ann)
for name in names:
    d = pickle.load(open(os.path.join(CACHE, f"{name}.pkl"), "rb"))
    sc = analyze_scene(d["obs"], d["meta"]["width"], d["meta"]["height"])
    if sc.seat is None:
        print(name, "NO CASHIER"); continue
    C, V, Hd, nin = frame_tokens(sc)
    lab = labels[name]
    a = ann[name]
    print(f"\n=== {name}  {a['label']}  layout={a['layout']} cab={lab.get('cabinet')}  types={[t for _, t in lab['types']]} ts={lab['ts']}  dur={sc.times[-1]:.0f}s")
    # compact timeline per 1 s
    step = 2
    print("  t  : " + "".join(f"{int(sc.times[i]) % 10}" for i in range(0, len(C), step)))
    print("  C  : " + "".join(C[i] for i in range(0, len(C), step)))
    print("  V  : " + "".join(V[i] for i in range(0, len(V), step)))
    print("  H  : " + "".join(Hd[i] for i in range(0, len(Hd), step)))
    visits = visit_runs(sc)
    sz = safe_zone(sc)
    vrows = []
    for (t0, t1) in visits:
        seq, ev = visit_grammar(sc, C, V, Hd, nin, t0, t1)
        print(f"  visit {t0:5.1f}–{t1:5.1f}: {' '.join(seq)}")
        vrows.append({"t0": t0, "t1": t1, "seq": seq})
    # cashier-alone session grammar (whole video, coarse): SEATED/SAFE/AWAY episodes
    ep = episodes(C, sc.times, min_len=1.0)
    sess = [s for s, _, _ in ep]
    sess2 = [sess[0]] + [s for p, s in zip(sess, sess[1:]) if s != p] if sess else []
    print(f"  session: {' '.join(sess2)}   safe_zone={'yes' if sz is not None else 'no'}  n_visits={len(visits)}")
    rows.append({"video": name, "label": a["label"], "layout": a["layout"], "cab": lab.get("cabinet"),
                 "types": [t for _, t in lab["types"]], "visits": vrows, "session": sess2, "dur": float(sc.times[-1])})

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_grammar.json"), "w"), ensure_ascii=False, indent=1)

# ---- variability of normal visits
print("\n\n######## GRAMMAR VARIABILITY")
normal = [(r["video"], v["seq"]) for r in rows if r["label"] == "no_violation" for v in r["visits"]]
viol = [(r["video"], v["seq"], r["types"]) for r in rows if r["label"] == "violation" for v in r["visits"]]
print(f"normal visits: {len(normal)} from {len({n for n,_ in normal})} videos; violation visits: {len(viol)}")
from collections import Counter
cnt = Counter(" ".join(s) for _, s in normal)
for s, c in cnt.most_common():
    print(f"  {c:2d}x  {s}")
# leave-one-video-out nearest-neighbour edit distance
def nn_dist(video, seq, pool):
    ds = [lev(seq, s) / max(len(seq), len(s)) for n, s in pool if n != video]
    return min(ds) if ds else None
print("\nnormal visits: LOVO nearest-normal normalised edit distance")
for n, s in normal:
    print(f"  {n:5s} d={nn_dist(n, s, normal):.2f}  {' '.join(s)}")
print("\nviolation visits: nearest-normal normalised edit distance")
for n, s, ty in viol:
    print(f"  {n:5s} d={nn_dist(n, s, normal):.2f}  {ty}  {' '.join(s)}")
# simple template checks
print("\ntemplate checks per visit (normal first):")
def checks(seq):
    i_in = seq.index("IN") if "IN" in seq else 0
    i_out = seq.index("OUT") if "OUT" in seq else len(seq)
    body = seq[i_in:i_out]
    after = seq[i_out:]
    return {"n_contact": body.count("CONTACT"), "work": "WORK" in body, "safe_before_out": "SAFE" in body,
            "safe_after_out": "SAFE" in after, "work_after_out": "WORK" in after, "c_away": "C_AWAY" in body,
            "crowd": "CROWD" in body, "contact_before_work": ("WORK" in body and body.index("WORK") > 0 and "CONTACT" in body[:body.index("WORK")])}
for n, s in normal:
    print(f"  N {n:5s} {checks(s)}")
for n, s, ty in viol:
    print(f"  V {n:5s} {ty} {checks(s)}")
