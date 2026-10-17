"""Post-perception pipeline shared by the runtime analyzer and offline evaluation:
scene → candidates → trained scorer → deduplicated events with descriptions and norm references.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from . import scorer as S
from .candidates import generate
from .events import analyze_scene, load_json_model
from .semantic import probe_scores

NORMS_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "norms.yaml")


@dataclass
class Event:
    type_id: str
    t: float
    i: int                 # index into obs
    evidence: list         # indices into obs
    p: float
    kind: str
    f: dict


class Models:
    def __init__(self):
        self.track = load_json_model("track_model.json")
        self.scorer = load_json_model("scorer.json") or {"kinds": {}}
        self.norms = yaml.safe_load(open(NORMS_PATH)) if os.path.exists(NORMS_PATH) else {}


def run(obs, frames, W, H, models: Models, clip=None, clip_wa: np.ndarray | None = None) -> tuple[list[Event], dict]:
    """obs: FrameObs list; frames: aligned BGR frames (any scale); clip: ClipEmbedder (optional)."""
    scene = analyze_scene(obs, W, H, track_model=models.track)
    probes = probe_scores(clip_wa, clip.probe_bank()) if (clip is not None and clip_wa is not None and len(clip_wa)) else {}
    cands = generate(scene, frames, probes)
    events: list[Event] = []
    for c in cands:
        for type_id, p in S.score(c, models.scorer):
            events.append(Event(type_id, c.t, c.i, list(dict.fromkeys(c.evidence))[:3], float(p), c.kind, dict(c.f)))
    # dedupe: same type within 6 s → keep the most confident
    events.sort(key=lambda e: -e.p)
    kept: list[Event] = []
    for e in events:
        if any(k.type_id == e.type_id and abs(k.t - e.t) < 6.0 for k in kept):
            continue
        kept.append(e)
    kept.sort(key=lambda e: e.t)
    info = {"n_candidates": len(cands), "cashier_found": scene.seat is not None,
            "candidates": [(c.kind, round(c.t, 1), {k: round(float(v), 3) for k, v in c.f.items()}) for c in cands]}
    return kept, info


TEMPLATES = {
    "left_cash_unattended": "Около {t:.0f} с кассир встаёт и уходит с рабочего места, не вернувшись до конца записи. "
                            "После ухода на столе кассира остаются предметы (по виду — наличные/ценности/документы); "
                            "перед уходом они не убраны в запираемое хранилище.",
    "workstation_unlocked": "Около {t:.0f} с кассир покидает рабочее место; после ухода экран ноутбука остаётся включённым "
                            "(экран в {screen:.1f} раза ярче среднего по кадру) — рабочая станция, вероятно, не заблокирована.",
    "door_left_open": "Около {t:.0f} с человек входит в помещение кассы; в течение следующих 4–10 с область двери "
                      "не возвращается к исходному (закрытому) виду — дверь, вероятно, оставлена открытой.",
    "unauthorized_person": "Около {t:.0f} с в помещении кассы находится посетитель, который не садится на клиентское место "
                           "и около {dur:.0f} с стоит рядом с рабочим местом кассира.",
    "count_not_in_client_presence": "Посетитель подходит к столу кассира и покидает помещение примерно через {dwell:.0f} с "
                                    "(около {t:.0f} с), до завершения операции — пересчёт наличных мог проводиться "
                                    "не в его присутствии.",
    "food_drink": "Около {t:.0f} с у кассира на рабочем месте напиток/ёмкость с едой рядом с наличными; кассир подносит "
                  "его к лицу.",
    "phone_use": "Около {t:.0f} с в руках кассира виден мобильный телефон (около {persist:.0f} с).",
    "personal_items": "Около {t:.0f} с на кассовом столе находятся личные вещи (сумка/одежда).",
}


def describe(e: Event, norms: dict) -> str:
    vals = {"t": e.t, **{k: float(v) for k, v in e.f.items()}}
    for k in ("screen", "dur", "dwell", "persist"):
        vals.setdefault(k, 0.0)
    text = TEMPLATES.get(e.type_id, "Около {t:.0f} с — признаки возможного нарушения, требуется проверка.").format(**vals)
    n = norms.get(e.type_id) or {}
    if n.get("source_verified") and n.get("document"):
        rel = " (связанное требование)" if n.get("interpretive") else ""
        text += f" Норматив{rel}: {n['document']}, {n['clause']}."
    return text + " Результат — подсказка для проверки человеком."


def title(type_id: str, norms: dict) -> str:
    return (norms.get(type_id) or {}).get("title", type_id)
