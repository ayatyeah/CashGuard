"""Post-perception pipeline shared by the runtime analyzer and offline evaluation:
scene → candidates → trained scorer → deduplicated events with descriptions and norm references.
"""
from __future__ import annotations

import os
import time
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


def uses_vlm(models) -> bool:
    return any(k.startswith("vlm_") and st.get("enabled", True)
               for k, st in (models.scorer or {}).get("stumps", {}).items())


class Models:
    def __init__(self):
        self.track = load_json_model("track_model.json")
        self.scorer = load_json_model("scorer.json") or {"kinds": {}}
        self.norms = yaml.safe_load(open(NORMS_PATH)) if os.path.exists(NORMS_PATH) else {}


def run(obs, frames, W, H, models: Models, clip=None, clip_wa: np.ndarray | None = None,
        vlm=None, get_frame=None, vlm_budget: float | None = None, vlm_moments: list | None = None,
        truncated: bool = False, owl=None, owl_cache: dict | None = None,
        inference_budget: float | None = None) -> tuple[list[Event], dict]:
    """obs: FrameObs list; frames: aligned BGR frames (any scale); clip: ClipEmbedder (optional);
    vlm + get_frame(obs_index → full-res BGR): optional VLM verification of key moments (within vlm_budget s);
    vlm_moments: precomputed answered moments (offline evaluation)."""
    from . import moments as M
    scene = analyze_scene(obs, W, H, track_model=models.track)
    probes = probe_scores(clip_wa, clip.probe_bank()) if (clip is not None and clip_wa is not None and len(clip_wa)) else {}
    cands = generate(scene, frames, probes)
    if truncated:  # "never came back" cannot be established on a cut video
        cands = [c for c in cands if c.kind != "leave"]
    owl_started = time.monotonic()
    owl_skipped = 0
    owl_cost = 12.0
    for c in cands:  # optional open-vocabulary check of the desk between visits
        if c.kind != "next_visit":
            continue
        key = round(c.t, 1)
        if owl_cache is not None and key in owl_cache:
            c.f["owl_item"] = float(owl_cache[key])
        elif owl is not None and get_frame is not None and c.extra.get("desk_box") is not None:
            from .candidates import _persons
            values = []
            complete = True
            for idxs in (c.extra["gap_idx"], c.extra["pre_idx"]):
                scores = []
                for i in idxs:
                    if inference_budget is not None and time.monotonic() - owl_started + owl_cost > inference_budget:
                        complete = False
                        break
                    started = time.monotonic()
                    scores.append(owl.item_score(get_frame(i), c.extra["desk_box"], _persons(obs[i], 0.3)))
                    owl_cost = max(owl_cost, 1.25 * (time.monotonic() - started))
                if not complete:
                    break
                values.append(max(scores, default=0.0))
            if complete:
                c.f["owl_item"] = max(0.0, values[0] - values[1])
            else:
                owl_skipped += 1  # never score a partially checked window
    if vlm_budget is not None:
        vlm_budget = max(0.0, vlm_budget - (time.monotonic() - owl_started))
    moms = vlm_moments
    if moms is None and vlm is not None and get_frame is not None and uses_vlm(models) and \
            (vlm_budget is None or vlm_budget > 30):
        import time as _time
        t_v = _time.time()
        moms = M.propose(scene, frames, [c for c in cands if c.kind == "leave"])
        M.ask_all(vlm, moms, get_frame, budget_sec=vlm_budget)
        # self-consistency ensemble: positives must survive 2 of 3 perturbed re-asks
        left = None if vlm_budget is None else max(0.0, vlm_budget - (_time.time() - t_v))
        M.confirm_positives(vlm, moms, get_frame, W, H, len(obs), budget_sec=left)
    if moms:
        cands += M.to_candidates(scene, moms)
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
    kept = sorted(kept, key=lambda e: -e.p)[:3]
    kept.sort(key=lambda e: e.t)
    info = {"owl_skipped": owl_skipped, "n_candidates": len(cands), "cashier_found": scene.seat is not None,
            "vlm_moments": len(moms or []), "vlm_skipped": sum(1 for m in (moms or []) if m.get("skipped")),
            "candidates": [(c.kind, round(c.t, 1), {k: round(float(v), 3) for k, v in c.f.items()}) for c in cands]}
    return kept, info


TEMPLATES = {
    "cashier_left_client_waiting": "Около {t:.0f} с кассир встаёт и уходит с рабочего места, пока посетитель остаётся у стола "
                                   "— обслуживание клиента не завершено.",
    "cash_payout_not_counted": "Около {t:.0f} с кассир передаёт наличные клиенту; непосредственно перед передачей эпизода "
                               "пересчёта (на счётной машинке или вручную) в присутствии клиента не видно.",
    "cash_returned_to_client": "Около {t:.0f} с после пересчёта кассир передаёт пачку наличных обратно клиенту; в сейф "
                               "деньги не убраны.",
    "collection_bag_emptied_no_count": "Около {t:.0f} с содержимое инкассаторской сумки перекладывается в сейф; между "
                                       "вскрытием сумки и закладкой в сейф пересчёта (машинка/вручную) не видно.",
    "cash_to_third_party": "Около {t:.0f} с наличные передаются лицу, не являющемуся обслуживаемым клиентом, либо убираются "
                           "вне запираемого хранилища.",
    "cash_left_on_desk": "Около {t:.0f} с в помещение входит следующий посетитель, а стол кассира не вернулся к виду до "
                         "предыдущей операции: на нём остались предметы (по виду — наличные предыдущего клиента), "
                         "в сейф они не убраны.",
    "safe_left_open": "Около {t:.0f} с сейф остаётся открытым, кассир отошла от него.",
    "safe_key_left": "Около {t:.0f} с кассир уходит, ключ остаётся в замке сейфа.",
    "personal_items": "Около {t:.0f} с на кассовом столе видны личные вещи (сумка/верхняя одежда) рядом с "
                      "кассовым оборудованием (подтверждено визуальной языковой моделью).",
    "collection_bag_no_check": "Около {t:.0f} с посетитель передаёт кассиру сумку (по виду — инкассаторскую); "
                               "осмотра целостности и пломбы сумки не видно.",
    "cash_not_counted": "Около {t:.0f} с на столе кассира видны пачки наличных, сложенные вместе, без пересчёта "
                        "на счётной машинке.",
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
}


def describe(e: Event, norms: dict) -> str:
    vals = {"t": e.t, **{k: float(v) for k, v in e.f.items()}}
    for k in ("screen", "dur", "dwell", "persist"):
        vals.setdefault(k, 0.0)
    obs_text = TEMPLATES.get(e.type_id, "Около {t:.0f} с — признаки возможного нарушения, требуется проверка.").format(**vals)
    n = norms.get(e.type_id) or {}
    text = f"Наблюдается: {obs_text} Квалификация: {n.get('title', e.type_id)}."
    if n.get("source_verified") and n.get("document"):
        rel = " (связанное требование)" if n.get("interpretive") else ""
        text += f" Норма{rel}: {n['document']}, {n['clause']}."
        if n.get("quote"):
            q = n["quote"] if len(n["quote"]) <= 220 else n["quote"][:217].rstrip() + "…"
            text += f" Цитата: «{q}»"
    return text + " Результат — подсказка для проверки человеком, не юридическое заключение."


def title(type_id: str, norms: dict) -> str:
    return (norms.get(type_id) or {}).get("title", type_id)
