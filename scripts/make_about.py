"""Build the violation-type catalogue from app/rules/norms.yaml + models/scorer.json (single source of truth).

Outputs:
  docs/violation_types.md              — table for SUBMISSION.md (ТЗ §8 п. 5: types with a short description)
  src/frontend/public/about.html       — public "about the solution" page served by the stand at /about.html
usage: python scripts/make_about.py
"""
from __future__ import annotations

import html
import json
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
URL = "https://fortecashguard.up.railway.app"

# How each type is detected on the production stand (CPU, no VLM). Kept short and literal: only what is measured.
HOW = {
    "left_cash_unattended": "Кассир уходит от рабочего места дальше 0,9 своего роста и не возвращается до конца записи; "
                            "на столе после ухода остаются предметы, которых кассир не убрал.",
    "workstation_unlocked": "После ухода кассира экран ноутбука остаётся светящимся (яркость зоны экрана относительно "
                            "кадра). Блокировку учётной записи по видео установить нельзя — фиксируется только видимое состояние экрана.",
    "door_left_open": "После входа кассира зона двери остаётся изменённой (дверь не закрыта) после того, как он отошёл от неё.",
    "unauthorized_person": "Посетитель стоит у стола, так и не сев (короткое пребывание < 12 с), или в помещении "
                           "одновременно три человека и более.",
    "food_drink": "Несколько кадров рабочей зоны совпадают с описанием «кассир пьёт / ест» (CLIP-проба).",
    "safe_left_open": "Зона сейфа (определяется по тому, где кассир стоит и работает руками) остаётся изменённой "
                      "после того, как кассир отошёл.",
    "safe_key_left": "OWLv2 подтверждает металлический ключ в зоне замка сейфа на двух кадрах после ухода кассира.",
    "cash_left_on_desk": "Между двумя клиентами на столе появляется пачка купюр / документ, которых не было до первого "
                         "клиента (OWLv2 на кропе стола).",
    "count_not_in_client_presence": "Клиент подходит к столу и уходит через 4–20 с, до завершения операции.",
    "cashier_left_client_waiting": "Кассир уходит с места, а клиент остаётся в помещении не менее 4 с.",
    "phone_use": "Телефон в руках кассира: детектор объектов у запястий и CLIP-проба устойчиво на нескольких кадрах.",
    "personal_items": "Сумка или одежда на кассовом столе — подтверждается визуально-языковой моделью (стенд на Mac).",
    "collection_bag_no_check": "Инкассатор передаёт сумку и сразу уходит — подтверждается визуально-языковой моделью "
                               "(на основном стенде такой визит распознаётся как пребывание постороннего).",
    "cash_not_counted": "Наличные разных клиентов смешаны на столе — подтверждается визуально-языковой моделью.",
}
PROD = {"left_cash_unattended", "workstation_unlocked", "door_left_open", "unauthorized_person", "food_drink",
        "safe_left_open", "safe_key_left", "cash_left_on_desk", "count_not_in_client_presence",
        "cashier_left_client_waiting", "phone_use"}
VLM = {"personal_items", "collection_bag_no_check", "cash_not_counted"}
RULE_ONLY = {"cashier_left_client_waiting", "phone_use"}  # conservative rules without training examples


def status(tid: str) -> str:
    if tid in PROD:
        return "распознаётся" + (" (правило, без обучающих примеров)" if tid in RULE_ONLY else "")
    if tid in VLM:
        return "только со стендом VLM (Mac)"
    return "норма подготовлена, детектора нет"


def rows():
    norms = yaml.safe_load(open(os.path.join(ROOT, "app", "rules", "norms.yaml")))
    order = sorted(norms, key=lambda t: (0 if t in PROD else 1 if t in VLM else 2, list(norms).index(t)))
    for tid in order:
        n = norms[tid]
        if not n.get("source_verified"):
            continue  # service entry (unknown_review) — not a violation type
        yield tid, n


def markdown() -> str:
    out = ["| Тип нарушения | Статус | Что видно на видео (как распознаётся) | Норма | Связь |",
           "|---|---|---|---|---|"]
    for tid, n in rows():
        how = HOW.get(tid) or n.get("observable", "")
        rel = "связанное требование" if n.get("interpretive") else "прямая"
        doc = n["document"].split("(")[-1].rstrip(")") if "(" in n["document"] else n["document"]
        out.append(f"| {n['title']} | {status(tid)} | {how} | {doc}, {n['clause']} | {rel} |")
    return "\n".join(out) + "\n"


CSS = """
:root{--bg:#f7f6f4;--fg:#1d1b1a;--muted:#5f5a57;--card:#fff;--line:#e4e0dc;--accent:#8b1d32;--ok:#1f6f43;--warn:#8a5a00;--off:#6b6b6b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161414;--fg:#efeceb;--muted:#a9a3a0;--card:#211e1e;--line:#3a3534;--accent:#e0788c;--ok:#6fcf97;--warn:#f2c06b;--off:#9a9a9a}}
:root[data-theme="dark"]{--bg:#161414;--fg:#efeceb;--muted:#a9a3a0;--card:#211e1e;--line:#3a3534;--accent:#e0788c;--ok:#6fcf97;--warn:#f2c06b;--off:#9a9a9a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 10px;color:var(--accent)}
p,li{color:var(--fg)}.muted{color:var(--muted)}code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid var(--line)}th{font-weight:600;color:var(--muted)}
.wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}.b{display:inline-block;border-radius:999px;padding:1px 8px;font-size:12px;border:1px solid currentColor;white-space:nowrap}
.ok{color:var(--ok)}.vlm{color:var(--warn)}.off{color:var(--off)}details{margin:6px 0}summary{cursor:pointer;color:var(--muted)}
blockquote{margin:6px 0 0;padding-left:10px;border-left:3px solid var(--line);color:var(--muted);font-size:13px}
@media (max-width:700px){table,thead,tbody,tr,th,td{display:block}thead{display:none}tr{border-bottom:1px solid var(--line);padding:6px 0}td{border:0;padding:4px 10px}}
"""


def page() -> str:
    e = html.escape
    trs = []
    for tid, n in rows():
        st = status(tid)
        cls = "ok" if tid in PROD else "vlm" if tid in VLM else "off"
        rel = "связанное требование" if n.get("interpretive") else "прямая норма"
        url = n.get("url", "")
        link = f'<a href="{e(url)}" target="_blank" rel="noopener">adilet</a>' if url else ""
        quote = n.get("quote", "")
        trs.append(
            f"<tr><td><strong>{e(n['title'])}</strong><br><span class='b {cls}'>{e(st)}</span></td>"
            f"<td>{e(HOW.get(tid) or n.get('observable', ''))}"
            + (f"<details><summary>исключения</summary>{e(n.get('exclusions', ''))}</details>" if n.get("exclusions") else "")
            + f"</td><td>{e(n['document'])}, <strong>{e(n['clause'])}</strong> · {e(rel)} · {link}"
            + (f"<blockquote>«{e(quote)}»</blockquote>" if quote else "") + "</td></tr>")
    m = json.load(open(os.path.join(ROOT, "models", "scorer.json")))
    n_on = sum(1 for s in m["stumps"].values() if s.get("enabled", True))
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CashGuard — о решении</title><meta name="description" content="Архитектура, нормативные источники, типы нарушений и ограничения CashGuard (Datathon 26, кейс ForteBank)">
<style>{CSS}</style></head><body><main>
<h1>CashGuard — о решении</h1>
<p class="muted">Datathon 26 · кейс ForteBank · контроль кассовой дисциплины по видео. Результат — подсказка для проверки человеком, не юридическое заключение.</p>

<h2>1. Адреса</h2>
<div class="card"><p>Интерфейс: <a href="/">{URL}</a></p>
<p>API (ТЗ §10): <code>POST {URL}/api/v1/analyze</code> (поле <code>file</code>) → <code>job_id</code>;
<code>GET {URL}/api/v1/jobs/{{job_id}}</code> → <code>processing | done | failed</code> и результат. Ключ — в заголовке <code>X-API-Key</code>, без ключа — HTTP 401.</p>
<pre><code>curl -X POST {URL}/api/v1/analyze -H "X-API-Key: &lt;ключ&gt;" -F "file=@video.mov"
curl {URL}/api/v1/jobs/&lt;job_id&gt; -H "X-API-Key: &lt;ключ&gt;"</code></pre></div>

<h2>2. Запуск и использование</h2>
<div class="card"><ul>
<li>Интерфейс: загрузить <code>.mov</code> → статус обработки → вердикт «Нарушение не обнаружено» / «Обнаружены признаки возможных нарушений», события с таймкодом, 1–3 кадра-доказательства, описание и норма.</li>
<li>Один контейнер: React-интерфейс, Go API, Python-анализатор; Postgres для истории. Сборка: <code>docker build .</code> (корневой Dockerfile) или <code>docker compose up -d --build</code>.</li>
<li>Все веса загружаются при сборке образа; для анализа интернет не нужен (<code>HF_HUB_OFFLINE=1</code>) — переносится во внутренний контур банка.</li>
<li>Лимит 5 минут считается сервером от момента загрузки; при одновременной загрузке нескольких видео время делится между ними, ответ приходит всегда.</li>
</ul></div>

<h2>3. Архитектура и модели</h2>
<div class="card"><ul>
<li><strong>Восприятие (всё локально, CPU):</strong> YOLOX-M/S (Apache-2.0) — люди и предметы; RTMPose-m (Apache-2.0) — поза; CLIP ViT-B/32 (MIT) — смысловые пробы; OWLv2-base (Apache-2.0) — предметы на столе и ключ в замке. Опционально Qwen3-VL (Apache-2.0) на Mac.</li>
<li><strong>Сцена без координат камеры:</strong> трекинг людей, выбор кассира и его рабочего места, зоны стола/двери/сейфа выводятся из движения людей; все расстояния — в долях роста кассира. Распознавания лиц нет.</li>
<li><strong>Решение:</strong> кандидаты событий → по одному измеренному признаку и обученному порогу на тип ({n_on} включённых правил); отбор признаков по перекрёстной проверке «исключи камеру целиком» (7 постановок); функция потерь 2·пропуск + ложное.</li>
<li><strong>Описание события</strong> строится только из измеренного: «Наблюдается / Квалификация / Норма + дословная цитата».</li>
<li><strong>Качество на 40 видео датасета:</strong> проверка «новая камера» (каждая из 7 постановок исключалась из обучения) — 0,775 верных вердиктов, 0 ложных тревог на 19 нормальных видео; все 40 видео на этом стенде — 35/40, 0 ложных; тип верен у 15 из 16 найденных; медиана ошибки таймкода 0,6 с (оценка на данных разработки).</li>
</ul></div>

<h2>4. Нормативные источники и сопоставление</h2>
<div class="card"><p>Только акты РК, тексты сверены с adilet.zan.kz: постановление Правления НБРК от 29.11.2019 № 231 (Правила кассовых операций и инкассации) — основной; постановление НБРК от 24.02.2020 № 14; постановление АРРФР от 03.04.2026 № 53.
Каждый тип связан с документом, пунктом и дословной цитатой. Если норма не описывает наблюдаемое действие прямо, она помечена «связанное требование». Сервис не утверждает того, что по видео проверить нельзя (проводки, сверки, блокировки учётной записи).</p></div>

<h2>5. Типы нарушений</h2>
<div class="wrap"><table><thead><tr><th style="width:26%">Тип и статус</th><th style="width:34%">Что видно на видео</th><th>Норма</th></tr></thead><tbody>
{''.join(trs)}
</tbody></table></div>

<h2>6. Ограничения и известные ошибки</h2>
<div class="card"><ul>
<li>Не распознаются действия рук с деньгами: приём без документа, возврат денег клиенту, вскрытие сумки без пересчёта, вложение отбракованных купюр.</li>
<li>Типы с 1–2 обучающими примерами (сейф, ключ, напиток, экран) имеют пороги без статистической опоры: на новом ракурсе возможен пропуск. Модель настроена не выдавать ложных тревог.</li>
<li>Экран считается по ноутбуку; отдельный монитор может не распознаваться как рабочая станция.</li>
<li>Посторонний распознаётся только при коротком пребывании стоя (&lt; 12 с) или при трёх и более людях; долгое присутствие стоящего человека может быть пропущено.</li>
<li>На общем CPU облака длинные видео анализируются с пониженной частотой кадров; если она ниже 1 кадра/с, результат помечается предупреждением.</li>
<li>Если рабочее место кассира не найдено (необычный ракурс), результат выдаётся с предупреждением «вывод ненадёжен».</li>
</ul></div>
<p class="muted">Код: приватный репозиторий команды; доступ — по запросу жюри.</p>
</main></body></html>
"""


def main():
    md = markdown()
    open(os.path.join(ROOT, "docs", "violation_types.md"), "w").write(
        "# Типы нарушений (генерируется `scripts/make_about.py` из `app/rules/norms.yaml`)\n\n" + md)
    open(os.path.join(ROOT, "src", "frontend", "public", "about.html"), "w").write(page())
    print(md)


if __name__ == "__main__":
    main()
