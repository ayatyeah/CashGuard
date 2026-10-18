# K8. Mac M3 Pro 18 GB + VLM 4B как второй стенд: go/no-go за 2 часа, безопасный запуск за день

Агент K8 из 10. Дата: 2026-10-18. Прочитано: `killer/CONTEXT.md`, `02_local_vlm_mac.md`, `09_mac_performance.md`,
`docs/mac_18gb.md`, `docs/deploy_mac.md`, `docs/deploy_railway.md`, `app/ml/vlm.py`, `app/ml/moments.py`,
`scripts/{run_vlm,confirm_vlm,make_vlm_sft,train_scorer,cashguard_analyzer}.py`, `models/scorer.json`,
`data/vlm/*.json` (128 моментов, ответы 2B), `data/sft/{train,val}.jsonl`. Модели не запускались (Linux).

---

## 0. Вердикт коротко

**Это не «killer», а умеренный апгрейд с реальным операционным риском.** Числа:

- VLM-признаки сейчас дают **+1 видео из 40** на train (Х-40, личные вещи): без VLM 0.775 / 0 FP (LOCO,
  `data/train_scorer_40_cv.log`), с VLM-2B 0.800 / 1 FP (`train_scorer_40_vlm.log`; FP на Х-16 — сумка
  инкассатора, позже закрыт фильтром `short_visit` в `moments.to_candidates`, в текущем `scorer.json`
  у `vlm_client_gives_bag` n_neg=0).
- На Railway `FORTE_USE_VLM=0` (CPU), но в production `models/scorer.json` **три VLM-стампа включены**
  (`vlm_teller_drinks`, `vlm_handbag_or_coat_on_desk`, `vlm_client_gives_bag`), каждый с **n_pos=1, n_neg=0**.
  То есть «включить VLM на Mac» = включить три правила, обученные на одном позитиве каждое.
- Ожидаемый прирост на жюри (5+ видео): **+3…+5 балла** при FP=0, **−5 за каждый ложный** (расчёт в §6).
  Единственное, что VLM даёт, чего нет у CV/OWL: типы `personal_items` и (потенциально) `phone_use`.

**Рекомендация.**
1. A/B за 2 часа по §2 с численным правилом. Если NO-GO — Mac остаётся демо для питча, сдача не меняется.
2. Если GO — **один URL для жюри остаётся Railway**; Mac подключается *за* Railway как удалённый анализатор
   с автоматическим откатом на локальный CPU (§4). Никогда не сдавать два адреса и не сдавать адрес ноутбука.
3. Модель для одного дня — `mlx-community/Qwen3-VL-4B-Instruct-4bit` (та же семья, что 2B: промпты, парсер и
   mlx-vlm 0.7.8 проверены). Qwen3.5-4B — только если 4B-VL провалит A/B и останется час: у неё thinking
   по умолчанию, а `vlm.py` не передаёт `enable_thinking=False`. 8B/9B — нет (13–14 GB резидентно, 9–12 с на
   вызов, `09_mac_performance.md` §1.2, §6).

---

## 1. Что говорят данные про 2B (база для A/B)

### 1.1. Ответы 2B против слабой «истины» по видео

`data/vlm/*.json` — 128 моментов на 40 видео (teller 96, visitor 19, desk 13), ответы transformers-ветки 2B на
Linux. «Истина» — словари `TELLER/DESK/VISITOR/BETWEEN` из `scripts/make_vlm_sft.py` (разметка по видео, не по
кропу; считались только ключи с известным ответом). После `confirm_vlm.py` (2-из-3) меняется только
`teller_drinks` (FP 1 → 0).

| Ключ | n | TP | FP | FN | TN | Вывод |
|---|---|---|---|---|---|---|
| handbag_or_coat_on_desk (**в prod**) | 50 | 2 | **0** | 0 | 48 | чистый; единственный источник `personal_items` |
| teller_drinks (**в prod**) | 50 | 1 | 1→**0** | 1 | 47 | дублирует `drink_clip`; FN на Х-46 в одном из моментов |
| client_gives_bag (**в prod**) | 13 | 2 | **0** | 0 | 11 | чистый; даёт верный тип Х-17 (`collection_bag_no_check`) |
| cup_or_food_on_desk | 50 | 1 | 2 | 1 | 46 | FP Х-14, Х-25 → выключен |
| teller_holds_phone | 50 | 0 | **13** | 0 | 37 | 26 % моментов — ложное «да» на 7 нормальных видео; `phone_use` включать нельзя |
| cash_on_desk (between) | — | — | — | — | — | в `data/vlm` между визитами нет моментов с истиной (Х-4/Х-38 не предложены) |
| cash_left_on_desk (desk) | 11 | 1 | **8** | 0 | 2 | бесполезен |
| bag_left_on_desk | 11 | 1 | 5 | 0 | 5 | бесполезен |
| documents_or_notebook_left | 11 | 1 | 5 | 0 | 5 | бесполезен |
| laptop_screen_on | 11 | 2 | 6 | 0 | 3 | хуже `leave_screen` (CV) |
| desk_is_tidy | 11 | 0 | 0 | 8 | 3 | всегда «нет» |
| client_shows_id_card | 13 | 3 | 4 | 2 | 4 | шум; X-2 (без ID) не ловится |
| client_gives_cash | 13 | 6 | 0 | 5 | 2 | recall 0.55 |

Итог: 2B надёжна ровно на трёх «крупнопредметных» ключах (сумка/куртка, сумка инкассатора, напиток) и шумит на всём,
что мельче ладони (деньги, документы, телефон, ID). Это и есть планка для 4B: **не потерять три чистых ключа и
снять FP хотя бы с `teller_holds_phone` и `cup_or_food_on_desk`**. Больше параметров не вернёт детали 10–40 px,
потерянные при `compose(max_side=640)` (`docs/mac_18gb.md`), поэтому ждать чуда на `cash_*` не стоит.

### 1.2. SFT-пары как тест для базовой модели (до любого дообучения)

`data/sft/train.jsonl` 231 строка = 77 оригиналов + 154 аугментации (`_aug0/_aug1`), `val.jsonl` 11 строк
(кабинеты C,E: Х-16, Х-17, Х-19, Х-31). Для оценки базовой модели брать **только `*_orig.jpg` + val = 88 пар**
(аугментации — те же кадры). Внимание: файлы в `data/sft` используют ключ `"image"` (строка), а текущий
`make_vlm_sft.py` пишет `"images"` (список) — скрипт оценки должен принимать оба.

Пары на вопрос (оригиналы train + val): teller 43+7 = **50**, visitor 10+3 = **13**, desk **11**, safe **9**, between 4+1 = **5**.
Позитивы (`true`) по ключам: teller_drinks 2, handbag_or_coat_on_desk 2, cup_or_food_on_desk 2, teller_holds_phone **0**,
client_gives_bag 2, client_shows_id_card 5, client_gives_cash 11, cash_left_on_desk 1, bag_left_on_desk 1,
documents_or_notebook_left 1, laptop_screen_on 2, desk_is_tidy 8, key_in_safe_lock 1, safe_door_open **0**, cash_on_desk 2.
Все метки — слабые (по видео): Х-20 размечен `safe_door_open=True`, но в SFT ни одного `true` — момент для него не
предложен. Пустой support (phone, safe_door_open) = оценивается только FP.

Это тот же источник истины, что в §1.1, поэтому 88 SFT-пар и 128 моментов `run_vlm.py` — **одна и та же проверка**,
просто SFT добавляет safe/between-моменты. Делать одно из двух; ниже — через `run_vlm.py`, потому что его выход
напрямую идёт в `train_scorer.py` (LOCO-гейт), а SFT-пары — как контроль формата.

### 1.3. Прочие факты для плана

- Видео: train/test ≤ **27.4 MB** (Х-14, 143 с), медиана ~8 MB; ТЗ — «~29 MB». Лимит тела 100 MB (Cloudflare Free)
  не достигается с запасом 3×. Go принимает до 501 MB (`http.go:108`), nginx фронта 501m.
- Данные для A/B на Mac: `data/cache` **705 MB** (42 pkl + 162 папок кадров, включая `__aug`) + `configs/` +
  `data/{train,test}/labels.csv`. Переносить целиком (rsync/AirDrop), иначе `run_vlm.py` не соберёт кропы.
- `load_labels()` при `test_released: true` помечает все 40 видео как `train` → `run_vlm.py --split train` обходит все 40.
  Скрипт пропускает видео с уже существующим файлом → всегда новый `--out`.
- mlx-ветка `VLM.ask` отдаёт системный промпт отдельной ролью (`deploy_mac.md` §8 предупреждает о расхождении с
  transformers-веткой) → для честного парного сравнения 2B надо тоже прогнать на MLX (7 минут), не брать Linux-ответы.
- Railway: 1.15 с/кадр, Х-14 упаковывается в 172 с от загрузки; очередь из 3 одновременных — хвост 188 с.
  Один воркер, `JOB_DEADLINE_SEC=285`, бюджет передаётся анализатору вторым аргументом.

---

## 2. A/B-протокол на Mac: 2 часа, численное правило

Предпосылка: Mac настроен по `docs/deploy_mac.md` §1–3 и `docs/mac_18gb.md` (venv `.venv`, mlx-vlm 0.7.8, веса
YOLOX/RTMPose/CLIP, 2B в HF-кэше). Все команды из корня репо, `source .venv/bin/activate`.

| Время | Шаг | Команда / что смотреть |
|---|---|---|
| 0:00–0:15 | Скачать 4B (2.8 GB), один замер | `bash scripts/download_models.sh --vlm mlx-community/Qwen3-VL-4B-Instruct-4bit`; затем `export HF_HUB_OFFLINE=1 FORTE_VLM_BACKEND=mlx FORTE_VLM_MLX=$PWD/models/vlm/mlx-community--Qwen3-VL-4B-Instruct-4bit`; `python -m mlx_vlm.generate --model "$FORTE_VLM_MLX" --image data/sft/images/X-14_teller_0_orig.jpg --prompt 'Is cash visible on the desk? Answer JSON {"cash_on_desk": ...}' --max-tokens 60 --temperature 0 --verbose` → prompt tok/s, gen tok/s, **peak memory** |
| 0:15–0:25 | 2B на MLX (пара для сравнения) | `FORTE_VLM_MLX=mlx-community/Qwen3-VL-2B-Instruct-4bit python scripts/run_vlm.py --split train --out data/vlm_2b_mlx` (128 вызовов × ~3 с) |
| 0:25–0:45 | 4B на тех же моментах | `python scripts/run_vlm.py --split train --out data/vlm_4b` (128 × ~5–6 с). В stdout у каждого видео печатается время — записать p50/p95 по видео |
| 0:45–0:55 | 2-из-3 только для позитивов | `python scripts/confirm_vlm.py --src data/vlm_2b_mlx --dst data/vlm_2b_mlx_confirmed`; то же для `data/vlm_4b` → `data/vlm_4b_confirmed` |
| 0:55–1:05 | Согласие с истиной по ключам | скрипт ниже (§2.1) для четырёх папок |
| 1:05–1:20 | LOCO-гейт | `cp models/scorer.json /tmp/scorer_prod.json`; `FORTE_VLM_DIR=data/vlm_4b_confirmed python scripts/train_scorer.py --with-vlm --vlm-confirmed --output /tmp/scorer_4b.json 2>&1 \| tee /tmp/loco_4b.log`; то же для `vlm_2b_mlx_confirmed` → `/tmp/scorer_2b.json`. Смотреть строку `verdict acc=… FP=…` блока CV (leave-one-CABINET-out). `models/scorer.json` не трогать |
| 1:20–1:50 | Сквозной прогон как у жюри | `cp /tmp/scorer_4b.json models/scorer.json` (только если G3 прошёл), `FORTE_USE_VLM=1 /usr/bin/time -l bin/cashguard-analyzer "$PWD/data/test/videos/Х-14.mov" 255 > /tmp/r14.json` → wall-time, `peak memory footprint`, в stderr `timings=` с `vlm_moments`, нет `VLM disabled`. Потом Go локально (`deploy_mac.md` §5) и `BASE=http://127.0.0.1:8080 KEY=… bash scripts/smoke_queue.sh Х-26.mov Х-27.mov Х-14.mov` |
| 1:50–2:00 | Решение по §2.2, запись чисел в этот файл | — |

Если 4B не влез по времени (p50 > 7 с) — не продолжать: `FORTE_VLM_MAX_SIDE` менять нельзя (признаки скорера), а
8B тем более. Параллельно с шагами 0:25–0:45 можно готовить туннель (§3) — это не мешает GPU.

### 2.1. Скрипт согласия с истиной (сохранить как `/tmp/vlm_agree.py`, запускать из корня)

```python
import glob, json, os, sys, collections
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from make_vlm_sft import TELLER, TELLER_NEG, DESK, VISITOR, BETWEEN, SAFE
import app.ml.vlm as V
neg_t = {"teller_holds_phone": False, "teller_drinks": False, "cup_or_food_on_desk": False, "handbag_or_coat_on_desk": False}
truth = {"teller": {**{n: neg_t for n in TELLER_NEG}, **TELLER}, "desk": DESK, "visitor": VISITOR,
         "between": BETWEEN, "safe": SAFE}
for src in sys.argv[1:]:
    st = collections.defaultdict(collections.Counter); fpv = collections.defaultdict(set)
    n_ans = n_full = 0
    for f in sorted(glob.glob(f"{src}/*.json")):
        name = os.path.basename(f)[:-5]
        for m in json.load(open(f)):
            a = m.get("answer") or {}
            if not m.get("skipped"):
                n_ans += 1; n_full += int(set(a) >= set(getattr(V, m["q"])[0]))
            t = truth.get(m["kind"], {}).get(name, {})
            for k, v in a.items():
                if k in t:
                    c = ("TP" if t[k] else "FP") if v else ("FN" if t[k] else "TN"); st[k][c] += 1
                    if c == "FP": fpv[k].add(name)
    print(f"== {src}: full-JSON {n_full}/{n_ans} = {n_full / max(n_ans, 1):.2f}")
    for k in sorted(st):
        s = st[k]; print(f"{k:30s} n={sum(s.values()):3d} TP={s['TP']:2d} FP={s['FP']:2d} FN={s['FN']:2d} TN={s['TN']:2d} FP@{sorted(fpv[k])}")
```

`python /tmp/vlm_agree.py data/vlm data/vlm_2b_mlx_confirmed data/vlm_4b data/vlm_4b_confirmed`. Эталон для `data/vlm`
— таблица §1.1 (должна воспроизвестись один в один).

### 2.2. Правило go/no-go (все пункты G обязательны)

| # | Критерий | Порог | Откуда порог |
|---|---|---|---|
| G1 | Скорость/память 4B | p50 вызова ≤ 7 с, p95 ≤ 12 с на склейке 640 px; peak memory анализатора ≤ 11 GB; swap = 0 после 3 видео подряд | бюджет VLM в `ml.py` ≥ 90 с → ≥ 10 вызовов на видео; `09` §6: 4B резидентно 10–11 GB |
| G2 | Формат | ≥ 95 % ответов содержат **все** запрошенные ключи (строка `full-JSON`) | иначе `ask_flags` теряет ключи → `count/frac` искажены |
| G3 | Три prod-ключа после confirm | `handbag_or_coat_on_desk`: FP=0, TP≥2; `client_gives_bag`: FP=0, TP≥2; `teller_drinks`: FP=0, TP≥1 | не хуже 2B (§1.1) — это ключи, которые реально в `scorer.json` |
| G4 | LOCO с 4B-ответами | `verdict acc ≥ 0.800` **и** `FP = 0` в CV-блоке `/tmp/loco_4b.log`; type correct ≥ 0.92 | правило проекта «принимается только если LOCO не падает» (0.775/0 FP без VLM; 0.800 с VLM-2B) |
| G5 | Сквозняк | Х-14 через `bin/cashguard-analyzer … 255` ≤ 200 с wall, `vlm_moments > 0`, `vlm_skipped = 0`; очередь из 3 через Go — все `done` ≤ 300 с | запас 55 с на туннель/переотправку (§4) |
| B1 (бонус, не гейт) | Новые ключи | `teller_holds_phone` FP ≤ 1 из 50 (2B: 13), `cup_or_food_on_desk` FP = 0 (2B: 2) | только тогда имеет смысл обсуждать `phone_use`/еду как VLM-типы; без позитивов в train стамп не обучить → правило вручную `count ≥ 2`, с явным риском |

- **GO** — G1–G5 выполнены → §3–§4, на Mac ставится `/tmp/scorer_4b.json` (на Railway `scorer.json` не меняется,
  там VLM выключена — но проверить, что `--with-vlm` не сдвинул CV-стампы; если сдвинул, на Railway класть тот же
  файл, он совместим: VLM-кандидатов там просто нет).
- **NEUTRAL** — G1–G3, G5 ок, но G4 даёт 0.775/0 FP (прироста нет) → стенд не нужен, Mac = демо на питче с живым
  прогоном Х-40/Х-46 и VLM-доказательствами.
- **NO-GO** — провал любого из G1–G4 → ничего не менять, не тратить вечер на туннель.

Что **не** считать аргументом «за»: улучшение на ключах `cash_*`/`laptop`/`desk_is_tidy` — они не входят в
`scorer.json`, и включать их по 1–3 позитивам нельзя (уже проверено на 2B: FP 5–8 из 11).

---

## 3. Mac как публичный стенд: туннель и окружение

### 3.1. Сравнение туннелей

| | cloudflared **named** tunnel | Tailscale **Funnel** | ngrok free |
|---|---|---|---|
| Цена | 0 | 0 (Personal) | 0 |
| Нужен свой домен | **да**, с DNS на Cloudflare (регистрация + смена NS — часы; если домена нет, за день рискованно) | нет: `https://<mac>.<tailnet>.ts.net` | нет: один фиксированный `*.ngrok-free.app`, имя выбрать нельзя |
| Лимит тела запроса | **100 MB** (Free/Pro) → 413; наши видео ≤ 28 MB | не документирован; трафик через релей Tailscale с **неопубликованным лимитом полосы** → обязательно замерить загрузку 30 MB | не документирован; **1 GB/мес исходящего**, 20 000 запросов, 3 endpoint'а |
| Таймауты | 100 с на ответ origin (524) — наш API асинхронный, ок | — | — |
| Порты | любой локальный | только 443 / 8443 / 10000 снаружи | любой |
| Особенности | как сервис (`cloudflared service install`), `--no-autoupdate`, `--protocol http2` если на площадке режут UDP | ACL `funnel` в политике tailnet, HTTPS включить в админке; CLI 10 минут | третьи источники: interstitial-страница для браузеров (curl/жюри не затронуты; UI в браузере — предупреждение) |
| Quick tunnel `trycloudflare.com` | случайный URL, без SLA — **только для smoke**, не для сдачи | — | — |

Выбор: **если домен на Cloudflare уже есть — cloudflared named; если нет — Tailscale Funnel** (ставится за 10 минут,
стабильное имя, без домена), ngrok — резерв. Так как адрес Mac **не** попадает в форму сдачи (§4), его стабильность
нужна только для канала Railway → Mac, и любой из трёх достаточен после теста загрузки 30 MB.

```bash
# Tailscale Funnel (Go слушает 127.0.0.1:8080)
brew install tailscale && sudo tailscale up            # в админке: DNS → HTTPS Certificates: Enable; Access controls: nodeAttrs funnel
tailscale funnel --bg --https=443 http://127.0.0.1:8080
tailscale funnel status                                  # → https://<mac>.<tailnet>.ts.net
curl -sS -m 120 -X POST https://<mac>.<tailnet>.ts.net/api/v1/analyze -H "X-API-Key: $KEY" -F file=@data/test/videos/Х-14.mov   # 27 MB — замер времени загрузки

# cloudflared named tunnel (есть домен на Cloudflare)
brew install cloudflared && cloudflared tunnel login
cloudflared tunnel create forte-mac && cloudflared tunnel route dns forte-mac mac.example.kz
cat > ~/.cloudflared/config.yml <<EOF
tunnel: forte-mac
credentials-file: /Users/$USER/.cloudflared/<UUID>.json
no-autoupdate: true
protocol: http2
ingress:
  - hostname: mac.example.kz
    service: http://127.0.0.1:8080
    originRequest: { connectTimeout: 30s, noTLSVerify: true }
  - service: http_status:404
EOF
sudo cloudflared service install                       # LaunchDaemon, логи /Library/Logs/com.cloudflare.cloudflared.*.log
```

### 3.2. Окружение Go-сервера на Mac (наследуется анализатором)

```bash
export API_KEY=<тот же ключ, что на Railway>   DEMO_MODE=false   DATABASE_URL='postgres://cashguard:…@127.0.0.1:55432/cashguard?sslmode=disable'
export ANALYZER_COMMAND="$PWD/bin/cashguard-analyzer"
export JOB_DEADLINE_SEC=200            # Mac отвечает быстрее, чем дедлайн Railway (285), чтобы остался запас на откат (§4)
export FORTE_USE_VLM=1 FORTE_VLM_BACKEND=mlx FORTE_VLM_MLX="$PWD/models/vlm/mlx-community--Qwen3-VL-4B-Instruct-4bit"
export FORTE_USE_OWL=1                 # как на Railway, иначе разные наборы признаков на двух контурах
export FORTE_COST_PER_FRAME=<замер (perception+clip)/frames на M3, ожидается 0.3–0.45> FORTE_THREADS=6
export FORTE_VLM_CALL_SEC=<p50 из A/B, ~6>   # стартовая оценка планировщика VLM (по умолчанию 35 с — первый момент может быть отброшен зря)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE
caffeinate -dimsu go -C src/backend run ./cmd/server   # или launchd с KeepAlive
```

Модель грузится на каждое видео (процесс на видео): 4B 6–10 с + компиляция Metal — укладывается, постоянный
сервер моделей (`09` §5) — работа на день, в этот план не входит.

---

## 4. Как показать жюри два контура без путаницы: один URL

Жюри гоняет API автоматически по адресу из формы сдачи; шортлист собирается из результатов API. Поэтому:

1. **В форме — только Railway** (`https://fortecashguard.up.railway.app`), он проверен (`deploy_railway.md`:
   Х-14 за 172 с, очередь из 3 — хвост 188 с), не спит, не зависит от Wi-Fi площадки.
2. **Mac — за Railway, не рядом.** Контракт `ANALYZER_COMMAND` позволяет сделать это без правок Go: лаунчер
   `bin/cashguard-analyzer-remote` (bash + python, urllib, ~60 строк, без зависимостей) вместо прямого запуска:

   ```
   budget = argv[2]  (Railway передаёт остаток до дедлайна от загрузки)
   if FORTE_REMOTE_URL задан and budget ≥ 150 s and GET $REMOTE/api/health ≤ 3 s == ok:
       POST $REMOTE/api/v1/analyze (file, X-API-Key)            # 27 MB → Mac за 5–30 с в зависимости от канала
       poll GET /api/v1/jobs/{id} каждые 5 с до done | failed | истекло min(budget − 70, 190) с
       done → напечатать {"status","verdict","violations"} (лишние поля Mac отбросить) → exit 0
   fallback: exec bin/cashguard-analyzer <video> <оставшийся budget>      # локальный CPU, как сейчас
   ```

   Свойства: один адрес; при любом сбое Mac (сон, Wi-Fi, 413, таймаут) ответ всё равно приходит от Railway в
   пределах 5 минут (при откате после 190 с остаётся ≥ 60 с — `ml.py` отдаёт CV-результат в режиме `cheap`,
   без OWL/VLM; это дешевле пропуска видео). Railway один воркер → на Mac приходит по одному видео, его очередь
   пуста. Хвост очереди из 5 одновременных (budget < 150 с) сразу идёт локально.
   Трудозатраты: 2 ч код + 1 ч тест `smoke_queue.sh` на Railway с включённым `FORTE_REMOTE_URL`. Выключение —
   удалить переменную в Railway (`railway variables`), редеплой не нужен, если лаунчер читает её при старте.
3. **Если нет 3 часов на лаунчер** — Mac только на питче: живой прогон Х-40/Х-46 с VLM-доказательствами и слайд
   «локальный контур на Apple Silicon: тот же API, + визуально-языковая модель». В SUBMISSION.md уже есть строка про
   резервный контур (`docs/mac_18gb.md`) — этого достаточно, URL Mac туда не вписывать.
4. Для прозрачности в ответе API можно добавить необязательное поле `engine: "cpu" | "mlx-vlm"` (доп. поля
   допустимы по ТЗ, `api_contract_gap.md`) — жюри и мы увидим, какой контур ответил.

Mac как **первичный** адрес отвергнут: ноутбук на Wi-Fi площадки, без автоматического отката (без своего домена с
DNS на Cloudflare переключить URL после сдачи невозможно), и за минус от одного зависшего видео (−11 баллов)
никакой VLM не отыгрывает.

---

## 5. Реестр рисков (на день жюри)

| # | Риск | Вероятность / ущерб | Митигация (часы) | Проверка |
|---|---|---|---|---|
| 1 | Сон / крышка / выключение дисплея | высокая / Mac недоступен → откат на Railway, минус VLM | `sudo pmset -a sleep 0 displaysleep 0 disksleep 0 standby 0 autopoweroff 0 powernap 0 tcpkeepalive 1`; питание от сети; **крышка открыта** (clamshell на Apple Silicon ненадёжен и греется); серверы под `caffeinate -dimsu`; авто-обновления и уведомления выключены (0.2 ч) | `pmset -g assertions`; оставить на 30 мин, проверить `/api/health` |
| 2 | Термотроттлинг (−20…50 % за 10 мин GPU) | средняя / VLM-моменты отбрасываются по бюджету | `FORTE_COST_PER_FRAME` ×1.3 от замера; `JOB_DEADLINE_SEC=200`; твёрдая поверхность, прохлада; адаптивное прореживание уже в `ml.py` (0.1 ч) | tok/s 1-го и 5-го видео подряд в `--verbose`; `vlm_skipped` в timings |
| 3 | Память MLX / swap | низкая при 4B (≈10–11 GB резидентно) / замедление ×10 | только 4B 4-bit; `FORTE_VLM_MAX_SIDE` не менять; один анализатор (1 воркер Go); закрыть браузеры/IDE; PostgreSQL через brew, не Docker; `iogpu.wired_limit_mb` не поднимать (`mac_18gb.md`) (0 ч) | `sysctl vm.swapusage` после очереди из 3; `peak memory footprint` ≤ 11 GB |
| 4 | Загрузка модели на каждое видео (6–10 с + Metal) | точно / −15…30 с бюджета | принять: укладывается в G5; холодный старт после перезагрузки — прогреть одним видео; постоянный сервер — не за день (0.1 ч) | первый прогон после ребута ≤ 200 с |
| 5 | Wi-Fi площадки / UDP закрыт / туннель упал | средняя / откат на Railway | Ethernet USB-C, резерв — хотспот телефона; cloudflared `--protocol http2`, `--no-autoupdate`; watchdog `launchctl kickstart -k` по провалу `/api/health` раз в 60 с (0.5 ч) | выдернуть сеть на минуту во время тестового задания → Railway всё равно `done` ≤ 300 с |
| 6 | 413 / долгая загрузка на Mac | низкая (≤ 28 MB vs 100 MB) / откат | лаунчер считает отказ POST откатом сразу, бюджет не теряется (в коде §4) | POST файла 27 MB через туннель < 30 с |
| 7 | Разное распределение ответов 4B vs 2B → стампы `thr 1.8` не срабатывают | средняя / VLM молчит | скорер переобучить на 4B-ответах (G4), на Mac класть `/tmp/scorer_4b.json`; Railway не трогать (0.3 ч) | G3/G4 |
| 8 | Qwen3.5 thinking → `<think>` в тексте, парсер берёт мусор | высокая для 3.5 / 0 для 3-VL-Instruct | не брать Qwen3.5 без правки `vlm.py` (`chat_template_kwargs={"enable_thinking": False}`) | — |
| 9 | Дрейф формата mlx-ветки (system отдельной ролью) | уже известен / признаки ≠ transformers | в A/B пара 2B-MLX vs 4B-MLX, скорер учится на MLX-ответах (§2) | воспроизводимость §1.1 на `data/vlm_2b_mlx` ±2 |
| 10 | Кто-то пользуется ноутбуком во время жюри / ручной smoke на Mac | средняя / конкуренция за GPU и память | ноутбук выделенный, экран заблокирован, никаких демо через Mac в окно автопроверки; демо для питча — отдельное время | — |
| 11 | Mac и Railway дают разные вердикты на одном видео | средняя / вопросы жюри | поле `engine` в ответе; на питче объяснить честно: CPU-контур = подмножество признаков | сравнить Х-26/Х-27/Х-14 на обоих контурах |
| 12 | Утечка данных | низкая | оба контура за `X-API-Key`; кадры/ответы не публиковать; `HF_HUB_OFFLINE=1`, в инференсе нет сетевых вызовов | `lsof -i` во время анализа |

---

## 6. Ожидаемый прирост в баллах

Оценка по `08_competition_tactics.md`: при N = 5 пропуск нарушения стоит до 55/N = **11** баллов (детекция +
локализация + классификация), ложное на нормальном видео — 25/N = **5** плюс доверие жюри.

| Тип | Что добавляет VLM | База 2B (FP на train) | Оценка P(в тесте) | Вклад |
|---|---|---|---|---|
| `personal_items` (сумка/куртка на столе) | единственный источник: CV/OWL не видят; Х-40 ловится только с VLM | 0 FP / 50 моментов, 2 TP | ~30 % (в списке типов организаторов) | +0.3 × 11 ≈ **+3.3** |
| `collection_bag_no_check` | верный **тип** для Х-17 (без VLM — `unauthorized_person` = классификация мимо, −3 из 15/N) и короткие визиты с сумкой | 0 FP / 13 (после фильтра short_visit) | ~20 % | +0.2 × 3 ≈ **+0.6** |
| `food_drink` | дублирует `drink_clip`; добавляет еду без питья (`cup_or_food_on_desk`), но у 2B 2 FP → выключен | — | ~30 % | ≈ 0 (страховка), с B1 — +0.5 |
| `phone_use` | потенциально новый тип | 2B: 13 FP / 50 → **нельзя** | ~30 % | 0 без B1; с B1 (FP ≤ 1/50) и ручным правилом `count ≥ 2`: +0.3 × 11 × 0.5 ≈ +1.6 при риске −5 × P(FP) |
| `cash_*`, `no_client_id`, `rejects_merged` | нет: детали 10–40 px теряются на 640 px | 5–8 FP / 11 | — | 0 (решается треком рук/OWL, не VLM) |

**Итого: ≈ +4 балла ожидания при FP = 0; каждый ложный на нормальном тестовом видео −5.** На train у 2B
FP по трём prod-ключам = 0 на 50/50/13 моментах — это даёт грубую верхнюю границу FP-риска ~2–7 % на видео
(правило трёх). Для 4B это надо подтвердить G3; если 4B покажет хоть 1 FP на prod-ключах — ожидание уходит в ноль.

Вывод тот же, что в §0: апгрейд оправдан только как «бесплатное» дополнение за Railway с автоматическим откатом
(§4), а не как самостоятельный стенд. Главная ценность для шортлиста — не баллы API, а питч: локальная VLM на
Apple Silicon с кадрами-доказательствами и объяснением, что она отвечает только на крупнопредметные вопросы, а
мелкие объекты закрыты CV/OWL — это честно и проверяемо.

---

## Источники

- Репо: `docs/research/02_local_vlm_mac.md` (модели, память, thinking/JSON), `docs/research/09_mac_performance.md`
  (цена вызова 4B/8B на M3 Pro, бюджет памяти §6, чек-лист ноутбука §7, лимиты Cloudflare), `docs/mac_18gb.md`
  (A/B и критерии приёмки), `docs/deploy_mac.md`, `docs/deploy_railway.md`, `docs/api_contract_gap.md`,
  `docs/research/08_competition_tactics.md` (цена пропуска/ложного), `scripts/make_vlm_sft.py` (словари истины),
  `data/train_scorer_40_cv.log`, `data/train_scorer_40_vlm.log`, `models/scorer.json`.
- Tailscale Funnel: порты 443/8443/10000, неконфигурируемые лимиты полосы, только `*.ts.net`, нужны HTTPS и атрибут
  `funnel` — https://tailscale.com/kb/1223/tailscale-funnel ; обзор лимитов — https://www.ssdnodes.com/learn/tailscale-funnel-limits-and-ports
- ngrok Free: 1 GB исходящего, 20 000 запросов, 3 endpoint'а, фиксированный домен без выбора имени —
  https://ngrok.com/docs/pricing-limits/free-plan-limits ; без таймаутов сессий — https://ngrok.com/docs/guides/other-guides/limits ;
  interstitial (третьи источники) — https://dev.to/instatunnel/ngrok-vs-localxpose-localtonet-the-udp-iot-and-bandwidth-fix-4dk
- Cloudflare Tunnel: named tunnel требует домен с DNS на Cloudflare, quick tunnel без SLA — https://flaviocopes.com/cloudflare-tunnel/ ;
  лимит тела 100 MB Free/Pro, 200 Business, 413 на edge — https://fast.io/resources/cloudflare-upload-limit.md ,
  https://developers.cloudflare.com/cache/concepts/default-cache-behavior/ ; настройка — https://developers.cloudflare.com/tunnel/setup
- Qwen3.5 в mlx-vlm: конверсии `mlx-community/Qwen3.5-4B-MLX-4bit` собраны из ветки `pc/fix-qwen35-predicate`, поддержка
  позже слита в main (версия релиза не найдена; 0.7.8 — проверить `load()` перед использованием) —
  https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit , https://github.com/ml-explore/mlx-lm/issues/1136
- Qwen3-VL-4B-Instruct 4-bit (Apache-2.0, 2.8 GB) — https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit
