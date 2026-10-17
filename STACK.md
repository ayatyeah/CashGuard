# Datathon 26 · ForteBank — стек и задачи для старта (v2)

Сервис принимает видео кассы (.mov), ищет визуальные признаки нарушений кассовой дисциплины и возвращает вердикт, таймкод, кадры-доказательства и ссылку на норматив РК. Полное ТЗ — `Кейс_ForteBank.docx`.

**Твоя часть: весь сервис, кроме моделей.** Вместо модели работает заглушка (`StubAnalyzer`). Потом проект запустят на Mac M3 (8 ГБ) с настоящей моделью, поэтому:
- ничего CUDA-специфичного (bitsandbytes, vllm, flash-attn и т.п.);
- анализ подключается только через интерфейс `Analyzer` (раздел 4), и замена заглушки на модель не трогает API, UI и воркер;
- воркер сразу делается **отдельным процессом**, даже для заглушки (раздел 3).

---

## 1. Стек

| Слой | Что берём | Зачем |
|---|---|---|
| Язык и окружение | **Python 3.11**, **uv**: `pyproject.toml` + `uv.lock` в git | Воспроизводимая установка; 3.12+ ломает часть ML-библиотек |
| API | **FastAPI** + **uvicorn[standard]** + `python-multipart` | Загрузка файла, асинхронные задачи |
| Схемы и конфиг | **pydantic v2**, **pydantic-settings** | Ответ строго по ТЗ, `.env` |
| Справочник | **PyYAML** | `norms.yaml` |
| Видео | **ffmpeg / ffprobe** (системные) + **opencv-python-headless**, numpy | Метаданные, кадры по индексу, JPEG |
| Воркер | `multiprocessing` (контекст `spawn`), **один** процесс-анализатор | Модель грузится один раз; зависший процесс можно убить и перезапустить |
| Хранение задач | `data/jobs/{job_id}/`: `video.mov`, `meta.json`, `status.json`, `result.json` | Атомарная запись; переживает перезапуск |
| Фронтенд | Одна страница `static/index.html` + чистый JS | Без сборки |
| Тесты | **pytest**, `httpx` | Контракт API, таймауты, восстановление |
| Метрики и замеры | `scripts/eval.py` (+ **openpyxl**), `scripts/benchmark.py` (+ **psutil**) | Качество и время/память |
| Публичный URL | **cloudflared** (Cloudflare Tunnel) | Адрес для жюри |

### Зависимости (`pyproject.toml`)
```toml
[project]
name = "forte-cashier-guard"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi", "uvicorn[standard]", "python-multipart",
  "pydantic>=2", "pydantic-settings", "pyyaml",
  "opencv-python-headless", "numpy",
]

[project.optional-dependencies]
ml = [  # ставлю я на Mac; тиммейту не нужно
  "mlx-vlm; sys_platform == 'darwin' and platform_machine == 'arm64'",
  "supervision", "trackers", "torch", "transformers",
]

[dependency-groups]
dev = ["pytest", "httpx", "openpyxl", "psutil"]
```
Установка: `uv sync` (сервис + dev), `uv sync --extra ml` (с моделями).

---

## 2. Структура проекта

```
datathon/
├── app/
│   ├── main.py              # FastAPI, роуты, static/, восстановление при старте
│   ├── config.py            # Settings (раздел 3.4)
│   ├── schemas.py           # pydantic-модели ответа API
│   ├── store.py             # атомарная запись/чтение JSON задач
│   ├── jobs.py              # очередь, дедлайны, супервизор воркера
│   ├── worker.py            # процесс-анализатор: грузит Analyzer один раз, крутит цикл
│   ├── assemble.py          # AnalysisResult → проверка → кадры → ответ API
│   ├── video.py             # ffprobe, кадр по frame_id → JPEG base64
│   ├── analyzers/
│   │   ├── base.py          # Analyzer, AnalysisResult, Detection (раздел 4)
│   │   ├── stub.py          # StubAnalyzer
│   │   └── ml.py            # (пусто) сюда встанет модель
│   └── rules/
│       ├── norms.yaml       # справочник нарушений (раздел 5)
│       └── norms.py         # загрузка, валидация, фильтр enabled/source_verified
├── static/index.html
├── scripts/eval.py
├── scripts/benchmark.py
├── tests/
│   ├── conftest.py          # фикстура: синтетический .mov через ffmpeg testsrc
│   └── test_api.py
├── open/                    # датасет — в .gitignore
├── data/                    # задачи — в .gitignore
├── pyproject.toml, uv.lock, .env.example, README.md
```

---

## 3. API, очередь и время

### 3.1 Эндпоинты — строго по ТЗ (жюри проверяет автоматически)

Все `/api/v1/*` требуют `X-API-Key`. Нет ключа или неверный → `401`.

**`POST /api/v1/analyze`** — multipart, поле `file`.
```json
{ "job_id": "3f2a9c1e", "status": "processing" }
```
- Файл сохраняется как `data/jobs/{job_id}/video.mov`; исходное имя → `meta.json` (`original_filename`, `received_at`, `size`, `sha256`).
- Лимит `MAX_UPLOAD_MB=95` → иначе `413`. У Cloudflare на Free/Pro предел тела запроса 100 МБ; фактический предел проверить через публичный URL. Ролики ~29 МБ проходят с запасом.
- Очередь больше `MAX_QUEUE=10` → `503`. Это защита от злоупотреблений, при нормальной проверке жюри (5+ видео) срабатывать не должна.

**`GET /api/v1/jobs/{job_id}`**
```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Кассир покинул рабочее место, оставив наличные без присмотра",
      "timecode_sec": 44.2,
      "description": "Что видно на видео и почему это может быть нарушением. Норматив: <документ>, <пункт>.",
      "evidence_frames": [ { "timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..." } ],
      "rule": { "document": "...", "clause": "п. 12", "url": "https://adilet.zan.kz/..." }
    }
  ]
}
```
- `status`: `processing` (в т.ч. ожидание в очереди) | `done` | `failed` (+ поле `error`).
- `verdict`: `violation` | `no_violation`; `violations` пустой ⇔ `no_violation`.
- `evidence_frames`: **1–3** JPEG, base64 **без** префикса `data:`.
- `rule` — наше дополнительное поле; `null`, если норматив не подтверждён (раздел 5). Ссылка дублируется в конце `description`.
- Неизвестный `job_id` → `404`.

**UI-маршруты без ключа:** `POST /web/analyze`, `GET /web/jobs/{id}`, `GET /web/jobs/{id}/video` (для плеера). Авторизация в интерфейсе по ТЗ не нужна.
**`GET /health`** → `{"ok": true, "analyzer": "stub", "worker_alive": true, "queue": 0}`.

### 3.2 Время — бюджет от момента приёма файла
Жюри даёт 5 минут на всё, включая загрузку через туннель. Поэтому:
- `deadline = received_at + JOB_DEADLINE_SEC` (270 с);
- воркер передаёт анализатору `budget_sec = deadline − now − ASSEMBLE_RESERVE_SEC` (15 с на кадры и JSON);
- анализатор обязан уложиться в бюджет: при маленьком бюджете — дешёвый режим, а не отказ;
- если задача дождалась очереди, а бюджета < `MIN_BUDGET_SEC` (20 с) → `failed` с `error: "deadline"`.

### 3.3 Воркер — отдельный процесс
- Главный процесс (FastAPI) владеет очередью и статусами. Воркер — один процесс `spawn`; при старте один раз создаёт `Analyzer` (будущая модель грузится здесь), затем берёт задачи из `multiprocessing.Queue` и отвечает через вторую очередь.
- **Супервизор** (поток в главном процессе) следит за текущей задачей. После дедлайна или если воркер умер: `terminate()` → `join(5)` → `kill()`, задача → `failed` (`"timeout"` / `"worker_crashed"`), новый воркер. `Future.result(timeout)` для этого не подходит: он не останавливает работающий код.
- Задачи строго по одной.
- Для тестов у заглушки есть `STUB_DELAY_SEC` и `STUB_HANG=true` (бесконечный цикл), чтобы проверить таймаут и перезапуск.

### 3.4 Восстановление и запись
- Любой JSON пишется атомарно: во временный файл в той же папке → `os.replace`.
- При старте сервиса все задачи в `processing`: если `video.mov` на месте и до дедлайна ≥ `MIN_BUDGET_SEC` — снова в очередь, иначе `failed` (`"service_restarted"`).

`.env.example`: `API_KEY`, `ANALYZER=stub|ml`, `DATA_DIR=data`, `MAX_UPLOAD_MB=95`, `MAX_QUEUE=10`, `JOB_DEADLINE_SEC=270`, `ASSEMBLE_RESERVE_SEC=15`, `MIN_BUDGET_SEC=20`, `STUB_DELAY_SEC=1`, `STUB_HANG=false`, `LABELS_XLSX=open/open_labels.xlsx`.

---

## 4. Интерфейс анализатора

```python
# app/analyzers/base.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class Detection:
    type_id: str                  # ключ из norms.yaml
    frame_id: int                 # главный кадр события (абсолютный индекс кадра в видео)
    evidence_frame_ids: list[int] # 1–3 кадра-доказательства
    description: str              # что видно на видео, без домыслов
    confidence: float             # 0..1

@dataclass
class AnalysisResult:
    detections: list[Detection]
    covered: list[tuple[float, float]]           # реально проверенные интервалы, сек
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # decode, detect, vlm, ...
    mode: str = "full"                           # full | cheap (мало бюджета)

class Analyzer(Protocol):
    def analyze(self, video_path: str, meta: dict, budget_sec: float) -> AnalysisResult: ...
```

**Правила сборки результата (`assemble.py`, общий код):**
- Исключение в анализаторе, сбой декодирования, невалидный результат → `failed`. **Никогда** не превращать ошибку в `no_violation`.
- Покрытие < 50% длительности → `failed` (`"insufficient_coverage"`); 50–99% → результат по проверенной части и предупреждение в `warnings` (UI показывает «проанализировано X из Y с»).
- Таймкоды считает **сервер**: `ts = frame_id / fps` по ffprobe. `frame_id` вне `[0, nb_frames)` → деталь отбрасывается с предупреждением; у события без валидных кадров берём главный кадр.
- `type`, `rule` — из `norms.yaml`; неизвестный `type_id` → `unknown_review`.
- Итоговые `warnings` и `timings` сохраняются в `result.json`, но **не** отдаются в API жюри (только в UI и логах).

**`StubAnalyzer`:** берёт `meta["original_filename"]`, нормализует (`Х`→`X`, без расширения), ищет в `LABELS_XLSX`. «Нарушение» → одно событие `unknown_review` с `frame_id = round(t × fps)`; иначе пустой список; `covered` = всё видео. Нет файла разметки или имени в ней → пустой список и предупреждение. Это только проверка контура: «100% на заглушке» ничего не говорит о качестве модели.

---

## 5. `norms.yaml` — справочник с проверкой источников

```yaml
left_cash_unattended:
  title: "Кассир покинул рабочее место, оставив ценности без присмотра"
  observable: "Кассир выходит из кадра/помещения; на столе остаются наличные или открыт сейф"
  conditions: "Ценности видны в кадре в момент ухода"
  exclusions: "Ценности убраны, сейф закрыт до ухода"
  document: "TODO"
  clause: "TODO"
  url: "TODO"
  quote: "TODO"
  enabled: true
  source_verified: false
# unauthorized_person, phone_use, food_drink, personal_items,
# cash_handling, collection — та же структура, пока enabled: true, source_verified: false
unknown_review:
  title: "Признаки возможного нарушения — требуется проверка"
  observable: "-"
  enabled: true
  source_verified: false
```
- Категории — **гипотезы** по датасету. Каждой нужны наблюдаемый признак, условия, исключения и проверенный источник (только акты РК: adilet.zan.kz, НБРК).
- В ответ API `rule` попадает только при `source_verified: true` и без `TODO`; иначе `rule: null`. `enabled: false` — тип вообще не выдаётся.
- `norms.py` при загрузке валидирует структуру и пишет в лог список неподтверждённых записей.

---

## 6. UI (`static/index.html`)

1. Загрузка .mov с прогрессом; ошибки 413/503 показываются понятным текстом.
2. Статус «Обрабатывается…» (опрос раз в 2 с).
3. Итог крупно: «Нарушение не обнаружено» или «Обнаружены признаки возможных нарушений».
4. Карточка на каждое нарушение: тип, таймкод `мм:сс`, описание, кадры (клик — увеличить), норматив или пометка «норматив уточняется».
5. Плеер; клик по таймкоду перематывает видео.
6. Предупреждения анализа (например, неполное покрытие).
7. Дисклеймер: «Результат — подсказка для проверки человеком, не юридическое заключение».

---

## 7. Скрипты

### `scripts/eval.py` — качество
- Прогоняет `open/videos/` через API (URL + ключ), сравнивает с `open_labels.xlsx`.
- Нормализация имён (`Х`→`X`), таймкодов («22 секунда», «1минута 8 секунда»).
- **Дубликаты по SHA-256:** 42 файла = 40 уникальных видео (`Х-4`≡`Х-47`, `Х-5`≡`Х-48`). В отчёте группировать; при будущем обучении или подборе порогов держать дубликаты в одной выборке.
- **Неоднозначность:** у `Х-4`/`Х-47` разметка 57 с и 68 с. Это либо два события, либо неточность разметки — не решать в коде, вынести в отдельную строку отчёта.
- Метрики: точность вердикта, **пропуски**, ложные срабатывания, ошибка таймкода, **число failed/таймаутов** (считаются ошибкой, из итога не исключаются), полное время от отправки до `done` (p50 / max).
- Отчёт: `data/eval_report.csv` + сводка в консоли.

### `scripts/benchmark.py` — время и память
- Для 1–N видео: время загрузки анализатора, декодирования, каждого этапа (`timings`), сборки ответа, полное время; пиковая память процесса-воркера (psutil).
- Режим «очередь»: отправить 5 видео одновременно и проверить, что последнее укладывается в 300 с.
- На заглушке проверяется сам скрипт; реальные числа снимаются на Mac с моделью.

---

## 8. ML-часть (позже, делаю я — для понимания стыков)

- **Железо:** Mac M3, 8 ГБ общей памяти. Одна загруженная VLM, кадры обрабатываются последовательно небольшими наборами.
- **VLM:** `mlx-vlm` + `mlx-community/Qwen3-VL-2B-Instruct-4bit` — кандидат по умолчанию; для сравнения `mlx-community/Qwen3.5-2B-4bit`. Обе Apache-2.0. Скорость не гарантирована — решает `benchmark.py`.
- **Кандидаты событий:** детектор движения → RT-DETRv2 (Apache-2.0) → `roboflow/trackers` (ByteTrack) → зоны `supervision` (стол, сейф, дверь) → правила.
- **Подтверждение:** VLM получает кадры кандидата и возвращает JSON `{type_id, confirmed, frame_ids, description}`; pydantic валидирует; при сбое VLM — шаблонное описание.
- Схема «CV-кандидаты → проверка через VLM» соответствует Alert Verification в NVIDIA VSS и подходу Frigate.

---

## 9. Чего не делать

- **Никаких облачных AI API** — дисквалификация.
- **Никакого распознавания лиц.**
- Видео датасета не класть в git и никуда не выкладывать.
- Не превращать ошибки в `no_violation`.
- Не выдавать норматив без `source_verified: true`.
- Модели — только с лицензией, разрешающей коммерческое использование (Apache-2.0 / MIT). Не берём Ultralytics YOLO (AGPL) и Qwen2.5-VL-3B (некоммерческая).

---

## 10. Готово, когда

- [ ] `uv sync && uv run uvicorn app.main:app` поднимает сервис; `/health` отвечает, `worker_alive: true`.
- [ ] `curl -X POST .../api/v1/analyze -H "X-API-Key: ..." -F "file=@open/videos/X-1.mov"` → `job_id`; затем `done` с кадром ~22 с.
- [ ] Без ключа → 401; неизвестный job → 404; файл > 95 МБ → 413; битый файл → `failed`.
- [ ] `STUB_HANG=true` → задача `failed` по таймауту, воркер перезапущен, следующая задача проходит.
- [ ] Перезапуск сервиса посреди задачи → задача доезжает или становится `failed`, но не висит в `processing`.
- [ ] UI проходит весь сценарий на любом видео датасета.
- [ ] `pytest` зелёный (на синтетическом ролике, без датасета).
- [ ] `eval.py` и `benchmark.py` работают на заглушке.
- [ ] Через публичный URL (cloudflared) загрузка ролика 29 МБ проходит.
- [ ] README: запуск, архитектура, `.env.example`, ограничения.
