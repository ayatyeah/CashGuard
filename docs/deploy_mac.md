# Развёртывание анализатора на Mac (Apple Silicon M3, 8 ГБ, Python 3.11)

Схема на Mac:

```
Go-бэкенд CashGuard (нативно, не в Docker)
  └─ ANALYZER_COMMAND = <repo>/bin/cashguard-analyzer <абсолютный путь к видео>
       └─ .venv/bin/python scripts/cashguard_analyzer.py
            ├─ YOLOX-M/S + RTMPose-m  — onnxruntime, CPUExecutionProvider
            ├─ CLIP ViT-B/32          — torch CPU + transformers
            └─ Qwen3-VL-2B-Instruct 4-bit — mlx-vlm (GPU через Metal), FORTE_VLM_BACKEND=mlx
```

Почему Go-бэкенд нужно запускать нативно. MLX работает только в macOS, а Docker Desktop запускает контейнеры
в Linux-VM без доступа к Metal. Кроме того, контейнер `backend` из `compose.yaml` не содержит Python-окружения.
PostgreSQL при этом можно оставить в Docker.

Совместимость `app/ml/vlm.py` с **mlx-vlm 0.7.8** (последний релиз на PyPI) проверена по исходному коду пакета:
`load(id) -> (model, processor)`, `apply_chat_template(processor, model.config, prompt, num_images=1) -> str`,
`generate(model, processor, prompt, [PIL.Image], max_tokens=..., verbose=False) -> GenerationResult` (текст в `.text`).
По умолчанию `temperature=0.0`, то есть жадное декодирование, как `do_sample=False` в transformers-ветке.

---

## 1. Требования

- Mac на Apple Silicon (M1–M4), **macOS 14 Sonoma или новее**. Колёса onnxruntime 1.31, torch 2.14 и mlx 0.32
  собраны только под `macosx_14_0_arm64` и новее.
- Нативный arm64-терминал, не Rosetta. Проверка: `uname -m` должен вывести `arm64`.
- Около 6 ГБ на диске: окружение (~2 ГБ), веса (~2,6 ГБ), кэш загрузок uv/HF.
- Homebrew: `brew install uv go`. Node.js нужен только для фронтенда: `brew install node`.
  ffmpeg нужен только для синтетического видео в smoke-тесте: `brew install ffmpeg`.
  `curl` и `unzip` есть в macOS.

## 2. Окружение Python 3.11 (uv)

```bash
cd ~/ForteWin                      # корень репозитория
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
```

- Используйте `requirements-mac.txt`, а не `requirements-ml.txt` с `--extra-index-url https://download.pytorch.org/whl/cpu`.
  На macOS torch ставится с обычного PyPI: это сборка CPU+MPS, другой не нужно.
- `requirements-mac.txt` содержит те же пакеты, что `requirements-ml.txt`, с версиями из проверенного Linux-окружения,
  плюс `mlx-vlm==0.7.8` и `mlx==0.32.4`. Разрешение зависимостей проверено командой `uv pip compile` для
  `aarch64-apple-darwin` / macOS 14 / CPython 3.11: 90 пакетов, все ставятся из готовых колёс, ничего не компилируется.
- `mlx-vlm` дополнительно тянет `mlx-audio`, `datasets`, `fastapi`, `opencv-python` и другие пакеты. Это нормально.
  Модуль `cv2` при этом дают три пакета (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`).
  Поэтому все три закреплены на одной версии 5.0.0.93.

Проверка установки:

```bash
.venv/bin/python -c "import platform, onnxruntime as ort, torch, mlx.core as mx, mlx_vlm, cv2; \
print(platform.machine(), 'ort', ort.__version__, ort.get_available_providers()); \
print('torch', torch.__version__, 'mps', torch.backends.mps.is_available()); \
print('mlx', mx.default_device(), 'mlx-vlm', mlx_vlm.__version__, 'cv2', cv2.__version__)"
```

Ожидается: `arm64`, onnxruntime 1.31.0 со списком провайдеров, где есть `CPUExecutionProvider`
(в колесе для macOS обычно есть и `CoreMLExecutionProvider`), `mlx Device(gpu, 0)`, `mlx-vlm 0.7.8`.

### onnxruntime: CPU или CoreML EP

**Оставляем CPU EP.** Код жёстко задаёт `providers=["CPUExecutionProvider"]` для YOLOX (`app/ml/perception.py`)
и `device="cpu"` для RTMPose (rtmlib). Пороги `models/scorer.json` и `models/track_model.json` обучены на выходах
CPU, и CPU EP на arm64 даёт те же числа. CoreML EP работает в FP16 и может разбить граф на части.
Кроме того, он компилирует модель при каждом создании сессии, а анализатор стартует новым процессом на каждое видео.
Переход на CoreML требует правки кода (rtmlib умеет `device="mps"`, тогда она выбирает CoreML EP)
и повторной проверки на train-видео. Отдельный пакет `onnxruntime-silicon` не нужен.

## 3. Веса моделей

### 3.1. Детекторы, поза, CLIP

```bash
source .venv/bin/activate          # скрипт вызывает `python` из PATH, ему нужен huggingface_hub из .venv
bash scripts/download_models.sh
```

Результат: `models/yolox_m.onnx` (97 МБ), `models/yolox_s.onnx` (35 МБ), `models/rtmpose_m/.../end2end.onnx` (52 МБ).
CLIP `openai/clip-vit-base-patch32` (605 МБ, MIT) попадает в кэш Hugging Face (`~/.cache/huggingface/hub`).

### 3.2. VLM для MLX (скрипт выше её **не** скачивает)

Модель проверена на Hugging Face Hub:

| Параметр | Значение |
|---|---|
| repo id | `mlx-community/Qwen3-VL-2B-Instruct-4bit` (совпадает с `FORTE_VLM_MLX` по умолчанию в `app/ml/vlm.py`) |
| лицензия | Apache-2.0, как у исходной `Qwen/Qwen3-VL-2B-Instruct` |
| размер | 1,80 ГБ (`model.safetensors` 1,78 ГБ + токенизатор и конфиги) |
| квантизация | 4 бит affine, group 64 (языковая часть); vision-энкодер в bf16 |
| происхождение | конвертирована mlx-vlm 0.3.4, `model_type: qwen3_vl`, не gated |
| ревизия (проверено) | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` (2025-10-21) |

**Вариант A: кэш HF, ничего настраивать не нужно** (тот же набор файлов, что качает `mlx_vlm.load`):

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
print(d('mlx-community/Qwen3-VL-2B-Instruct-4bit', allow_patterns=['*.json','*.jsonl','*.safetensors','*.py','*.model','*.tiktoken','*.txt','*.jinja']))"
```

В этом варианте не указывайте `revision=<sha>`. Анализатор работает офлайн (`HF_HUB_OFFLINE=1`) и ищет модель по ссылке
`refs/main` в кэше. При загрузке только по sha эта ссылка не создаётся, и офлайн-загрузка падает.

**Вариант B: папка внутри `models/`, не зависит от `$HOME`.** Удобно, если Go-сервер запускается от другого
пользователя или через launchd. `models/*` уже в `.gitignore`.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
d('mlx-community/Qwen3-VL-2B-Instruct-4bit', revision='9c4f5209e57b31f4b9dfba735de3fb983739c9cc', \
local_dir='models/qwen3-vl-2b-instruct-4bit', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])"
export FORTE_VLM_MLX="$PWD/models/qwen3-vl-2b-instruct-4bit"   # mlx_vlm.load принимает локальный путь
```

Не скачивайте на Mac `Qwen/Qwen3-VL-2B-Instruct` (4,3 ГБ bf16). Она нужна только transformers-ветке на CPU
и на 8 ГБ уходит в swap.

## 4. Переменные окружения

| Переменная | На Mac | Смысл |
|---|---|---|
| `FORTE_VLM_BACKEND` | **`mlx`** (обязательно) | По умолчанию `transformers`. Без этой переменной анализатор пытается загрузить bf16-модель на CPU, а если её нет, тихо отключает VLM (в stderr `VLM disabled: …`) и работает только на CV |
| `FORTE_USE_VLM` | `1` (по умолчанию) | `0` — режим только CV: без VLM, примерно на 2 ГБ меньше памяти, но без VLM-признаков |
| `FORTE_VLM_MLX` | не задавать (вариант A) или абсолютный путь (вариант B) | id или путь MLX-модели |
| `ANALYZER_BUDGET_SEC` | `240` (по умолчанию) | Бюджет `analyze()` в секундах: из него выводится частота кадров (до 2/с) и остаток времени на VLM. Загрузка моделей (~10–20 с) и кодирование кадров-доказательств в бюджет не входят. Go-бэкенд убивает анализатор через 15 минут |
| `FORTE_COST_PER_FRAME` | `0.6` (по умолчанию) | Оценка секунд CPU на кадр, по ней выбирается fps. Значение измерено на Ryzen 5 5600H; на M3 его стоит перемерить (см. §7) |
| `FORTE_VLM_MAX_SIDE` | `640` (по умолчанию) | Длинная сторона склейки кадров для VLM. При меньшем значении меньше токенов, памяти и времени, но признаки отличаются от тех, на которых обучался скорер |
| `CASHGUARD_PYTHON` | не задавать | Python для лаунчера, по умолчанию `<repo>/.venv/bin/python` |
| `FORTE_MODELS_DIR` | не задавать | Каталог ONNX-весов, по умолчанию `<repo>/models` |
| `HF_HOME` | только при необходимости | Если сервер работает с другим `$HOME`, укажите каталог кэша, куда скачаны CLIP и VLM |

`HF_HUB_OFFLINE=1` код выставляет сам: во время анализа сеть не используется.

## 5. Подключение к Go-бэкенду (`ANALYZER_COMMAND`)

Адаптер (`src/backend/internal/infrastructure/analyzer.go`) запускает `exec.CommandContext(ctx, ANALYZER_COMMAND, <путь к видео>)`
без shell и без `cmd.Env`. Поэтому **анализатор наследует окружение Go-сервера**: переменные `FORTE_*`
нужно экспортировать до запуска сервера. stderr анализатора Go не сохраняет, поэтому отлаживайте запуском вручную (§7).

```bash
# PostgreSQL: из compose (в Docker Desktop → Settings → Resources ограничьте память VM до ~1–1.5 ГБ)
docker compose up -d db            # или: brew install postgresql@17 (без VM, экономнее на 8 ГБ)

cd ~/ForteWin
chmod +x bin/cashguard-analyzer scripts/download_models.sh
export DATABASE_URL='postgres://cashguard:ПАРОЛЬ_ИЗ_.env@127.0.0.1:55432/cashguard?sslmode=disable'
export API_KEY='ключ_из_.env_не_короче_16_символов'
export DEMO_MODE=false
export ANALYZER_COMMAND="$PWD/bin/cashguard-analyzer"      # абсолютный путь
export FORTE_VLM_BACKEND=mlx
export FORTE_USE_VLM=1
export ANALYZER_BUDGET_SEC=240
go -C src/backend run ./cmd/server                          # API: http://127.0.0.1:8080/api
```

Если не хотите держать `FORTE_*` в окружении сервера, сделайте свою обёртку вне репозитория,
например `~/bin/cashguard-analyzer-mac`, и укажите её в `ANALYZER_COMMAND`:

```bash
#!/usr/bin/env bash
export FORTE_VLM_BACKEND=mlx FORTE_USE_VLM=1 ANALYZER_BUDGET_SEC=240
exec "$HOME/ForteWin/bin/cashguard-analyzer" "$@"
```

Фронтенд. Контейнер `frontend` из compose проксирует `/api` на `backend:8080` внутри Docker-сети,
поэтому с нативным бэкендом он работать не будет. Используйте Vite: `npm --prefix src/frontend install && npm --prefix src/frontend run dev`.
Vite по умолчанию проксирует на `127.0.0.1:8080`. Другой вариант — отдать `src/frontend/dist` любым веб-сервером
с прокси `/api` → `127.0.0.1:8080`.

`bin/cashguard-analyzer` — простой bash-скрипт, совместимый с системным bash 3.2 в macOS.
Он пишет в stdout только JSON по контракту, а логи — в stderr.

## 6. Память на 8 ГБ

Оценка пика процесса-анализатора (не замер на Mac; замерьте по §7):

| Компонент | Примерно |
|---|---|
| Qwen3-VL-2B 4-bit в MLX (веса 1,8 ГБ + активации/KV для одной склейки ≤640 px) | 2,0–2,4 ГБ |
| torch + CLIP ViT-B/32 (fp32) | ~0,9 ГБ |
| onnxruntime: YOLOX-M, YOLOX-S, RTMPose-m | 0,4–0,6 ГБ |
| Кадры в памяти: ½ от 960 px + кропы рабочей зоны; 5 мин видео при 2 к/с | 0,3–0,6 ГБ |
| Python, OpenCV, прочее | ~0,3 ГБ |
| **Итого** | **≈ 4–4,5 ГБ** |

Плюс сама macOS (2–3 ГБ), Go-сервер с PostgreSQL и VM Docker Desktop, если она запущена. Запас небольшой, поэтому:

- Анализатор должен работать в одном экземпляре. В Go 1 worker; не запускайте ручной smoke-тест, пока сервер обрабатывает задание.
- Не используйте transformers-бэкенд VLM на Mac (`FORTE_VLM_BACKEND` обязательно `mlx`).
- Закройте браузеры и IDE. Ограничьте память Docker Desktop или поставьте PostgreSQL через brew.
- Следите за давлением памяти: Activity Monitor → Memory, swap должен оставаться около нуля.
- Если давление жёлтое или красное: сначала `FORTE_VLM_MAX_SIDE=512` (меньше визуальных токенов), затем `FORTE_USE_VLM=0`.
  Вариант `-3bit` (1,57 ГБ) экономит всего ~0,2 ГБ, потому что vision-энкодер остаётся в bf16. Вариант `-8bit` (2,64 ГБ)
  на 8 ГБ не рекомендуется.
- Первый вызов MLX после старта процесса медленнее: компилируются Metal-ядра.

## 7. Smoke-тест

Все команды выполняются из корня репозитория. **Не используйте `data/test`**: это отложенная выборка только
для финальной оценки. Для проверок берите синтетическое видео или `data/train/videos`. У `scripts/evaluate.py`
по умолчанию `--split test`, поэтому на Mac запускайте его только с `--split train`.

**7.1. VLM отдельно** (синтетическая картинка, без видео):

```bash
FORTE_VLM_BACKEND=mlx .venv/bin/python - <<'EOF'
import time, numpy as np, cv2
from app.ml.vlm import VLM, Q_DESK
img = np.full((360, 640, 3), 200, np.uint8); cv2.rectangle(img, (200, 150), (440, 300), (40, 120, 40), -1)
t = time.time(); v = VLM(); print("load", round(time.time() - t, 1), "s")
t = time.time(); print(v.ask_flags([img], Q_DESK), round(time.time() - t, 1), "s")
EOF
```

Ожидается словарь с булевыми ключами из `Q_DESK` (может быть заполнен не полностью) и время ответа в секундах. Ошибка `LocalEntryNotFoundError`
или похожая означает, что модель не скачана (§3.2) или кэш лежит в другом `HF_HOME`.

**7.2. Анализатор целиком, так же, как его вызывает Go:**

```bash
ffmpeg -loglevel error -f lavfi -i testsrc=duration=20:size=1280x720:rate=25 -pix_fmt yuv420p /tmp/cg_smoke.mp4
FORTE_VLM_BACKEND=mlx ANALYZER_BUDGET_SEC=120 /usr/bin/time -l bin/cashguard-analyzer /tmp/cg_smoke.mp4 > /tmp/cg_smoke.json
.venv/bin/python -c "import json; d=json.load(open('/tmp/cg_smoke.json')); print(d['status'], d['verdict'], len(d['violations']))"
```

Ожидается `done no_violation 0`. В stderr будет строка `timings=… warnings=[… рабочее место кассира не найдено …]`:
для синтетики это нормально. В выводе `/usr/bin/time -l` смотрите `peak memory footprint`.
Затем то же на реальном видео: `bin/cashguard-analyzer "$PWD/data/train/videos/<файл>.mov" > /tmp/r.json`.
В stderr не должно быть `VLM disabled`, а в `timings` должно быть `vlm_moments` > 0.

Калибровка `FORTE_COST_PER_FRAME` на M3: `.venv/bin/python scripts/analyze_video.py data/train/videos/<файл>.mov --out /tmp/r.json`.
В `_timings` возьмите `(perception + clip) / frames`, это фактическая стоимость кадра. Если она заметно меньше 0,6,
экспортируйте найденное значение: анализатор будет брать больше кадров в том же бюджете.

**7.3. Через Go-бэкенд:**

```bash
curl -H "X-API-Key: $API_KEY" -F "video=@/tmp/cg_smoke.mp4" http://127.0.0.1:8080/api/jobs     # → {"job_id": ...}
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8080/api/jobs/JOB_ID                              # → status: done
```

## 8. Типичные проблемы

| Симптом | Причина и решение |
|---|---|
| `No matching distribution` для `mlx`/`onnxruntime`/`torch` | Python x86_64 (Rosetta) или macOS < 14. Проверьте `uname -m`, `.venv/bin/python -c "import platform; print(platform.machine())"` и `sw_vers` |
| Задание в Go падает с «локальный анализатор завершился с ошибкой» | Запустите `bin/cashguard-analyzer <видео>` вручную с тем же окружением и посмотрите stderr |
| В результате мало нарушений, в stderr `VLM disabled: …` | Не задан `FORTE_VLM_BACKEND=mlx`, не установлен `mlx-vlm` или не скачана модель |
| `OMP: Error #15 … libomp.dylib already initialized` | Несколько колёс несут свой libomp. Обходной путь: `export KMP_DUPLICATE_LIB_OK=TRUE` |
| Предупреждение «мало времени: кадры анализируются с частотой …» | Увеличьте `ANALYZER_BUDGET_SEC` (Go ждёт до 15 мин) или откалибруйте `FORTE_COST_PER_FRAME` (§7) |
| Swap растёт, всё тормозит | §6: один анализатор, без transformers-VLM, `FORTE_VLM_MAX_SIDE=512` или `FORTE_USE_VLM=0` |

Известное расхождение бэкендов. В mlx-ветке `VLM.ask` системный промпт склеивается с вопросом в одно
user-сообщение, а в transformers-ветке он идёт отдельной ролью `system`. Скорер обучался на ответах transformers-ветки,
поэтому перед выводами о качестве сравните ответы MLX на нескольких train-видео (`scripts/run_vlm.py --split train`
с `FORTE_VLM_BACKEND=mlx`). Ответы сохраняются в `data/vlm/<видео>.json`, а видео с уже существующим файлом
скрипт пропускает. Поэтому старые файлы сначала перенесите в другую папку: так их будет с чем сравнить.
