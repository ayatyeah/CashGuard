# 09. Производительность и надёжность на Mac M3 Pro 18 ГБ (трек 9/10)

Цель: 145-секундное видео (и очередь из 5) с включённым VLM должно укладываться в 5-минутный лимит жюри.
Исходная точка — `app/ml/perception.py` (YOLOX-M/S + RTMPose-m, onnxruntime **CPU EP**, `threads = cpu_count()-2`),
`app/analyzers/ml.py` (2 к/с, адаптивное прореживание, CLIP ViT-B/32 на torch CPU, VLM в остатке бюджета),
`docs/deploy_mac.md` (Go спавнит **процесс на каждое видео**, бюджет 240 с, mlx-vlm Qwen3-VL-2B 4-bit).

Где число взято из источника — стоит ссылка; где это расчёт/экстраполяция — пометка **(оценка)**. Ничего не запускалось.
Веб-поиск был ограничен общим лимитом сессии, поэтому часть пунктов (например, скорость декодирования видео) оценена
по спецификациям, а не по замерам; для каждой такой позиции дана команда проверки.

---

## 1. Железо и из чего складывается время

**M3 Pro (12-ядерный вариант):** 6 P-ядер (до 4,06 ГГц) + 6 E-ядер, GPU 18 ядер, пропускная способность памяти
**~150 ГБ/с** (153,6), в биннинге 11-ядерном — 5P+6E и 14 GPU-ядер
([Wikipedia: Apple M3](https://en.wikipedia.org/wiki/Apple_M3), [Apple Newsroom](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/)).
Для сравнения: M1 Max 400 ГБ/с, M4 Max ~546 ГБ/с — все найденные замеры VLM на MLX сделаны на Max-чипах,
поэтому их приходится масштабировать вниз в ~2,5–3,5 раза (по памяти и по GPU-ядрам).

Три независимых вычислителя, которые можно нагружать **одновременно**: CPU (ORT CPU EP, OpenCV), GPU/Metal (MLX, torch MPS,
CoreML GPU), Neural Engine (только через CoreML). Сейчас пайплайн использует CPU, а потом GPU, последовательно.

### 1.1. Объём работы на 145 с видео (2 к/с → 290 кадров, 464×848 → max_side 960, т.е. без апскейла 464×848)

| Этап | Операции на кадр | Сейчас (CPU EP, оценка на M3 Pro) | После оптимизаций |
|---|---|---|---|
| Старт процесса: import torch/transformers/cv2/ort | — | 5–10 с на видео | 0 (постоянный сервер) |
| Загрузка моделей: ORT ×3 (~1–2 с), CLIP (~2–4 с), VLM 2B (3–5 с; 4B 6–10 с) + компиляция Metal-ядер при первом вызове | — | 10–20 с на видео (`deploy_mac.md` §4) | 0 (прогрев один раз) |
| Декодирование H.264 464×848, 4350 кадров (`grab()` декодирует каждый кадр) | — | 5–10 с **(оценка)**, в критическом пути | ~0 (поток-префетчер) |
| YOLOX-M 640 (73,8 GFLOPs) | 1 | 120–200 мс → 35–60 с | CoreML MLProgram: 25–50 мс → 8–15 с **(оценка по [Xybrid] 6,8× для CNN)** |
| YOLOX-S на кропе рабочей зоны 640 (26,8 GFLOPs) | 1 | 50–80 мс → 15–25 с | 12–25 мс → 4–8 с |
| RTMPose-m 256×192 (~2,2 GFLOPs на человека) | 1–3 | 15–25 мс/чел → 8–15 с | на CPU оставить (или CoreML 2–3× при полном размещении) |
| CLIP ViT-B/32 (4,4 GFLOPs), батч 32 | 1 | 20–30 мс → 6–10 с | torch MPS / MLX: 1–2 с |
| Склейка: resize, NMS, numpy-декод головы 8400×85, cvtColor | — | 10–15 мс → 3–5 с | то же |
| **Perception + CLIP итого** | | **0,26–0,43 с/кадр → 75–125 с** | **0,08–0,14 с/кадр → 20–40 с** |

Сопоставление: на Ryzen 5 5600H измерено 0,6 с/кадр (`FORTE_COST_PER_FRAME`). Пиковая FP32-производительность 6 P-ядер M3
(4×128-бит FMA/такт) сопоставима с 6 ядрами Zen3 AVX2, ORT MLAS не использует AMX/Accelerate, поэтому ждать на CPU больше
~1,5–2× выигрыша не стоит; это и даёт диапазон 0,3–0,45 с/кадр **(оценка)**. Замерить: `scripts/analyze_video.py … --out /tmp/r.json`
→ `(perception+clip)/frames` (`deploy_mac.md` §7).

### 1.2. VLM: сколько стоит один вызов

Токенизация изображения в Qwen3-VL: патч 16 px, merge 2×2 → **один визуальный токен = 32×32 px**; размеры округляются до
кратных 32 ([QwenLM/Qwen3-VL README](https://github.com/qwenlm/qwen3-vl), `preprocessor_config.json` модели
[mlx-community/Qwen3-VL-4B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/raw/main/preprocessor_config.json):
`patch_size=16, merge_size=2, shortest_edge=65536 px², longest_edge=16777216 px²`, т.е. **процессор по умолчанию не уменьшает**
картинку до 16 Мпикс — уменьшение делает только наш `compose(max_side=640)`).

| Вход | Визуальных токенов | Комментарий |
|---|---|---|
| Текущая склейка 3 кадров 464×848 → 1392×848 → **640×390** | ~20×12 = **240** | уже экономно |
| 1 Мпикс (1024×1024) | **1024** | ×4,3 к prefill |
| 2 Мпикс | **~2048** | ×8,5 к prefill, растёт и KV |
| Кроп стола 512×512 вместо всего кадра | 256 | детальнее при той же цене — предпочтительно вместо роста разрешения |

Замеры на MLX (все на Max-чипах, пересчёт на M3 Pro — **оценка** ÷2,5–3,5):

- vllm-mlx, **M4 Max 128 ГБ**, Qwen3-VL-**4B**: холодный запрос с картинкой 224² — 0,8 с, 448² — 1,2 с, 768² — 1,8 с, 1024² — 2,1 с;
  с кешем vision-эмбеддингов повторный запрос 0,12–0,16 с (6,7–13×). Qwen3-VL-**8B**, 1024²: 21,7 с холодный (в статье — время
  всего хода, включая генерацию), 1,15 с с кешем (19×), 0,78 с при 3+ ходе (28×); из ускорения 7,8× даёт кеш vision, 1,2× — KV
  ([arXiv 2601.19139](https://arxiv.org/html/2601.19139v2), табл. 2–5).
- vllm-mlx video benchmark: Qwen3-VL-**8B-4bit** на **M4 Max** 2 кадра → 4,48 с / 57 tok/s / 6,4 ГБ; на **M1 Max (400 ГБ/с)**
  4 кадра → 8,84 с / 29 tok/s; Qwen3-VL-**4B-3bit** на M1 Max 4 кадра → 5,09 с / 29,5 tok/s
  ([vllm-mlx/docs/benchmarks/video.md](https://github.com/waybarrios/vllm-mlx/blob/main/docs/benchmarks/video.md)).
- Декод ограничен памятью: для M3 Pro 18 ГБ оценка 4B Q4 ≈ **48 tok/s**, 8B Q4 ≈ **25 tok/s**
  ([llmcheck.net](https://llmcheck.net/benchmarks), расчёт из полосы 150 ГБ/с; 2,3 ГБ весов языковой части 4B → теоретически 65 tok/s).
- Prefill mlx-vlm на M4 Max: ~245 tok/s на текстовом промпте 1300 токенов
  ([mlx-vlm #717](https://github.com/Blaizzy/mlx-vlm/issues/717)); на M4 Pro для 27B 4-bit — 55–108 tok/s **в зависимости от нагрева**
  ([kortexa-ai/models.server #22](https://github.com/kortexa-ai/models.server/issues/22)).

Оценка одного вызова `VLM.ask` (склейка 640 px, ~240 vis + ~150 text токенов, 80 выходных) на **M3 Pro**:

| Модель (4-bit) | Prefill + vision | Декод 80 токенов | Итого на вызов | 3 вызова (2-из-3) | Память |
|---|---|---|---|---|---|
| Qwen3-VL-2B (сейчас) | 1–1,5 с | ~1 с (70–80 tok/s) | **2,5–3,5 с** | 8–10 с | 2,0–2,4 ГБ |
| Qwen3-VL-4B | 2–3 с | ~2 с (40–45 tok/s) | **4,5–6 с** | 14–18 с; с кешем vision+KV на 2-й/3-й ≈ 8–10 с | 3,1 ГБ веса + 0,5–0,8 |
| Qwen3-VL-8B | 4–6 с | ~3,5 с (20–25 tok/s) | **9–12 с** | 27–36 с; с кешем ≈ 16–20 с | 5,7–5,8 ГБ веса + ~1 |
| Любая, картинка 1 Мпикс | ×3–4 к prefill | — | 4B: 8–12 с | — | +KV ~0,15 ГБ/картинку |

Вывод: при бюджете 240 с и perception 20–40 с остаётся ~180 с на VLM → ~35–40 вызовов 4B или ~15–20 вызовов 8B
(без кеша). 8B имеет смысл только с постоянным сервером и кешем vision-эмбеддингов; по умолчанию — **4B**.

### 1.3. Очередь из 5 видео — главный риск

`scripts/cashguard_analyzer.py` вычитает время ожидания в очереди из 255 с, Go-воркер один (`usecase/service.go`).
Если жюри загрузит 5 видео **одновременно**, у k-го видео бюджет = 255 − (k−1)·T_job. Чтобы все 5 получили полный анализ,
нужно **T_job ≲ 50 с**, иначе 4-е и 5-е уйдут в режим `cheap`/CV-only (не упадут — но потеряют VLM-признаки).
Если загрузка последовательная (следующее после ответа) — хватает T_job ≤ 240 с. Отсюда два требования к архитектуре:
убрать ~20–30 с фиксированных накладных на видео и **совмещать** perception (CPU/ANE) видео N+1 с VLM (GPU) видео N.

---

## 2. CoreML EP для YOLOX / RTMPose: что ждать и подводные камни

Факты из документации [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html):

- Опции (строковые, в питоне `providers=[("CoreMLExecutionProvider", {...}), "CPUExecutionProvider"]`): `ModelFormat`
  (`NeuralNetwork` **по умолчанию** / `MLProgram`), `MLComputeUnits` (`ALL` по умолчанию / `CPUOnly` / `CPUAndGPU` /
  `CPUAndNeuralEngine`), `RequireStaticInputShapes` (0/1), `ModelCacheDirectory` (без него **компиляция при каждом создании
  сессии**, «до минут» для сложных моделей), `ProfileComputePlan=1` (логирует, на каком устройстве исполняется каждый оператор),
  `SpecializationStrategy=FastPrediction`, `AllowLowPrecisionAccumulationOnGPU`.
  Строковые опции доступны с ORT 1.21 ([roboflow/inference PR #3082](https://github.com/roboflow/inference/pull/3082)); у нас 1.31.
- Кеш ключуется хешем пути модели (или `CACHE_KEY` в метаданных) и **не валидируется** — при обновлении .onnx чистить каталог.
  Пример из RF-DETR: первая загрузка ~20 с (компиляция), с кешем 0,2 с; MLProgram сократил число партиций с 97 до 13.
- Поддержка операторов (MLProgram): Conv, Resize (с ограничениями), Sigmoid, Mul, Concat, Transpose, Reshape, Slice (константные
  starts/ends/steps), Split, MaxPool 2D, Softmax, MatMul, Gemm (B константа), LayerNormalization (только MLProgram), ArgMax.
  **Не** в списке: Exp, Where, Upsample (старый), Gather для MLProgram.
- Крэш CoreML EP на macOS 15 (`Cannot provide zero compute device types`) был в 1.19.2 и исправлен PR #22277 (окт. 2024)
  ([onnxruntime #22275](https://github.com/microsoft/onnxruntime/issues/22275)) — 1.31 не затронут.

**Числа из реальных отчётов** (все — другие модели, ориентир по классу):

| Источник | Железо | Модель | CPU EP | CoreML EP | Вывод |
|---|---|---|---|---|---|
| [Xybrid blog](https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml) | M2 MBP | MobileNet v2 224² (CNN, статич. форма) | 12,4 мс | 1,8 мс | **6,8×**; первая компиляция 1–5 с |
| там же | M2 | MNIST (крошечная) | 0,15 | 0,8 | 0,2× — накладные ANE |
| там же | M2 | Kokoro 82M (динамич. формы) | 280 | 290 | ~1×, перекомпиляция на новую форму |
| [SWHL blog](https://swhl.github.io/main/blog/accelerate-onnx-models-with-coreml-on-macos/) | Apple Silicon, ORT 1.22 | MobileNetV3-small 224² | 5,2 мс | 7,5 мс | **−30 %**; нативный CoreML (coremltools) — 0,31 мс (20× быстрее ORT-CoreML) |
| [ym2132](https://ym2132.github.io/ONNX_MLProgram_NN_exploration) | Mac GPU | классификатор | — | — | формат `NeuralNetwork` **молча исполняет FP16** на GPU (расхождение ~3,7e-4, 2 предсказания из 464 перевернулись у порога 0,5); `MLProgram` сохраняет FP32 |
| [onnxruntime #17033](https://github.com/microsoft/onnxruntime/issues/17033) | M1 Pro | FP16-модель | — | в 8× медленнее FP32 | не конвертировать веса в FP16 самим |
| [onnxruntime #16934](https://github.com/microsoft/onnxruntime/issues/16934) | Intel Mac | Gemm/Relu | 78 мс | 163 мс | флаг CPU-only у CoreML; смотреть verbose-лог `All nodes placed on [CoreMLExecutionProvider]` |
| [onnxruntime #21227](https://github.com/microsoft/onnxruntime/issues/21227) | M1 | YOLOv9 динамический batch | ок | SIGSEGV | фиксировать batch=1 |

Прогноз для наших графов (по операторам, **нужно подтвердить `ProfileComputePlan`**):

- **YOLOX-M/S (официальный экспорт, сырая голова).** Focus = 4 Slice со `steps=2` (константы) + Concat; SiLU = Sigmoid·Mul;
  SPP = MaxPool 5/9/13 stride 1; PAFPN Upsample nearest → Resize; голова: Conv, Sigmoid, Reshape/Transpose/Concat
  ([YOLOX network_blocks.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py)).
  Exp мы делаем в numpy. Все операторы из списка поддержки MLProgram, вход статический 1×3×640×640 → ожидаю **одну партицию**
  и ускорение класса MobileNet/ResNet (3–6×). Кроп-детектор YOLOX-S тоже 640×640 (паддинг) — форма статична.
- **RTMPose-m (SimCC, голова RTMCC/GAU).** В голове ScaleNorm (ReduceL2/Div), `bmm(q, kᵀ)` между активациями, `relu²`
  (Pow), RoPE через sin/cos (при экспорте с фиксированным seq_len сворачивается в константы), Split, Pad
  ([mmpose rtmcc_block.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/rtmcc_block.py)).
  Вероятны 2–5 партиций с CPU-фолбэком на голове; бэкбон CSPNeXt ляжет на CoreML. Выигрыш неопределённый (общий вклад этапа
  всего 8–15 с), поэтому **RTMPose оставить на CPU EP** в первой итерации. Внимание: `rtmlib(device="mps")` включает CoreML EP
  с настройками по умолчанию = `NeuralNetwork` = **FP16**; если переводить — создавать сессию самим с `MLProgram`.
- **Численная совместимость с обученными порогами.** Пороги `scorer.json`/`track_model.json` обучены на выходах CPU FP32.
  `MLProgram` + `MLComputeUnits=CPUAndGPU` сохраняет FP32 (ym2132), `ALL` отдаёт часть на ANE, который считает **только FP16**.
  План: сначала `CPUAndGPU` (быстро и численно безопасно), затем `ALL` с регрессией на train-видео (сравнить боксы/события,
  `scripts/evaluate.py --split train`). Разница в FP16 — порядка 1e-3 по score, что для порогов 0,25/0,15 почти всегда безвредно,
  но проверять надо.
- Конвертация в **нативный CoreML (coremltools)** даёт ещё в разы быстрее ORT-CoreML (SWHL: 20×), но требует вторую реализацию
  препроцессинга и декода, зависимость от `coremltools` (BSD-3) и валидацию; в рамках дататона — не стоит. **MLX-порт YOLOX/RTMPose**
  тем более (нет готовых весов, нужно писать модель руками).

---

## 3. Остальные компоненты

**Потоки ORT.** Сейчас `threads = cpu_count()-2 = 10` на 12-ядерном M3 Pro — это захватывает E-ядра. На M1 Max 8 потоков
(= P-ядра) были быстрее дефолта «все ядра» на 16–27 %; на M4 переход с 4 на min(ядер,8) дал −26 % геомеан, но на мелких
тензорах +66 % (накладные синхронизации) ([numen #403](https://github.com/jiva-studio/numen/issues/403),
[PR #423](https://github.com/jiva-studio/numen/pull/423)). Рекомендация: `intra_op_num_threads = sysctl hw.perflevel0.logicalcpu`
(**6**), и столько же `torch.set_num_threads`. Для RTMPose (мелкие тензоры, 1–3 кропа) — 2–4 потока.

**Декодирование видео.** `opencv-python` на macOS собран с FFmpeg (программный декодер) и AVFoundation; `CAP_PROP_HW_ACCELERATION`
работает только с бэкендами FFmpeg/GStreamer/MSMF ([OpenCV videoio hwaccel](https://docs.opencv.org/4.13.0/db/dc4/group__videoio__hwaccel.html)).
Для 464×848 H.264 программный декод на одном P-ядре даёт сотни кадров/с **(оценка; замер:** `ffmpeg -benchmark -i v.mov -f null -`
и то же с `-hwaccel videotoolbox`**)** → 4350 кадров ≈ 5–10 с. Выигрыш от VideoToolbox (разгрузка CPU) мал; гораздо проще вынести
декод в фоновый поток с очередью 4–8 кадров — cv2 отпускает GIL, и декод полностью перекроется инференсом. На 2 к/с нужно
декодировать все кадры (P-кадры), seek по ключевым не поможет; `grab()` без `retrieve()` уже экономит конвертацию цвета.

**CLIP.** Одна строка `.to("mps")` в `ClipEmbedder` + батч 32: 290 картинок за 1–2 с вместо 6–10 **(оценка**; GPU M3 даёт
~2,5 TFLOPS против 1,4 на CPU в матричных тестах [arXiv 2502.05317](https://arxiv.org/html/2502.05317v1)**)**. Вариант MLX
([ml-explore/mlx-examples/clip](https://github.com/ml-explore/mlx-examples), [harperreed/mlx_clip](https://github.com/harperreed/mlx_clip))
убирает torch из процесса (−0,9 ГБ), но это 2–3 часа работы и проверка эмбеддингов против обученных линейных проб; на потом.

**Батчинг детекций.** Статический batch N для YOLOX требует переэкспорта; на CPU выигрыш 10–20 %, на ANE заметнее, но
усложняет адаптивное прореживание. RTMPose в rtmlib гоняет кропы по одному — батч 2–3 человека экономит ~мс. Низкий приоритет.

**Стратегия выборки кадров.** Скорер обучен на признаках 2 к/с — менять частоту рискованно (CV по кабинетам). Безопасная экономия:
`motion` уже считается в `Perception.frame`; при `motion < ε` и неизменном числе людей пропускать YOLOX-S-кроп и RTMPose
(копировать прошлые), сохраняя YOLOX-M на каждом кадре. Экономия до 30 % на статичных участках; проверять на train.

**multiprocessing.** На macOS `fork` небезопасен после инициализации системных фреймворков (libdispatch) — стандартный
метод с Python 3.8 `spawn` ([docs.python.org](https://docs.python.org/3/library/multiprocessing.html),
[bpo-33725](https://bugs.python.org/issue33725)). Не форкать процесс после создания ORT-сессий / Metal-контекста. Внутри одного
процесса достаточно потоков: ORT, cv2, MLX, torch отпускают GIL. Если нужны отдельные процессы (perception и VLM) — запускать
их независимо и общаться по Unix-сокету.

**Память MLX.** `mx.set_cache_limit`, `mx.clear_cache()` после каждого видео (буферный кеш MLX иначе держит пик);
`mx.set_wired_limit` — только macOS 15+, нельзя выше системного лимита (`mx.device_info()["max_recommended_working_set_size"]`),
который по умолчанию ≈ 75 % RAM (**~13,5 ГБ на 18 ГБ**); поднимается `sudo sysctl iogpu.wired_limit_mb=…` до перезагрузки
([mlx set_wired_limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html),
[ModelPiper](https://modelpiper.com/blog/iogpu-wired-limit-mb-mac)). `set_memory_limit` — рекомендация, не жёсткий предел
([mlx-field-guide](https://github.com/dgrauet/mlx-field-guide/pull/21)).

**Термотроттлинг.** Ноутбучный M3 Pro под длительной GPU-нагрузкой сбрасывает частоты; отчёты от «почти нет» до −50 %
за 10 минут; на M4 Pro prefill одного и того же промпта 500 с холодным и 750 с прогретым
([kortexa #22](https://github.com/kortexa-ai/models.server/issues/22), [llmconfigurator](https://llmconfigurator.com/en/guides/troubleshooting/macbook-thermal-throttling-llm)).
Закладывать запас 20–30 % в бюджет; адаптивное перепланирование в `ml.py` это частично компенсирует.

---

## 4. Ранжированный список оптимизаций (выигрыш / усилия)

| # | Что | Выигрыш на 145-с видео | Усилия | Риск | Примечание |
|---|---|---|---|---|---|
| 1 | **Постоянный сервер моделей** вместо процесса на видео (§5) | −15…30 с фикс. накладных; открывает #7, #6 | 1 день | низкий (fallback на старый путь) | главный пункт |
| 2 | **ORT threads = 6 P-ядер** (не 10), torch тоже | −10…25 % perception | 5 мин | нет | `sysctl hw.perflevel0.logicalcpu` |
| 3 | **CoreML EP MLProgram для YOLOX-M/S** + `ModelCacheDirectory` + `CPUAndGPU`→`ALL` после регрессии | perception 75–125 с → 25–45 с | 0,5–1 день с валидацией | средний (FP16 на ANE, партиции) | проверка `ProfileComputePlan=1`, verbose-лог |
| 4 | **Декод в фоновом потоке** (очередь кадров) | −5…10 с | 1 ч | нет | cv2 отпускает GIL |
| 5 | **CLIP на MPS** | −5…8 с | 15 мин + сверка эмбеддингов | низкий | `.to("mps")`, fp32 |
| 6 | **Кеш vision/KV между 3 вызовами 2-из-3** (один prefill, три декода) или `mlx_vlm.server` с vision-cache/APC | VLM −40…60 % | 0,5 дня | низкий | в vllm-mlx 7,8–19× на повторном запросе |
| 7 | **Конвейер: perception(CPU/ANE) видео N+1 ‖ VLM(GPU) видео N** внутри сервера | пропускная способность очереди ×1,6–2 | внутри #1, +0,5 дня | средний (память, см. §6) | решает проблему 5 одновременных загрузок |
| 8 | **Прогрев при старте** (Metal-ядра, CoreML-кеш, 1 синтетический кадр) | −5…20 с на первом видео | 30 мин | нет | |
| 9 | Пропуск YOLOX-S-кропа/RTMPose на статичных кадрах (`motion<ε`) | до −30 % perception | 2 ч + train-проверка | средний (дрейф признаков) | |
| 10 | Переход VLM 2B → **4B** 4-bit | качество, не время: +2–3 с на вызов | полдня (переобучить скорер на новых ответах) | средний | 8B — только с #1+#6 |
| 11 | RTMPose на CoreML (MLProgram, своя сессия) | −5…10 с | 0,5 дня | средний (фолбэки в GAU) | после #3 |
| 12 | Батч N кадров для YOLOX (переэкспорт) | −10…20 % на CPU | 0,5 дня | низкий | мало смысла после #3 |
| 13 | Нативный CoreML (coremltools) / MLX-порт детекторов | ещё 2–10× на детекторах | 2–4 дня | высокий | не для дататона |
| 14 | ffmpeg `-hwaccel videotoolbox` вместо cv2 | −3…5 с CPU-времени | 2 ч | низкий | не нужно после #4 |

**Про разрешение для VLM:** не поднимать склейку до 1–2 Мпикс (prefill ×4–8, +KV). Если нужны детали — кроп зоны стола
512×512 (256 токенов) отдельным вызовом, а не апскейл целого кадра; `--max-pixels`/`resize_shape` в mlx-vlm server ограничивают
размер на стороне сервера ([mlx-vlm README](https://github.com/Blaizzy/mlx-vlm/blob/main/README.md)).

Ожидаемый итог на 145-с видео после #1–#8 **(оценка)**: perception+CLIP 25–45 с, VLM 4B на 8–10 моментов ≈ 60–100 с
(с кешем), доказательства 2–3 с → **~90–150 с**, запас к 240 с есть даже при троттлинге; очередь из 5 с совмещением #7 —
~2 мин на видео сквозной пропускной способности.

---

## 5. Архитектура для прогона жюри: постоянный сервер моделей

**Проблема текущей схемы.** `exec.CommandContext(ctx, ANALYZER_COMMAND, path)` на каждое видео → импорт torch/transformers
(5–10 с) + загрузка весов (10–20 с) + компиляция Metal/CoreML при первом вызове. При VLM 4B это 25–40 с из 255. Плюс CPU и GPU
работают по очереди.

**Рекомендация: один долгоживущий Python-процесс `forte-model-server`** (uvicorn/FastAPI — уже в зависимостях через mlx-vlm,
или `http.server` на Unix-сокете), который держит ORT-сессии (CoreML-кеш прогрет), CLIP (MPS) и VLM (MLX) и исполняет две стадии
двумя воркерами-потоками:

```
Go backend (1 worker, как сейчас)
  └─ ANALYZER_COMMAND = bin/cashguard-analyzer <video>          # контракт не меняется
       ├─ если /health сервера OK → POST /analyze (Unix-socket), ждём JSON, печатаем в stdout
       └─ иначе → старый путь (in-process MLAnalyzer), т.е. graceful degradation

forte-model-server (launchd, KeepAlive)
  ├─ очередь A: decode-thread → perception (ORT CoreML/CPU, 6 потоков) → CLIP (MPS)      [CPU + ANE/GPU немного]
  ├─ очередь B: VLM-моменты (MLX, GPU), кеш vision-эмбеддингов по hash склейки           [GPU]
  ├─ бюджет на видео считается как сейчас (mtime файла → ожидание в очереди вычитается)
  ├─ после каждого видео: mx.clear_cache(); при RSS > порога — перезапуск VLM-части
  └─ /health, /metrics (время стадий, tok/s, пик памяти), прогрев при старте
```

Почему так, а не иначе:

- **Контракт с Go не меняется** — `ANALYZER_COMMAND` остаётся, но лаунчер становится тонким клиентом (<1 с). Go-таймаут 15 мин
  остаётся предохранителем. Один воркер в Go — ок, параллелизм живёт внутри сервера (стадии A и B для разных видео).
- **Один процесс, потоки, не fork** (§3): ORT/cv2/MLX отпускают GIL. Разделять на два процесса (perception / VLM) имеет смысл
  только ради изоляции падений VLM — тогда второй процесс стартует отдельно, а не форком.
- **`mlx_vlm.server` как готовая альтернатива VLM-части**: OpenAI-совместимый `/v1/chat/completions`, предзагрузка `--model`,
  `/health`, кеш vision-фич (`--vision-cache-size 20`, в README заявлено >11× ускорение prompt processing на повторной картинке),
  continuous batching (картинки prefill-ятся по одной, декод общий), `APC_ENABLED=1` для префикса системного промпта,
  `--max-pixels`, `MLX_VLM_TOKEN_QUEUE_TIMEOUT` (дефолт 600 с) ([mlx-vlm README](https://github.com/Blaizzy/mlx-vlm/blob/main/README.md),
  [kortexa #22](https://github.com/kortexa-ai/models.server/issues/22)). Минусы: ещё один демон, формат промпта через chat-template
  (системное сообщение отдельной ролью — ближе к transformers-ветке, на которой обучен скорер, что скорее плюс), зависимость от
  поведения конкретной версии 0.7.8. Разумный компромисс: собственный сервер для perception, а VLM — через `mlx_vlm.server`
  на localhost (кеш и батчинг даром), с fallback на in-process `mlx_vlm.generate`.
- **Процесс на видео остаётся как аварийный режим** — это уже написано и протестировано.

Что замерить перед жюри (один скрипт, прогнать 3 train-видео): время старта и прогрева сервера; p50/p95 этапов; пик памяти
(`/usr/bin/time -l`, `mx.get_peak_memory()`); деградацию tok/s между 1-м и 5-м видео подряд (термо).

---

## 6. Бюджет памяти (18 ГБ unified, Metal по умолчанию ~13,5 ГБ)

| Компонент | 2B (сейчас) | 4B (рекомендация) | 8B | Источник / комментарий |
|---|---|---|---|---|
| macOS, WindowServer, системные демоны | 3–4 ГБ | 3–4 | 3–4 | без браузера/IDE |
| Go-сервер + PostgreSQL (brew, не Docker) | 0,3 | 0,3 | 0,3 | Docker Desktop VM +1–2 ГБ — не запускать |
| ORT: YOLOX-M, YOLOX-S, RTMPose-m (CPU) | 0,4–0,6 | 0,4–0,6 | 0,4–0,6 | `deploy_mac.md` §6 |
| + CoreML EP (скомпилированные копии, буферы) | +0,3–0,5 | +0,3–0,5 | +0,3–0,5 | **(оценка)** |
| torch + CLIP ViT-B/32 fp32 (CPU/MPS) | 0,9–1,2 | 0,9–1,2 | 0,9–1,2 | `deploy_mac.md`; MPS +~0,3 |
| Кадры (290 × ½-разрешения + кропы 224²) | 0,25 | 0,25 | 0,25 | 145 с @ 2 к/с |
| VLM веса | 1,8 | 3,1 | 5,7–5,8 | HF: [2B](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit), [4B](https://huggingface.co/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit), [8B](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit/discussions/2) |
| VLM активации + KV (склейка 640 px, ≤400 токенов) + кеш MLX | 0,4–0,6 | 0,6–0,9 | 0,8–1,2 | vllm-mlx: 4B 2 кадра 3,2 ГБ всего |
| Python/OpenCV/прочее | 0,3 | 0,3 | 0,3 | |
| **Итого резидентно** | **≈ 8–9 ГБ** | **≈ 10–11 ГБ** | **≈ 13–14 ГБ** | |
| Запас до 18 ГБ | 9–10 | 7–8 | **4–5 — впритык** | |
| Второе видео в стадии A параллельно (+кадры +активации ORT) | +0,5 | +0,5 | +0,5 | конвейер §5 |

Выводы: **4B 4-bit с конвейером помещается с запасом**; 8B возможен только без параллелизма стадий и с `iogpu.wired_limit_mb`
≈ 15000 (оставить ОС ≥3 ГБ), swap должен оставаться ~0 (`Activity Monitor → Memory`, `sysctl vm.swapusage`). Контроль:
`mx.get_peak_memory()` в логе каждого видео; при давлении — `FORTE_VLM_MAX_SIDE=512` → `FORTE_USE_VLM=0` (как в `deploy_mac.md`).

---

## 7. Чек-лист надёжности ноутбука-сервера (1–2 дня жюри)

**Питание и сон**
- [ ] Только от сети. `caffeinate -s` действует **только на AC** и не мешает сну по закрытию крышки
  ([man caffeinate](https://ss64.com/mac/caffeinate.html)). Запускать серверы под `caffeinate -dimsu -w <pid>` или в LaunchDaemon.
- [ ] `sudo pmset -a sleep 0 displaysleep 0 disksleep 0 standby 0 autopoweroff 0 powernap 0 tcpkeepalive 1`; проверить `pmset -g`
  и `pmset -g assertions` ([man pmset](https://ss64.com/mac/pmset.html)). `sudo pmset -a disablesleep 1` — работа с закрытой крышкой
  без монитора; отчёты о надёжности на Apple Silicon противоречивы → **крышку держать открытой** (это ещё и лучше для охлаждения),
  либо подключить внешний монитор (clamshell).
- [ ] Отключить автообновления macOS и автоматическую перезагрузку; снять «Низкое энергопотребление»; ноутбук на твёрдой
  поверхности/подставке, прохладное помещение. Высокопроизводительный режим (High Power Mode) на M3 Pro недоступен (только Max).
- [ ] Отключить Spotlight-индексацию каталога загрузок видео (`sudo mdutil -i off <dir>`) и Time Machine на время жюри.

**Процессы**
- [ ] Go-сервер, model-server и `cloudflared` — как LaunchDaemon/LaunchAgent с `KeepAlive=true` и логами в файлы; проверить
  автоперезапуск, убив процесс (`kill -9`) и перезагрузив Mac полностью (холодный старт → прогрев → health OK).
- [ ] Watchdog-скрипт (launchd каждые 60 с): `GET /health` Go и model-server, свободная память/диск, swap; при провале —
  `launchctl kickstart -k`.
- [ ] Graceful degradation: при недоступности model-server лаунчер идёт старым путём (процесс на видео, CV-only при нехватке
  времени); при падении VLM — CV-only результат, не таймаут (уже так в `ml.py`).
- [ ] Прогрев при старте (синтетический кадр через все модели), `ModelCacheDirectory` заполнен заранее, HF-кеш офлайн
  (`HF_HUB_OFFLINE=1`), все веса локально; ни одного сетевого вызова в инференсе (требование ТЗ).
- [ ] Логи с ротацией; на диске ≥ 10 ГБ свободно (загрузки видео, JPEG-доказательства, кеши).

**Сеть / Cloudflare Tunnel**
- [ ] **Лимит тела запроса 100 МБ на Free/Pro** (200 Business), иначе HTTP 413
  ([Cloudflare cache limits](https://developers.cloudflare.com/cache/concepts/default-cache-behavior/)). Проверить размеры
  тестовых .mov (145 с CCTV 464×848 может быть 40–150 МБ). Если близко к лимиту: чанковая загрузка на фронте, либо отдельный
  непроксируемый путь (Tailscale Funnel / VPS с reverse-SSH или WireGuard), либо Business.
- [ ] Таймаут чтения ответа от origin **125 с → 524**, запись 30 с, idle 900 с
  ([Cloudflare connection limits](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)). Наш API асинхронный
  (`POST /api/jobs` → `job_id`, опрос) — долгих запросов нет; убедиться, что загрузка отвечает сразу после записи файла, не после анализа.
- [ ] `cloudflared` как daemon: `sudo cloudflared service install`, конфиг в `/etc/cloudflared`, логи
  `/Library/Logs/com.cloudflare.cloudflared.{out,err}.log`
  ([Cloudflare docs macOS service](https://developers.cloudflare.com/tunnel/advanced/local-management/as-a-service/macos/)).
  Named tunnel (не quick tunnel — URL меняется). Флаги: `--no-autoupdate` (автообновление перезапускает процесс и рвёт соединения),
  `--protocol http2` если на Wi-Fi площадки заблокирован UDP/QUIC (по умолчанию `auto` с фолбэком), `--retries` не трогать,
  `--grace-period 30s` ([run parameters](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/cloudflared-parameters/run-parameters/)).
  Origin: `connectTimeout 30s`, `keepAliveTimeout 1m30s` по умолчанию достаточно
  ([origin parameters](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/cloudflared-parameters/origin-parameters/)).
- [ ] Проводной Ethernet (USB-C) предпочтительнее Wi-Fi; отключить «Переключение сетей»; резерв — раздача с телефона.
  Проверить с внешнего устройства: загрузка файла 100 МБ, опрос статуса, скачивание результата.

**Репетиция**
- [ ] Прогнать 5 train-видео подряд через туннель с замером: сквозное время каждого, пик памяти, swap=0, tok/s 1-го и 5-го
  (термо), результаты совпадают с офлайн-прогоном (регрессия CoreML/MPS).
- [ ] Загрузить 5 видео одновременно и убедиться, что 5-е получает корректный (хотя бы CV-only) ответ в пределах 5 минут.
- [ ] Отдельная «кнопка» отката: переменные `FORTE_ORT_EP=cpu`, `FORTE_USE_VLM=0`, `FORTE_MODEL_SERVER=0`.

---

## 8. Что осталось непроверенным (нужны замеры на самом Mac)

1. Фактическая цена кадра на CPU EP и CoreML EP (YOLOX-M/S) — есть только аналогии по классу моделей.
2. Доля графа RTMPose, которая ложится на CoreML (ожидаю фолбэки в GAU-голове).
3. Реальные prefill/decode tok/s mlx-vlm для Qwen3-VL-4B/8B на M3 Pro — все найденные замеры на M1/M4 Max.
4. Скорость декодирования 464×848 H.264 в cv2 (ожидаю >500 к/с).
5. Размеры тестовых видео относительно лимита Cloudflare 100 МБ.

## Источники

- ONNX Runtime CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ORT issues: #22275 (крэш macOS 15), #17033 (FP16 медленнее), #16934 (CPU-only флаг), #21227 (SIGSEGV при динамическом batch)
- Xybrid, ORT CoreML на M2: https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml
- SWHL, CoreML EP vs CPU vs нативный CoreML: https://swhl.github.io/main/blog/accelerate-onnx-models-with-coreml-on-macos/
- ym2132, скрытый FP16 в формате NeuralNetwork: https://ym2132.github.io/ONNX_MLProgram_NN_exploration
- Roboflow PR #3082 (кеш CoreML, MLProgram-партиции): https://github.com/roboflow/inference/pull/3082
- macgpu: CoreML EP vs CPU EP, динамические формы: https://macgpu.com/en/blog/2026-0420-mac-onnx-runtime-coreml-ep-vs-cpu-dynamic-shapes-remote.html
- numen #403 / PR #423 (потоки ORT на Apple Silicon): https://github.com/jiva-studio/numen/issues/403 , https://github.com/jiva-studio/numen/pull/423
- YOLOX network_blocks.py: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py
- mmpose rtmcc_block.py: https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/rtmcc_block.py
- vllm-mlx, статья (кеш vision, Qwen3-VL-4B/8B на M4 Max): https://arxiv.org/html/2601.19139v2
- vllm-mlx video benchmarks (M4 Max, M1 Max): https://github.com/waybarrios/vllm-mlx/blob/main/docs/benchmarks/video.md
- mlx-vlm README (server, vision cache, APC, max-pixels): https://github.com/Blaizzy/mlx-vlm/blob/main/README.md
- mlx-vlm #717 (prefill tok/s на M4 Max): https://github.com/Blaizzy/mlx-vlm/issues/717
- kortexa models.server #22 (prefill vs нагрев, таймаут 600 с): https://github.com/kortexa-ai/models.server/issues/22
- Qwen3-VL (patch 16, 32 px/токен): https://github.com/qwenlm/qwen3-vl ; preprocessor_config 4B-4bit: https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/raw/main/preprocessor_config.json
- Размеры весов: https://huggingface.co/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit , https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit/discussions/2
- llmcheck (оценки tok/s по полосе памяти, M3 Pro 18 ГБ): https://llmcheck.net/benchmarks
- Apple M3 / M3 Pro спецификации: https://en.wikipedia.org/wiki/Apple_M3
- MLX память: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html , https://modelpiper.com/blog/iogpu-wired-limit-mb-mac , https://github.com/dgrauet/mlx-field-guide/pull/21
- Матричная производительность CPU/GPU M-серии: https://arxiv.org/html/2502.05317v1
- OpenCV hwaccel (бэкенды): https://docs.opencv.org/4.13.0/db/dc4/group__videoio__hwaccel.html
- multiprocessing на macOS: https://docs.python.org/3/library/multiprocessing.html , https://bugs.python.org/issue33725
- Термотроттлинг: https://llmconfigurator.com/en/guides/troubleshooting/macbook-thermal-throttling-llm
- caffeinate / pmset: https://ss64.com/mac/caffeinate.html , https://ss64.com/mac/pmset.html ; крышка: https://www.machinefriendly.com/blog/keep-macbook-awake-lid-closed-awaketoggle
- Cloudflare: лимит загрузки https://developers.cloudflare.com/cache/concepts/default-cache-behavior/ ; таймауты https://developers.cloudflare.com/fundamentals/reference/connection-limits/ ; сервис macOS https://developers.cloudflare.com/tunnel/advanced/local-management/as-a-service/macos/ ; run/origin параметры https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/cloudflared-parameters/run-parameters/ , https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/cloudflared-parameters/origin-parameters/
