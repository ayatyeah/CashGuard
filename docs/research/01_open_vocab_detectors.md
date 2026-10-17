# 01 · Zero-shot / open-vocabulary детекторы мелких объектов и их состояний (10–40 px, 464×848)

Исследовательский агент #1 из 10. Дата: 2026-10-17. Источники — карточки HF, репозитории, статьи, issues (ссылки в конце строк).
Ничего не запускалось; оценки CPU-скорости — из чужих замеров и расчёта по архитектуре.

## 0. Главный вывод

1. **Детекторы ищут существительные, а не состояния.** На бенчмарке FG-OVD (CVPR'24) лучший open-vocab детектор OWLv2-L/14
   даёт ~37 mAP по атрибутам (цвет 53, материал 37, узор 23); Grounding DINO «struggles with fine-grained detection».
   Поэтому «дверь открыта / ключ в замке / пачка пересчитана» надёжнее решать как **детектор (где объект) + классификатор состояния на кропе**
   (DINOv2 kNN/логрег или уже имеющийся Qwen3-VL yes/no), а не одним промптом «open safe door».
2. **Проблема 10–40 px решается кропами, а не моделью.** Все кандидаты режут вход до 800–960 px; объект 20 px в полном кадре — это 1–2 патча.
   Кроп 240×240 вокруг зоны интереса (стол, сейф, запястья из RTMPose) с апскейлом до 960 даёт объект 80–160 px — «обычный» размер для любого детектора.
3. **Лицензии:** годятся OWLv2, Grounding DINO 1.0, MM-Grounding-DINO, LLMDet, Florence-2, DINOv2, SAM 2.1, OmDet-Turbo, Moondream2 (Apache-2.0/MIT).
   **Не годятся:** YOLO-World (GPL-3.0), YOLOE (AGPL-3.0), Grounding DINO 1.5/1.6 и DINO-X (только облачный API = дисквалификация),
   PaliGemma 2 (Gemma Terms, не OSI), DINOv3 (кастомная лицензия Meta), Moondream 3 (BSL).

## 1. Сводная таблица кандидатов

| Модель | Лицензия кода / весов (проверено) | Вердикт | Размер | CPU / Apple Silicon | Мелкие объекты и состояния | Ссылки |
|---|---|---|---|---|---|---|
| **OWLv2** base-patch16-ensemble / large-patch14 | Apache-2.0 / Apache-2.0 (карточка HF) | **ok** | 0.2B (base), ~0.5B (large) | PyTorch CPU/MPS; фиксированный вход 960×960, patch16 → 3600 токенов; CPU ~7× медленнее OWL-ViT (форум HF: 440 тайлов за 3 мин 03 с, ≈0.4 с/тайл на многоядерном CPU); ONNX: `Xenova/owlv2-base-patch16-ensemble` (transformers.js, подходит и для onnxruntime); CoreML — нет готового | Лучший по мелким объектам среди OVD: в аэросъёмке (LAE-80C) recall в 3× выше остальных; «distinct advantage in detecting smaller objects» (eyewear study); лучший на FG-OVD по атрибутам. Минус: слабее на описаниях с отношениями («phone in hand») | [HF](https://huggingface.co/google/owlv2-base-patch16-ensemble), [docs](https://huggingface.co/docs/transformers/en/model_doc/owlv2), [ONNX](https://huggingface.co/Xenova/owlv2-base-patch16-ensemble), [aerial](https://arxiv.org/abs/2601.22164), [CPU thread](https://discuss.huggingface.co/t/owlv2-significantly-slower-than-owlvit/73223), [padding bug](https://github.com/huggingface/transformers/issues/27205) |
| OWL-ST | Это рецепт самообучения, веса = OWLv2 (Apache-2.0) | ok (= OWLv2) | — | — | — | [paper](https://arxiv.org/abs/2306.09683) |
| **Grounding DINO 1.0** tiny/base | Apache-2.0 / Apache-2.0 (HF IDEA-Research) | **ok** | 0.2B tiny | transformers; ONNX требует `disable_custom_kernels=True`; CPU 2–5 с/кадр при 800 px (анекдоты: 5 с на i5-10th, ONNX) | Свободные фразы, но слабее на атрибутах; small objects — только через кропы | [HF](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [docs](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino), [TRT issue](https://github.com/NVIDIA/TensorRT/issues/3580) |
| Grounding DINO 1.5 / 1.6 (Pro, Edge) | Код Apache-2.0, **веса не выдаются — только API DeepDataSpace по токену** | **не ok** (облако) | — | — | Edge-веса в репо не найдены (поиск по issues «edge weights» — 0 результатов) | [repo](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) |
| **MM-Grounding-DINO** (mmdet) tiny/base/large | Apache-2.0 / Apache-2.0 (`openmmlab-community/*` на HF) | **ok** | 0.2B tiny | Нативно в transformers (`MMGroundingDinoForObjectDetection`), без mmcv; CPU как GD-T (2–5 с при 800 px, ~1 с на кропе 480) | LVIS minival AP 41.4 vs 27.4 у GD-T (+12.6), APr 34.2 — заметно лучше на редких классах («collection bag», «key») | [HF docs](https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino), [HF](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det), [mmdet README](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) |
| **LLMDet** tiny/base/large | Apache-2.0 / Apache-2.0 (HF) | **ok** | 0.2B tiny | Тот же класс transformers, что и MM-GDINO (≥4.55) | LVIS minival AP 50.7, APr 44.7 (tiny) — лучший из локальных open-source на редких классах | [repo](https://github.com/iSEE-Laboratory/LLMDet), [HF](https://huggingface.co/iSEE-Laboratory/llmdet_tiny) |
| **Florence-2** base/large | MIT / MIT | **ok** | 0.23B / 0.77B | Нативно в transformers с 2025-08 (`florence-community/Florence-2-base`, без remote code); ONNX `onnx-community/Florence-2-base` (MIT); MLX: папка `florence2` есть в mlx-vlm; CPU «несколько секунд» на запрос (Roboflow), L4 GPU 0.65 с | `<OD>` консервативен (видит только «уверенные» объекты); `<CAPTION_TO_PHRASE_GROUNDING>` и `<OPEN_VOCABULARY_DETECTION>` работают по фразам; есть готовый рецепт SAHI-слайсинга через `sv.InferenceSlicer` | [HF](https://huggingface.co/microsoft/Florence-2-base), [native](https://huggingface.co/florence-community/Florence-2-base), [ONNX](https://huggingface.co/onnx-community/Florence-2-base), [SAHI](https://roboflow.com/how-to-use-sahi/florence-2), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) |
| YOLO-World S/M/L/X | **GPL-3.0** код и веса (HF `wondervictor/YOLO-World` тег gpl-3.0); порт в ultralytics — AGPL-3.0 | **не ok** | 13–110M | ONNX есть | Быстрый, но лицензия закрывает | [repo](https://github.com/AILab-CVC/YOLO-World), [HF](https://huggingface.co/wondervictor/YOLO-World) |
| YOLOE v8/11 S/M/L | **AGPL-3.0** (на базе ultralytics) | **не ok** | 10–50M | ONNX/CoreML export | — | [repo](https://github.com/THU-MIG/yoloe) |
| DINO-X | Код Apache-2.0, модель **только в облаке** DeepDataSpace | **не ok** | — | — | — | [repo](https://github.com/IDEA-Research/DINO-X-API) |
| PaliGemma 2 3B mix-448/896 | **Gemma Terms of Use** (не OSI; Prohibited Use Policy, Google «reserves the right to restrict (remotely or otherwise) usage», обязательства при распространении) | **не ok** по нашему списку (Apache/MIT/BSD) | 3B | MLX-порты есть; CPU непригодно | `detect a ; b` — боксы через loc-токены; 896 px лучше на мелких деталях | [HF](https://huggingface.co/google/paligemma2-3b-mix-448), [terms](https://ai.google.dev/gemma/terms) |
| SAM 2.1 hiera-tiny/small | Apache-2.0 / Apache-2.0 | **ok** | 39–46M | Apple выложила CoreML (`apple/coreml-sam2-tiny`, только image-режим); MLX-порт есть только для SAM3; PyTorch CPU ~1 с/кадр | Не детектор и не классификатор состояний; полезен для трекинга найденной сумки/пачки между кадрами | [HF](https://huggingface.co/facebook/sam2.1-hiera-small), [CoreML](https://huggingface.co/apple/coreml-sam2-tiny) |
| **DINOv2** ViT-S/14, B/14 | Apache-2.0 / Apache-2.0 | **ok** | 22M / 86M | Тривиальный ONNX/CoreML; есть в mlx-vlm (`dinov2`); ViT-S на кропе 224 — ~10–30 мс CPU | Patch-фичи + kNN/логрег — стандарт few-shot (AnomalyDINO: one-shot AUROC 96.6 на MVTec). Для состояний (открыто/закрыто, ключ есть/нет) на тесных кропах | [HF](https://huggingface.co/facebook/dinov2-small), [AnomalyDINO](https://arxiv.org/abs/2405.14529) |
| DINOv3 | **DINOv3 License** (кастом Meta: ITAR/trade-control запреты, gated-доступ, Meta может менять условия) | **риск** — не Apache/MIT/BSD | 21M–7B | — | Выигрыш над DINOv2 на наших кропах не доказан; не стоит риска | [LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| CLIPSeg rd64-refined | Apache-2.0 | ok, но слабый | 0.2B | CPU ок | Карта 352×352, грубая; для 10–40 px бесполезна без кропа; уступает OWLv2 | [HF](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
| APE (A/B/C/D, Ti) | Apache-2.0 / Apache-2.0 | ok по лицензии, **не ok по железу** | EVA-02-L (≈300M+) | Deformable-attention CUDA-ops, detectron2-подобный стек; CPU/Mac — только с портированием | — | [repo](https://github.com/shenyunhang/APE), [HF](https://huggingface.co/shenyunhang/APE) |
| OV-DINO Swin-T | Apache-2.0 / Apache-2.0 | ok по лицензии, **не ok по железу** | ~0.2B | CUDA 11.6 ops, ONNX в TODO | — | [repo](https://github.com/wanghao9610/OV-DINO) |
| OmDet-Turbo Swin-T (бонус) | Apache-2.0 | ok | 0.1B | В transformers; «real-time» | Запас на случай, если MM-GDINO слишком медленный | [HF](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) |
| Moondream2 (бонус) | Apache-2.0 | ok | 2B | MPS/MLX (mlx-vlm); CPU медленно | `detect()/point()/query()`; альтернатива Qwen3-VL-2B как верификатор. Moondream 3 preview — BSL, **не ok** | [HF](https://huggingface.co/vikhyatk/moondream2) |

## 2. Почему кропы, и как именно

- Полный кадр 464×848 → OWLv2 паддит до квадрата 848 и масштабирует к 960 (×1.13): объект 20 px → 23 px ≈ 1.4 патча. Безнадёжно.
- Кроп 240×240 → 960 (×4): объект 20 px → 80 px (5 патчей), 40 px → 160 px. Это зона уверенной работы всех детекторов.
- У нас уже есть якоря для кропов: зона стола, зона сейфа (из «safe zone change»), запястья кассира/клиента из RTMPose. Кропы по якорям дешевле и чище SAHI-слайсинга всего кадра (3–6 тайлов ~460×460 с overlap 20%).
- Для «ключ в замке» (≈10 px) — фиксированный кроп 120×120 вокруг замка сейфа (координаты задаются один раз на кабинет, в относительных единицах), апскейл ×8; дальше не детектор, а DINOv2-kNN «ключ есть / нет» по 1–3 эталонам.
- OWLv2 подводный камень: процессор паддит до квадрата, поэтому `target_sizes=[(max(H,W), max(H,W))]`, иначе боксы «уезжают» по Y (issue #27205). На квадратных кропах проблема исчезает сама.
- OWLv2 умеет **image-guided (one-shot) detection**: эталонный кроп «инкассаторская сумка» / «пачка купюр» из нашего же видео вместо текста — обходит «vocabulary confusion», которую авторы аэро-бенчмарка назвали главным узким местом. Ограничение: один query-кроп за вызов, время растёт линейно по числу эталонов — кэшировать эмбеддинги.
- Бюджет времени (Mac M3 Pro, MPS): OWLv2-base ≈0.3–0.6 с/кроп, MM-GDINO-T на 480 px ≈0.7–1.5 с, DINOv2-S ≈0.02 с. При 5 мин/видео детектор можно вызывать только на событийно-отобранных моментах (вход/выход клиента, открытие сейфа, руки над столом): 30–80 кропов на видео → 1–2 мин. На Linux-CPU-dev-машине ориентироваться на ×3–5 медленнее.

## 3. Top-5 для нашей задачи

| # | Вариант | Лицензия | Какие пропуски закрывает | Ожидаемый эффект | Часы | Риск |
|---|---|---|---|---|---|---|
| 1 | **OWLv2-base на кропах стола/рук + image-guided эталоны** («banknote bundle», «ID card», «collection bag», «cup», «phone») | ok (Apache-2.0) | cash handed back; cash taken without ID; cash of previous client left on desk; collection bag emptied; rejected notes merged (через счётчик «пачка на столе» по времени) | Главный кандидат: лучший recall на мелких объектах и атрибутах среди OVD; появляется сигнал «деньги/ID на столе в момент t» | 8–12 (кропы по якорям, пороги на 40 видео, кэш эмбеддингов, ONNX опционально) | FP при низком threshold (0.1 по умолчанию) — калибровать per-class на видео без нарушений; 0.3–0.6 с/кроп |
| 2 | **DINOv2-S patch-фичи + kNN/логрег на тесных кропах для СОСТОЯНИЙ** (дверь сейфа открыта/закрыта, ключ в замке, сумка пустая/полная) | ok (Apache-2.0) | key left in safe lock; door state after entry (усиление); collection bag emptied | Дёшево (мс), few-shot; закрывает «ключ в замке», который детекторы не видят вовсе | 4–6 | CLIP-линейный проб на кропах стола уже ухудшал cabinet-CV. Отличие: кроп тесный и якорный (замок, дверь), kNN вместо линейной модели, фичи — patch-токены, не CLS. Проверять только по leave-one-cabinet-out |
| 3 | **MM-Grounding-DINO-T или LLMDet-T (transformers)** на тех же кропах с фразами | ok (Apache-2.0) | те же, что #1, плюс фразы с отношением («hand holding banknotes», «bag over safe») | Лучше OWLv2 на редких классах (LVIS APr 34–45 vs ~25); фразы с контекстом | 6–8 (та же обвязка, что #1) | 2–3× медленнее OWLv2; слабее на атрибутах; 900 queries → много мусора при низких порогах |
| 4 | **Florence-2-base** `<CAPTION_TO_PHRASE_GROUNDING>` / `<OPEN_VOCABULARY_DETECTION>` на кропах + `<DENSE_REGION_CAPTION>` для текста доказательств | ok (MIT) | как #1; плюс человекочитаемое описание кадра-доказательства («продукт» 10 баллов) | Второе мнение (ансамбль с #1 снижает FP); ONNX и MLX есть | 6–8 | `<OD>` консервативен; генеративная модель — непредсказуемая латентность (несколько секунд на CPU) |
| 5 | **SAM 2.1-tiny для трекинга найденной пачки/сумки** между опорными кадрами | ok (Apache-2.0) | cash of previous client left on desk when next enters (объект «живёт» на столе через смену клиента) | Превращает разовые детекции в временные интервалы, снимает дребезг | 6–10 | Память/время на видео; CoreML-порт только image-режим; выигрыш против простого IoU-трекинга боксов не гарантирован — делать после #1–#3 |

Отвергнуто: YOLO-World (GPL), YOLOE (AGPL), GD 1.5/1.6 и DINO-X (облако), PaliGemma 2 (Gemma Terms), DINOv3 (кастом), Moondream 3 (BSL); APE и OV-DINO — CUDA-ops, на Mac/CPU не поднять без портирования; CLIPSeg — слишком грубый.

## 4. Evidence по surveillance-подобным кадрам

- Прямых кейсов «OVD на CCTV кассы/магазина» не найдено (ни статей, ни блогов с цифрами). Ближайшее: аэросъёмка (мелкие объекты, вид сверху) — OWLv2 лучший, но абсолютные числа низкие (F1 27.6 %, FPR 69 %), и главный фактор — размер словаря: сокращение с 80 до ~3 классов улучшило результат ~15×. Для нас: **короткий словарь на кроп (2–4 класса), а не 20 промптов**.
- Отраслевой паттерн: OVD используют для разметки, затем учат маленький детектор (RF-DETR Apache-2.0 и т.п.). У нас 40 видео — можно разметить OWLv2 полуавтоматом сотни кропов «купюры/ID/сумка» и обучить лёгкий классификатор кропов (DINOv2-head) — это и есть #2 в усиленном виде.
- Robotics-пайплайн (NILS, arXiv 2410.17772): Grounding DINO лучше на специфичных описаниях с цветом/отношениями, OWLv2 — на простых существительных. Делить промпты: OWLv2 — существительные, MM-GDINO — фразы.

## 5. Что пробовать первым (10 строк)

1. Взять `google/owlv2-base-patch16-ensemble` через transformers (MPS на Mac, CPU на dev), transformers ≥4.4x; `threshold` начать с 0.2.
2. Кропы: квадрат 240–320 px вокруг (а) центра стола, (б) каждого запястья кассира/клиента из RTMPose, (в) зоны сейфа; апскейл до 960 билинейно; `target_sizes=[(S,S)]`.
3. Словарь на кроп ≤4 классов: стол — `banknotes`, `stack of cash`, `ID card`, `passport`; руки — `banknotes`, `phone`, `cup`; сейф — `bag`, `sack`, `cash`.
4. Вызывать только в событийные окна (±2 с от входа/выхода клиента, открытия сейфа, «руки над столом») с шагом 0.5–1 с → ≤80 кропов/видео.
5. Калибровать пороги per-class на 19 видео без нарушений (цель 0 FP), затем проверять recall на 21 с нарушениями; метрика — leave-one-cabinet-out.
6. Вырезать 1–3 эталонных кропа «инкассаторская сумка» и «пачка купюр» из своих видео → `image_guided_detection`, кэш эмбеддингов; сравнить с текстовыми промптами.
7. Для «ключ в замке» и «дверь открыта»: фиксированный кроп замка/двери 120–160 px (относительные координаты на кабинет) → DINOv2-S patch-фичи (mean-pool последних 4 слоёв) → kNN по 3–5 эталонам на состояние.
8. Собрать временные ряды «деньги на столе», «ID на столе», «сумка у сейфа» → в существующие candidate events (cash left on desk при смене клиента; cash без ID; сумка→сейф без паузы «пересчёт»).
9. Если OWLv2 даёт recall <0.7 на мелких классах — повторить п.2–5 с `iSEE-Laboratory/llmdet_tiny` (тот же код через `AutoModelForZeroShotObjectDetection`), сравнить; ансамбль «оба согласны» для снижения FP.
10. Только после стабилизации #1–#2: Florence-2-base `<DENSE_REGION_CAPTION>` на кадре-доказательстве для текста описания; SAM 2.1 — если дребезг детекций мешает временным событиям.

## 6. Ограничения этого обзора

- Лимит веб-поиска сессии исчерпан на ~85 % задач; не проверены: наличие готового CoreML/ONNX для MM-GDINO (ONNX — через `disable_custom_kernels`, CoreML — скорее нет), точные CPU-замеры OWLv2/Florence-2 на M-серии, кейсы «OVD на банкнотах/ID». Это стоит замерить локально (скрипт на 20 кропах), а не искать.
- Утверждения о скорости — оценки по архитектуре и чужим анекдотам (Jetson Orin TRT 400+ мс; форум HF ≈0.4 с/тайл CPU; GD ONNX 5 с на i5).
