# Mac 18 ГБ: VLM побольше и дообучение (LoRA) — инструкция для Аслана

Что это даёт: на Linux-ноутбуке VLM Qwen3-VL-**2B** не видит мелкое (пачку денег на столе, ключ в замке, дверцу
сейфа). На Mac с 18 ГБ влезает **Qwen3-VL-8B** в 4 бита (5.8 ГБ, Apache-2.0) — на вопросах «да/нет» разница большая.
Второй шаг — дообучить её (LoRA) на наших кадрах с известными ответами.

Общая база (venv, веса YOLOX/RTMPose/CLIP, видео в `data/train`, `data/test`) — по `docs/deploy_mac.md`.
Запускать всё из корня репозитория, ветка `aslan-18gb`.

## 0. Установка (один раз)
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt          # включает mlx-vlm, onnxruntime, torch (CPU), rtmlib
bash scripts/download_models.sh                 # YOLOX, RTMPose, CLIP
python -c "from huggingface_hub import snapshot_download as s; s('mlx-community/Qwen3-VL-8B-Instruct-4bit')"
```
Кэш признаков по видео (один раз, ~15 мин на M3):
```bash
python scripts/extract_features.py data/train/videos data/test/videos --out data/cache --fps 2 --threads 8
python scripts/extract_clip.py --cache data/cache
```

## 1. Прогон 8B-модели по ключевым моментам (без обучения)
```bash
export FORTE_VLM_BACKEND=mlx
export FORTE_VLM_MLX=mlx-community/Qwen3-VL-8B-Instruct-4bit
python scripts/run_vlm.py --split train --out artifacts/vlm_8b              # все 40 видео (test отпущен → split train)
python scripts/confirm_vlm.py --src artifacts/vlm_8b --dst artifacts/vlm_8b_confirmed   # ансамбль самопроверки 2-из-3
git add artifacts && git commit -m "VLM 8B answers" && git push
```
Ориентир по времени: 2–7 вопросов на видео, на M3 Pro ~5–10 с на вопрос → весь датасет ≈ 20–30 мин.
Дальше я (на Linux) обучаю скорер на этих ответах: `FORTE_VLM_DIR=artifacts/vlm_8b_confirmed python scripts/train_scorer.py --with-vlm`
и сравниваю по кабинетам с текущими 0.725 / 0 FP.

## 2. LoRA-дообучение на наших кадрах
Пары «кроп + вопрос + JSON-ответ» генерируются из разметки (разбиение по кабинетам C,E → val):
```bash
python scripts/make_vlm_sft.py --out data/sft --val-cabinets C,E --aug 2
# → data/sft/train.jsonl, data/sft/val.jsonl, data/sft/images/*.jpg  (~200–400 пар)
```
Обучение (mlx-vlm, LoRA на языковой части; vision-энкодер заморожен):
```bash
python -m mlx_vlm.lora \
  --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --dataset data/sft --image-field image --messages-field messages \
  --epochs 3 --batch-size 2 --learning-rate 1e-4 --lora-rank 8 \
  --output-path artifacts/vlm_lora
```
> Проверить флаги по установленной версии: `python -m mlx_vlm.lora --help`. В разных версиях датасет задаётся
> как папка с `train.jsonl`/`val.jsonl` или как HF-датасет; формат строки у нас — `{"image": ..., "messages": [...]}`.
> Если памяти не хватает — `--batch-size 1`, или взять `mlx-community/Qwen3-VL-4B-Instruct-4bit` (3.1 ГБ).

Слить адаптер в веса и прогнать как в п. 1:
```bash
python -m mlx_vlm.fuse --model mlx-community/Qwen3-VL-8B-Instruct-4bit --adapter-path artifacts/vlm_lora --save-path models/qwen3vl-8b-forte
export FORTE_VLM_MLX=$PWD/models/qwen3vl-8b-forte
python scripts/run_vlm.py --split train --out artifacts/vlm_lora_answers
python scripts/confirm_vlm.py --src artifacts/vlm_lora_answers --dst artifacts/vlm_lora_confirmed
```
Честная проверка дообучения — **по кабинетам C и E** (val), которых модель не видела; сравнить ответы на них
с `data/sft/val.jsonl`. Если на val улучшения нет — оставить базовую 8B.

Ограничение по времени: 300–400 пар, 3 эпохи, 8B 4-bit на M3 Pro — порядка 1–3 часов. Можно оставить на ночь.

## 3. Как это будет работать при запуске (сдача)
Ничего сложного: тот же `bin/cashguard-analyzer`, бэкенд вызывает его на каждое видео. Разница только в переменных
окружения **перед стартом Go-сервера** (он передаёт своё окружение анализатору):
```bash
export FORTE_USE_VLM=1
export FORTE_VLM_BACKEND=mlx
export FORTE_VLM_MLX=mlx-community/Qwen3-VL-8B-Instruct-4bit   # или $PWD/models/qwen3vl-8b-forte после LoRA
export ANALYZER_JURY_LIMIT_SEC=255
DEMO_MODE=false ANALYZER_COMMAND=$PWD/bin/cashguard-analyzer go -C src/backend run ./cmd/server
```
- Модель загружается при каждом запуске анализатора (на каждое видео): 8B 4-bit с SSD ≈ 5–10 с — заложено в бюджет.
- Бюджет времени считается от момента загрузки видео (mtime файла), поэтому очередь из 5 видео учитывается
  автоматически: когда времени мало, VLM пропускается и возвращается CV-результат. Таймаут невозможен.
- Память: YOLOX+RTMPose+CLIP ≈ 1.5 ГБ, 8B 4-bit ≈ 6–7 ГБ с активациями → в 18 ГБ умещается с запасом.
- Смоук-тест перед сдачей: 5 видео подряд через публичный URL, каждое должно вернуть `done` быстрее 5 минут
  (`docs/api_contract_gap.md` — пути и поля, как у жюри).
- Веса — локальные, интернет после загрузки не нужен (`HF_HUB_OFFLINE=1` выставляется автоматически).
