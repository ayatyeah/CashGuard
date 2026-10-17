# Mac M3 Pro 18 ГБ: проверяемый эксперимент VLM/SFT

Статус 17.10.2026: модель в production не заменена. Прирост, память и скорость на этом Mac **не измерены**.
Сначала сравнить текущую 2B с Qwen3-VL-4B-Instruct 4-bit; 8B/9B отложить до измерений.
Больше параметров не восстанавливают детали 10–40 px, утраченные при уменьшении мозаики до 640 px.
У Qwen3-VL-4B [карточка производителя](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) указывает Apache-2.0,
у [MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/main/LICENSE) — MIT.
Карточки конверсий [4B](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/raw/main/README.md)
и [2B](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit/raw/main/README.md) также указывают Apache-2.0.
Скрипт загрузки сохраняет точный revision в `forte_download.json` рядом с весами.
Загрузка 4B превышает 2 ГБ; она оправдана только запуском A/B на Mac и здесь не выполнялась.

## Окружение и базовый A/B

Запускать из корня, после установки зависимостей проекта по `docs/deploy_mac.md`.
Для SFT использовать отдельный venv: текущий upstream сменил CLI с `mlx_vlm.lora` на `mlx_vlm.train`.
Ниже команды привязаны к проверенному исходному коду, а не к плавающему `pip install -U`:

```bash
python3.11 -m venv .venv-vlm
source .venv-vlm/bin/activate
python -m pip install -r requirements-mac.txt
python -m pip install 'mlx-vlm[train] @ git+https://github.com/Blaizzy/mlx-vlm.git@73d366ecf9233b3fc6549531a7f2386726d09ff6'
python -m mlx_vlm.train --help
mkdir -p models
python -m pip freeze > models/mac-vlm-environment.txt
# Один раз, интернет нужен только для получения весов; данные остаются локально.
bash scripts/download_models.sh --vlm mlx-community/Qwen3-VL-4B-Instruct-4bit
export HF_HUB_OFFLINE=1
export FORTE_VLM_BACKEND=mlx
export FORTE_VLM_MLX="$PWD/models/vlm/mlx-community--Qwen3-VL-4B-Instruct-4bit"
# Это окружение содержит и зависимости проекта, и закреплённый MLX-VLM.
python scripts/run_vlm.py --split train --out data/vlm_4b_trial
python scripts/confirm_vlm.py --src data/vlm_4b_trial --dst data/vlm_4b_trial_confirmed
cp models/scorer.json data/scorer_before_vlm_trial.json
FORTE_VLM_DIR=data/vlm_4b_trial_confirmed python scripts/train_scorer.py --vlm-confirmed --with-vlm
cp models/scorer.json data/scorer_vlm_trial.json
cp data/scorer_before_vlm_trial.json models/scorer.json
```

Предварительно измерить один вопрос через `python -m mlx_vlm.generate --model "$FORTE_VLM_MLX" --image LOCAL.jpg --prompt 'Is cash visible on the desk? Answer JSON with boolean key cash_on_desk.' --max-tokens 80 --temperature 0`.
Вместо LOCAL.jpg взять локальную мозаику из генератора ниже. Замерить cold start и warm вызовы,
peak memory, swap, число вопросов, полный upload-to-done на самом длинном видео и очереди из пяти.
Плановый резерв: 30–90 минут на A/B, **оценка, не измерение**. Не публиковать изображения, адаптеры и ответы с содержимым видео.

## Датасет и аудит разметки

```bash
# В окружении проекта; модели не загружаются, проверяются реальные кэши.
source .venv/bin/activate
python scripts/make_vlm_sft.py --dry-run --val-cabinets C,E --aug 0
python scripts/make_vlm_sft.py --out data/sft_ce --val-cabinets C,E --aug 0
```

Получаются `train.jsonl`, `validation.jsonl`, `images/`, `manifest.json`. Строка:
`{"images":["images/file.jpg"],"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"{...}"}]}`.
Это формат [текущего loader/CLI](https://github.com/Blaizzy/mlx-vlm/blob/73d366ecf9233b3fc6549531a7f2386726d09ff6/docs/training.md),
пути изображений разрешаются относительно папки split. Переносить всю папку, не только JSONL.

Это **предложения со слабыми метками**, не проверенный ground truth для кропа:
старые словари описывают целое видео, а нужный объект может отсутствовать в выбранных кадрах.
Перед обучением локально просмотреть каждую мозаику и ответ; удалить неоднозначные ключи или целые строки,
сохранить журнал проверки отдельно. Отсутствие документа в трёх кадрах не доказывает отсутствие предъявления во всём визите.
`None` исключается из ответа, не превращается в false. Не добавлять в SFT неподтверждённые VLM-ответы.

C/E никогда не попадают в train, аугментация только train (по умолчанию отключена), дубликаты 47/48
удаляются общим loader. Неизвестный кабинет, отсутствующий кадр, одинаковое изображение в разных split — ошибка.
Существующий непустой каталог не перезаписывается, чтобы не смешать старые split.
Манифест содержит кабинет, исходное видео, frame IDs, hash изображения и статус слабой метки.

## SFT и локальная проверка адаптера

После ручной проверки строк, в SFT-окружении:

```bash
source .venv-vlm/bin/activate
export HF_HUB_OFFLINE=1
python -m mlx_vlm.train --model "$PWD/models/vlm/mlx-community--Qwen3-VL-4B-Instruct-4bit" \
  --data data/sft_ce --algorithm sft --train-on-completions \
  --train-type lora --lora-rank 8 --batch-size 1 \
  --gradient-accumulation-steps 4 --iters 20 --learning-rate 1e-5 \
  --adapter-file models/vlm_lora_ce/weights.safetensors
```

20 microbatches — проверка загрузки/памяти, не заявка на качество. При успехе отдельный запуск на 1 эпоху
(`--epochs 1` вместо `--iters 20`); не включать `--train-vision` или full training.
По [реализации подготовки](https://github.com/Blaizzy/mlx-vlm/blob/73d366ecf9233b3fc6549531a7f2386726d09ff6/docs/training.md)
база заморожена, адаптеры по умолчанию добавляются к языковым linear-слоям.
Время 0.5–3 часа — **бюджетная оценка**, измерить первые шаги и пересчитать ETA; fit в 18 ГБ не гарантирован.

Проверка загрузки адаптера (его config должен лежать рядом с весами):

```python
from mlx_vlm import load, generate
from mlx_vlm.trainer import prepare_model_for_training
from mlx_vlm.prompt_utils import apply_chat_template
import json
from pathlib import Path
model, processor = load('models/vlm/mlx-community--Qwen3-VL-4B-Instruct-4bit')
model = prepare_model_for_training(model, checkpoint_path='models/vlm_lora_ce/weights.safetensors')
model.eval()
for line in Path('data/sft_ce/validation.jsonl').read_text().splitlines():
    row = json.loads(line)
    prompt = apply_chat_template(processor, model.config, row['messages'][:-1], num_images=1)
    result = generate(model, processor, prompt,
                      [str(Path('data/sft_ce') / row['images'][0])], max_tokens=90, temperature=0.0)
    print(json.dumps({'image': row['images'][0], 'expected': row['messages'][-1]['content'],
                      'predicted': getattr(result, 'text', str(result))}))
```

Повторить без `prepare_model_for_training` на той же базе для честного сравнения.
Runtime пока не принимает путь к адаптеру: не указывать его как `FORTE_VLM_MLX` и не обещать совместимость
старой команды fuse. Интеграция адаптера/экспорт — отдельное изменение только после проверки качества.

## Критерии принятия

C/E — 8 видео, holdout **только для нового SFT**: существующий track/scorer уже использовал эти кабинеты.
Сравнить base/adapted на одинаковых просмотренных строках: валидность JSON 100%, per-key TP/FP/FN,
recall при FP=0 не ниже base, отдельно число позитивов каждого ключа. Пустой support не означает успех.
Отчёт по признакам не заменяет event-level метрику на 8 полных видео: не увеличивать FP, не уменьшать recall,
проверить типы и таймкоды. Выбор эпох/промптов по C/E превращает их в development validation.

Нельзя обучить один адаптер на A/B/D/F/G и объявить LOCO по всем семи кабинетам независимой CV:
для каждого внешнего fold требуется заново обучить адаптер и все обучаемые upstream-компоненты без этого кабинета,
выбрать эпоху/пороги во внутренних folds. Текущий скрипт оценивает скорер с готовыми признаками и фиксированным
track_model; не считать его полной nested CV пайплайна. До такого сравнения SFT не принят.
Для базовой замены VLM минимум — не хуже воспроизведённой кабинетной метрики и FP=0.

Runtime: отсутствие таймаута не гарантируется. В `scripts/cashguard_analyzer.py` инициализация анализатора
предшествует расчёту бюджета, остаток ограничивается снизу 25 с, упаковка evidence идёт после analyze.
Очередь, cold start и base64 могут выйти за 300 с; проверить upload-to-done, при необходимости менять scheduler.
Не повышать wired memory limit и не запускать параллельно несколько VLM на 18 ГБ.
