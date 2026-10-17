# 02. Локальная VLM для yes/no-вопросов по CCTV-кадрам на Mac M3 Pro 18 GB (MLX)

Агент №2 из 10. Дата: 2026-10-17. Источники — карточки моделей на HF, README/доки mlx-vlm, GitHub-issues,
arXiv-отчёты, бенчмарки сообщества (ссылки в конце). Моделей не скачивал, `data/` не открывал.

**Ограничение исследования:** общий лимит веб-поиска сессии исчерпался до того, как я нашёл прямые замеры
mlx-vlm «картинка → токены/с» именно на M3 Pro. Все цифры скорости для M3 Pro ниже — **экстраполяция** с
M4 Max / M5 Max / M4 Pro по пропускной способности памяти (M3 Pro ≈ 150 GB/s, M4 Pro ≈ 273, M4 Max ≈ 546) и
помечены «≈». Их нужно подтвердить одним прогоном `mlx_vlm.generate --verbose` на целевой машине.

---

## 1. Итог: топ-3 для нашей задачи

| Ранг | Модель (репозиторий mlx-community, 4-bit) | Лицензия | Память (пик, инференс) | Скорость на M3 Pro, ≈ оценка | Зачем |
|---|---|---|---|---|---|
| **1** | `mlx-community/Qwen3.5-9B-MLX-4bit` (альт.: `mlx-community/Qwen3.5-9B-4bit`, конверсия mlx-vlm 0.3.12) | Apache-2.0 | ~6.0 GB (вес 5.6–6.0 GB) | декод ≈18–25 tok/s; 1 кадр 464×848 (~400 vis-токенов) ≈3–4 с; 3 кадра ≈6–9 с на вопрос | Лучшее мелкозернистое восприятие в классе ≤10B: RealWorldQA 80.3, V* 90.1, CountBench 97.2, HallusionBench 69.3. Нативно мультимодальная (не отдельный VL-форк), Apache-2.0 |
| **2** | `mlx-community/Qwen3.5-4B-MLX-4bit` (альт.: `mlx-community/Qwen3.5-4B-4bit`) | Apache-2.0 | ~5.2 GB при 4k ctx (вес ~3.0 GB) | декод ≈35–45 tok/s; 1 кадр ≈1.5–2.5 с; 3 кадра ≈3–5 с | Почти та же точность (RealWorldQA 79.5, V* 84.3/86.4, CountBench 96.3), в ~1.7× быстрее, идеальный кандидат для QLoRA на 18 GB. HallusionBench ниже (65.0) — чуть больше галлюцинаций |
| **3** | `mlx-community/Qwen3-VL-8B-Instruct-4bit` (лёгкая альтернатива: `mlx-community/Qwen3-VL-4B-Instruct-4bit`, 20.6k скачиваний — самая обкатанная) | Apache-2.0 | ~5.7–6.5 GB (8B); ~3.5–4.5 GB (4B) | 8B: декод ≈20–25 tok/s, 1 кадр ≈3–4 с; 4B: ≈35–45 tok/s, 1 кадр ≈1.5–2 с | Зрелая, самая проверенная в mlx-vlm линейка (поддержка с 0.3.4, нет thinking по умолчанию у Instruct, стабильный путь LoRA). Точность ниже Qwen3.5: RealWorldQA ~73, V* 77.5, CountBench 91.5 (8B, thinking-режим) |

**Рекомендация:** основной — Qwen3.5-9B 4-bit (если замеры на M3 Pro дадут ≤10 с на вопрос с 3 кадрами),
резервный/быстрый — Qwen3.5-4B 4-bit. Qwen3-VL-4B/8B Instruct держать как «страховку совместимости»: если
Qwen3.5 (гибридная архитектура Gated DeltaNet + attention) в текущем mlx-vlm даст баги, Qwen3-VL работает
гарантированно и имеет тот же препроцессор изображений (patch 16, merge 2 → 32×32 px на токен).

Текущий Qwen3-VL-2B в пайплайне — заменить: 4B того же поколения даёт +7–10 пунктов на RealWorldQA/V*, а
Qwen3.5-4B при той же памяти — ещё больше.

---

## 2. Сводная таблица всех кандидатов

Обозначения: «✓ mlx» — есть конверсия в `mlx-community`; «LoRA» — поддержка в `mlx-vlm` (доки: «fine-tuning
поддерживается для всех моделей, кроме Gemma3n и Qwen3-Omni»; явно перечислены Qwen2/2.5/3/3.5-VL, LLaVA,
Deepseek-VL, Mllama, Pixtral, Idefics3, SmolVLM).

| Модель | Лицензия → вердикт | Репо mlx-community 4-bit | Вес/пик памяти | Мультикадр | LoRA в mlx-vlm | Вердикт для нас |
|---|---|---|---|---|---|---|
| **Qwen3.5-9B** | Apache-2.0 → OK | `Qwen3.5-9B-MLX-4bit` (31.8k скач.), `Qwen3.5-9B-4bit`, `Qwen3.5-9B-OptiQ-4bit` | 5.6–6.0 GB / ~6 GB | да (Qwen2VL-процессор, interleaved) | да (Qwen3.5 VL в списке LORA.MD) | **№1** |
| **Qwen3.5-4B** | Apache-2.0 → OK | `Qwen3.5-4B-MLX-4bit` (39.8k), `Qwen3.5-4B-4bit` (32.3k) | 2.9–3.0 GB / 5.2 GB | да | да | **№2** |
| **Qwen3-VL-8B-Instruct** | Apache-2.0 → OK | `Qwen3-VL-8B-Instruct-4bit` (7.0k). Внимание: в репо discussion #2 — index.json описывает 4 шарда/17.5 GB, а лежит 2 шарда/5.8 GB; грузится, но проверить `load()` | ~5.8 GB / 6–7 GB | да | да | **№3** |
| Qwen3-VL-4B-Instruct | Apache-2.0 → OK | `Qwen3-VL-4B-Instruct-4bit` (20.6k) | ~2.8 GB / 4–5 GB | да | да | №3-lite, самый безопасный апгрейд с 2B |
| Qwen3-VL-30B-A3B-Instruct | Apache-2.0 → OK | `Qwen3-VL-30B-A3B-Instruct-4bit` | ~17–18 GB только веса | да | да | **нет**: не влезает в 18 GB вместе с ОС и YOLOX/RTMPose |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 → OK | `Qwen2.5-VL-7B-Instruct-4bit` (19.4k) | ~5 GB / 6–7 GB | да (полная поддержка в доках multi-image) | да | устарела: RealWorldQA ~68.5, HallusionBench ~52.9 (карточка Qwen); известны всплески памяти на больших картинках (lmstudio mlx-engine #98) |
| Gemma 4 E4B-it | **Apache-2.0** (Gemma 4 — первая Gemma под Apache) → OK | `gemma-4-e4b-it-4bit` (31k), `gemma-4-E4B-it-qat-4bit` | ~4.5 GB / 6.4 GB при 4k | «text and images in any order» | не исключена (исключения: Gemma3n, Qwen3-Omni) — проверить | B-кандидат на A/B: 127 tok/s на M5 Max, но опубликованы только MMMU-Pro 52.6 / OmniDocBench; нет данных по RealWorldQA/V*/CountBench. Бюджет 70–1120 vis-токенов/изобр. |
| Gemma 3 4B/12B-it | **Gemma Terms of Use** (не OSI: Prohibited Use Policy, право Google «restrict (remotely or otherwise) usage», обязательный notice при распространении) → **НЕТ** по правилам хакатона (только Apache/MIT/BSD) | `gemma-3-4b-it-qat-4bit`, `gemma-3-12b-it-qat-4bit` | 12B: ~7–8 GB | да | да | **исключить**; плюс слабое восприятие: RealWorldQA 45.5/52.2 (PT), CountBenchQA 26.1/17.8, картинка жёстко 896×896 → 256 токенов (мелкие объекты теряются) |
| InternVL3.5-8B | Apache-2.0 (карточка: «released under the apache-2.0 License», LLM — Qwen3) → OK | **нет 8B**: только `InternVL3_5-1B-4bit`, `InternVL3_5-30B-A3B-4bit`, `InternVL3_5-GPT-OSS-20B-A4B-Preview-4bit`. InternVL3-8B: `InternVL3-8B-MLX-4bit` | — | да | не в списке | **нет**: в compat-отчёте mlx-vlm (01.2026) InternVL3-14B упал с `InternVLImageProcessor … TypeError`; нет готовой 8B-конверсии |
| MiniCPM-V 4.5 / 4.6 | Карточка HF 4.5: «weights and code are open-sourced under the Apache-2.0 license», регистрация «optional». Старые версии шли под MiniCPM Model License (лимит 5000 устройств / 1M DAU) → **проверить LICENSE-файл в репо весов** перед использованием | 4.5: только `MiniCPM-o-4_5-4bit`; `MiniCPM-V-4.6-4bit` (623), `MiniCPM-V-4.7-1B-4bit`; 4.7-35B-A3B не влезает | 4.6: ~5–6 GB | да | не в списке | C-кандидат: OpenCompass 77.0 у 4.5, но лицензионная неоднозначность и меньше обкатки в mlx-vlm |
| Pixtral 12B | Apache-2.0 → OK | `pixtral-12b-4bit` (271) | ~7.5 GB / 8-bit 16 GB | да | да (Pixtral в списке) | слабее Qwen2.5-VL-7B на perception, 2024 г., медленнее (34 tok/s @8bit на M4 Max) → нет |
| Molmo-7B-D-0924 | Apache-2.0 → OK | `Molmo-7B-D-0924-4bit` (118) | **41 GB @8bit, 48 GB bf16** в mlx-vlm (патология памяти) | — | — | **нет** |
| SmolVLM2-2.2B | Apache-2.0 → OK | `SmolVLM2-2.2B-Instruct-mlx` | 5.5 GB; 116 tok/s на M4 Max | да | да | слишком слабая для мелких объектов; только как аварийный fallback по скорости |
| Phi-4-multimodal | MIT → OK | **нет** конверсии в mlx-community (есть только `Phi-3.5-vision-instruct-4bit`); в доках mlx-vlm поддержка заявлена | — | да (в списке multi-image) | ? | нет готового 4-bit, аудио-часть лишняя → нет |
| Idefics3-8B-Llama3 | Apache-2.0 → OK | `Idefics3-8B-Llama3-4bit` (27 скач.) | bf16 19 GB; 4-bit ~5 GB | да | да | 2024 г., уступает Qwen2.5-VL-7B → нет |
| Llama 3.2 11B Vision | **Llama 3.2 Community License** + для multimodal-моделей права **не предоставляются** компаниям/лицам из ЕС → **НЕТ** (не Apache/MIT/BSD) | `Llama-3.2-11B-Vision-Instruct-4bit` | 8-bit 15 GB; **8.3 tok/s @8bit на M4 Max** | ограниченно | да (Mllama) | **исключить**: лицензия + очень медленная |
| Kimi-VL-A3B-Instruct | MIT → OK | только Thinking: `Kimi-VL-A3B-Thinking-4bit`; Instruct нет | 16B total → ~9–10 GB @4bit | да | ? | **нет**: в compat-отчёте 01.2026 обе Kimi-VL упали (ImportError); RealWorldQA 68.1 |
| GLM-4.1V-9B-Thinking | MIT → OK | `GLM-4.1V-9B-Thinking-4bit` (110) | ~6 GB; до 4K-входа | не указано | ? | thinking-only → десятки секунд рассуждений на каждый yes/no; нет Instruct-варианта → нет. GLM-4.6V-Flash-4bit (`GLM-4.6V-Flash-4bit`, 46.7 tok/s @6bit M4 Max) — возможный B-кандидат, тоже thinking |
| Ovis2 | Apache-2.0 | **нет** (только `OvisOCR2-*`) | — | — | — | нет |

---

## 3. Бенчмарки мелкозернистого восприятия (чем выше, тем лучше)

Источники: карточки `Qwen/Qwen3.5-9B` и `Qwen/Qwen3.5-4B` (официальные таблицы), для Qwen3-VL 4B/8B —
сводный gist по техотчёту (**Thinking-варианты**; Instruct обычно на 1–3 пункта ниже на perception, выше на
HallusionBench-подобных), Kimi-VL — карточка, Gemma 3 — карточка (PT-варианты).

| Бенчмарк | Qwen3.5-9B | Qwen3.5-4B | Qwen3-VL-30B-A3B | Qwen3-VL-8B (Th.) | Qwen3-VL-4B (Th.) | Gemini-2.5-Flash-Lite | Kimi-VL-A3B | Gemma-3-12B (PT) |
|---|---|---|---|---|---|---|---|---|
| MMBench EN v1.1 | **90.1** | 89.4 | 88.9 | 87.5 | 86.7 | 82.7 | 83.1 | — |
| RealWorldQA | **80.3** | 79.5 | 77.4 | 73.5 | 73.2 | 72.2 | 68.1 | 52.2 |
| V* (мелкие объекты, high-res) | **90.1 / 88.5** | 84.3 / 86.4 | 83.2 | 77.5 | 74.9 | 70.2 | — | — |
| HRBench4K / 8K | — | — | — | 72.4 / 68.1 | 73.5 / 67.1 | 77.8 / 75.5 | — | — |
| CountBench | **97.2** | 96.3 | 90.0 | 91.5 | 89.4 | 79.2 | — | 17.8 (CountBenchQA) |
| HallusionBench (устойчивость к галлюцинациям) | **69.3** | 65.0 | 66.0 | 65.4 | 64.1 | 64.5 | — | — |
| MMStar | **79.7** | 78.3 | 75.5 | 75.3 | 73.2 | 69.1 | 61.3 | — |
| DocVQA / OmniDocBench1.5 | 87.7 (Omni) | 86.2 (Omni) | 86.8 (Omni) | 95.3 (DocVQA) | 94.2 (DocVQA) | 92.5 | — | 82.3 |
| OCRBench | 89.2 | 85.0 | 83.9 | 819 | 808 | 825 | — | — |
| ERQA (embodied/spatial) | **55.5** | 54.0 | 45.3 | 46.8 | 47.3 | 44.3 | — | — |

Выводы: (1) Qwen3.5-9B/4B выигрывают у Qwen3-VL-8B везде, где важны мелкие объекты и счёт (V* +12,
CountBench +5); (2) разрыв 9B vs 4B мал (1–6 пунктов), крупнее всего на HallusionBench (+4.3) и V*;
(3) Gemma 3 и Kimi-VL на этих метриках далеко позади.

---

## 4. Скорость и память: что реально измерено и как переносится на M3 Pro

| Что измерено | Железо | Результат | Источник |
|---|---|---|---|
| Qwen3-VL-4B, картинка 1024² cold-prefill+ответ | M4 Max 128 GB (vllm-mlx) | 2.1 с (768²: 1.8 с; 448²: 1.2 с; 224²: 0.8 с); с кэшем vision-эмбеддингов 0.16 с | arXiv 2601.19139 |
| Qwen3-VL-8B, 1024², cold | M4 Max | 21.7 с 1-й ход (включая генерацию длинного ответа), 1.15 с при кэше | там же |
| Qwen3-VL-4B, видео 8 кадров | M4 Max | 3.6 с, 4.6 GB; 16 кадров 5.8 с, 6.2 GB | там же |
| Qwen3-VL-2B bf16 декод | M4 Max, mlx-vlm 0.3.10 | 71.9 tok/s, 12 GB | mlx-vlm issue #661 |
| SmolVLM2-2.2B | M4 Max | 116 tok/s, 5.5 GB | #661 |
| Llama-3.2-11B-Vision 8bit | M4 Max | 8.3 tok/s, 15 GB | #661 |
| Qwen3.5-4B 4bit, 4k ctx (текст) | M5 Max 128 GB, mlx-vlm 0.4.4 | 156 tok/s, 5.2 GB | Incept5/gemma4-benchmark |
| Gemma 4 E4B 4bit | M5 Max | 127 tok/s, 6.4 GB | там же |
| Qwen3.5-9B 4bit декод (текст) | Mac mini M4 Pro | 78–115 tok/s (oMLX / Rapid-MLX), ~50 tok/s на 512-ток. промпте в community-таблице; агрегаторы дают 32–42 | siliconscore / Rapid-MLX / willitrunai |
| Qwen3.5-9B 4bit пик памяти | — | ~6.0 GB (8bit: 10.4 GB) | willitrunai (агрегат) |
| Qwen3.5-27B 4bit | — | 16.8 GB → не для 18 GB | там же |

**Экстраполяция на M3 Pro 18 GB (150 GB/s, 18-core GPU):**

- Вес 4-bit: ~2 байта/параметр → 4B ≈ 3 GB, 8–9B ≈ 5.5–6 GB. Декод ограничен пропускной способностью:
  4B ≈ 35–45 tok/s, 8–9B ≈ 18–25 tok/s. Ответ-JSON в 20–40 токенов — ~1–2 с.
- Токены изображения у Qwen3-VL/Qwen3.5: patch 16, merge 2 → один токен на 32×32 px. Кадр 464×848 (округление
  до кратного 32: 480×864) → **15×27 ≈ 405 токенов**. Три кадра ≈ 1200 токенов. Промпт ≈ 150 токенов.
- Prefill (vision-encoder + LLM) — ограничен вычислением; M3 Pro ≈ 2.5–3× медленнее M4 Max. Отсюда
  ≈1.5–2.5 с на 1 кадр для 4B и ≈3–4 с для 8–9B; на 3 кадра ≈3–5 с и ≈6–9 с соответственно.
- Бюджет 5 мин/видео: при 10–20 вопросов на видео и ≤9 с на вопрос VLM-часть укладывается в 1.5–3 мин.
  Запас остаётся, но **обязательно замерить** на реальном Mac: `python -m mlx_vlm.generate --model … --image a.jpg b.jpg c.jpg --prompt … --verbose` печатает prompt tok/s, generation tok/s и peak memory.
- Память при одновременной работе с YOLOX/RTMPose/CLIP (ONNX, ~1–2 GB): 9B 4-bit (≈6–7 GB пик) + ОС (~4 GB) —
  помещается; оставить `--prefill-step-size 1024` и `--max-kv-size 4096`, чтобы срезать пики.

---

## 5. Галлюцинации, формат ответа, JSON

- **HallusionBench**: Qwen3.5-9B 69.3 > Qwen3-VL-8B 65.4 ≈ Qwen3.5-4B 65.0 > Qwen3-VL-4B 64.1 ≫ Qwen2.5-VL-7B
  (~53). Для 4B-моделей разумно оставить self-consistency или перейти на logprob-скоринг (ниже).
- Qwen3-VL 2B в стороннем исследовании (WADE, zero-shot) показал высокий CHAIR (0.68) — склонность к
  перечислению несуществующих объектов в открытых описаниях; на закрытых yes/no это проявляется как «да» на
  наводящие вопросы. Митигация — нейтральная формулировка и явный третий вариант `unclear`.
- **Thinking-режим**: Qwen3.5 по умолчанию «думает» (`<think>`) — для yes/no это лишние сотни токенов и
  известный баг mlx-vlm 0.6.0/0.6.1: `response_format=json_schema` + thinking → HTTP 500 `ParserTooComplex`
  или невалидный JSON (issue #1294; закрыт PR #1299 05.06.2026). В `mlx_vlm.generate` флаг `--enable-thinking`
  по умолчанию **False**, в Python/сервере — `chat_template_kwargs={"enable_thinking": False}`. Qwen3-VL-Instruct
  thinking не имеет вообще (плюс для стабильности).
- **Строгий JSON**: в `mlx_vlm.server` есть `response_format: {"type":"json_schema", ...}` (constrained decoding
  через llguidance) на `/v1/chat/completions` и `/v1/responses`; несовместим со speculative decoding. В Python
  API — проще: greedy + схема в промпте + парсер с fallback на regex `"(yes|no|unclear)"`.
- **Logprobs**: сервер mlx-vlm отдаёт log probabilities → вместо 2-из-3 сэмплирования можно брать
  `P(yes)/(P(yes)+P(no))` первого токена ответа как калиброванный скор. Один forward вместо трёх, детерминизм.

---

## 6. Мультикадр

- mlx-vlm: несколько `--image` (CLI) или список `images` + `num_images=len(images)` в `apply_chat_template`
  (Python). В доках «полная поддержка» заявлена для Qwen2/2.5-VL, LLaVA(-Next), Idefics2/3, Phi-4-MM; Qwen3-VL
  и Qwen3.5 используют тот же `Qwen2VLImageProcessorFast` и interleaved-шаблон — работает, но проверить на
  текущей версии. Модели без нативной поддержки «обрабатывают только первое изображение» — молча.
- **Надёжнее для нашего кейса — мозаика**: склеить 2–3 кадра в одно изображение с подписями `t-2s | t | t+2s`
  (или `before / after`) поверх. Стоимость в токенах та же (~405 на кадр), зато нет зависимости от мультикадровых
  путей в mlx-vlm/LoRA-тренере и проще задавать вопросы «что изменилось».

---

## 7. Fine-tuning (LoRA/QLoRA) в mlx-vlm

| Параметр | Значение |
|---|---|
| Поддержка | SFT/ORPO/DPO/GRPO с LoRA, QLoRA, DoRA, full FT; `pip install "mlx-vlm[train]"`; всё, кроме Gemma3n и Qwen3-Omni; явно: Qwen2/2.5/3/3.5-VL, LLaVA, Deepseek-VL, Mllama, Pixtral, Idefics3, SmolVLM |
| CLI | `python -m mlx_vlm.lora --model-path <repo> --dataset <hf-id|path> --batch-size 1..4 --epochs N --learning-rate 2e-5 (LoRA) / 2e-4 (QLoRA) --lora-rank 8 (def.) --lora-alpha 16 --grad-checkpoint --image-resize-shape 480 864 --max-seq-length 2048 --train-on-completions --output-path adapters.safetensors`; `--train-vision` — дообучать и энкодер |
| Датасет | колонки `images` + `messages` (chat-формат модели); либо `question/answer` + `--custom-prompt-format` |
| Инференс с адаптером | `mlx_vlm.generate --model <repo> --adapter-path ./adapters.safetensors …`; сервер: `mlx_vlm.server --adapter-path` |
| Память, оценка для 18 GB | QLoRA на 4-bit **4B** (Qwen3.5-4B / Qwen3-VL-4B): веса 3 GB + LoRA + активации при batch 1–2, 1 кадр 480×864 (~400 vis-токенов), `--grad-checkpoint` → **≈7–11 GB пик — комфортно**. QLoRA на 4-bit **8–9B**: ≈12–16 GB пик при batch 1 + grad-checkpoint — **на грани**, закрыть всё остальное; при 2–3 кадрах в примере — скорее не влезет. Full FT / LoRA на bf16 — нет. (Единственная найденная внешняя оценка: «M3 Pro 36 GB: ~14 GB LoRA, ~9 GB QLoRA» без указания модели.) |
| Время, оценка | 4B, batch 1, ~600 токенов/пример: ≈3–8 с/шаг на M3 Pro → 500–1000 шагов = 1–2 ч. 9B ≈ в 2 раза дольше |
| Риски | Реальных отчётов о времени/памяти LoRA VLM на 16–18 GB Mac не найдено — только документация. Qwen3.5 (гибридная архитектура) в тренере новее, чем Qwen3-VL; 4B-конверсия `Qwen3.5-4B-MLX-4bit` собрана из ветки `pc/fix-qwen35-predicate` — брать свежий `mlx-vlm` (≥0.6.x) |
| Что имеет смысл дообучать | НЕ «детектор новых типов нарушений» (на 21 позитивном видео переобучится — это уже показал опыт с VLM-2B и CLIP-probe). Имеет смысл **узкая калибровка формата и yes/no** на 300–1000 пар «кадр/мозаика + вопрос → {"answer": …}» с hold-out по кабинету: учит строго отвечать JSON, использовать `unclear`, не поддаваться наводящим вопросам. Rank 8–16, 1–2 эпохи, lr 1e-4–2e-4 |

---

## 8. Рекомендуемая настройка промпта и декодинга

**Модель/рантайм:** `mlx-community/Qwen3.5-9B-MLX-4bit` (fallback `Qwen3.5-4B-MLX-4bit`), свежий `mlx-vlm`
(≥0.6.x, где закрыт #1294), thinking выключен.

**Изображения (мелкие объекты на кадре 464×848):**
1. Кадр подавать **в нативном разрешении** — не даунскейлить: задать процессору `min_pixels = 256*32*32`
   (262 144) и `max_pixels = 1280*32*32` (1 310 720); наш кадр (393 k px) пройдёт без ресайза → ~405 токенов.
2. Для сущностей размером <40 px (ключ в замке, купюры на столе, мешок инкассации) добавлять **ROI-кроп** из
   нашего же трекера (зона стола / сейфа / кассира относительно seat-координат), апскейл ×2 до ~640×640
   (~400 токенов). Подавать как мозаику «полный кадр | кроп» или как два изображения. Апскейл сам информации не
   добавляет, но заставляет модель потратить на ROI в 4–8 раз больше визуальных токенов.
3. Временные вопросы («дверь закрылась после входа?», «деньги остались на столе, когда зашёл следующий?») —
   мозаика 2–3 кадров с подписями времени, один вопрос — одна мозаика.
4. Несколько вопросов к одним и тем же кадрам — **один промпт с одним JSON-объектом ответов** (один prefill
   вместо N). В сервере включить `APC_ENABLED=1` (prefix cache) — системный промпт кэшируется; кэш
   vision-эмбеддингов даёт до 19× на повторных вопросах по тому же изображению (vllm-mlx).

**Промпт (системный):**
```
You are a strict visual auditor of a bank cashier workplace on overhead CCTV frames.
Answer ONLY with JSON matching the schema. Use "unclear" when the frame does not show enough evidence.
Do not assume anything that is not visible. Never guess.
```
**Пользовательский:** `Frames: [t-2s | t | t+2s]. Question: Is there cash lying on the desk in frame "t"
while no client is present? Schema: {"answer":"yes"|"no"|"unclear","evidence":"<=12 words"}`

Правила формулировок: без наводящих «подтвердите, что…»; симметричные формулировки (половина вопросов
сформулирована через «есть», половина через «нет», либо спрашивать сразу обе стороны); всегда три класса.

**Декодинг:** `temperature=0` (greedy), `max_tokens=48`, `repetition_penalty=1.0`, thinking off; в сервере
`response_format=json_schema` с `enum` для `answer`. Скор уверенности — logprob первого токена `answer`
(порог 0.65–0.7 на «yes» для минимизации FP; ниже → `unclear` → не создавать событие). Если logprobs
недоступны в Python API — оставить 2-из-3, но с `temperature=0.3` и тремя разными ROI/мозаиками, а не
тремя сэмплами одного промпта (так самосогласованность реально проверяет восприятие, а не шум).

**Контроль памяти:** `--prefill-step-size 1024`, `--max-kv-size 4096`, выгружать модель между видео не нужно
(загрузка 9B 4-bit ≈ 3–6 с с SSD, лучше держать резидентно).

---

## 9. Что осталось непроверенным (сделать на целевом Mac за 30 минут)

1. `mlx_vlm.generate --verbose` для `Qwen3.5-9B-MLX-4bit`, `Qwen3.5-4B-MLX-4bit`, `Qwen3-VL-8B-Instruct-4bit`
   с 1 и 3 кадрами 464×848: prompt tok/s, gen tok/s, peak memory. Это заменит все «≈» выше.
2. Мультикадровый вызов с `num_images=3` для Qwen3.5 на текущей версии mlx-vlm (или сразу мозаика).
3. `response_format=json_schema` в `mlx_vlm.server` с `enable_thinking=False` на Qwen3.5 (после фикса #1299).
4. 20–30 контрольных вопросов из наших «промахов» (ключ в замке, купюры на столе, мешок в сейф) на трёх моделях —
   сравнить accuracy/FP и выбрать между 9B и 4B.
5. Для `Qwen3-VL-8B-Instruct-4bit` убедиться, что `load()` не спотыкается о несогласованный `model.safetensors.index.json`
   (discussion #2 в репо); при проблемах — `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit`.

---

## Источники

- mlx-vlm README (поддерживаемые модели, json_schema, APC, fine-tuning): https://github.com/Blaizzy/mlx-vlm
- Доки mlx-vlm (зеркало): LoRA https://yocxy2-mlx-vlm.mintlify.app/fine-tuning/lora ; overview https://yocxy2-mlx-vlm.mintlify.app/fine-tuning/overview ; CLI https://yocxy2-mlx-vlm.mintlify.app/inference/cli ; multi-image https://yocxy2-mlx-vlm.mintlify.app/inference/multi-image ; server https://yocxy2-mlx-vlm.mintlify.app/inference/server
- mlx-vlm compat report 2026-01-16 (M4 Max, память/скорость): https://github.com/Blaizzy/mlx-vlm/issues/661
- mlx-vlm issue json_schema + thinking: https://github.com/Blaizzy/mlx-vlm/issues/1294
- Gemma3 fine-tuning issue: https://github.com/Blaizzy/mlx-vlm/issues/245
- Qwen3.5-9B карточка (бенчмарки, thinking, лицензия): https://huggingface.co/Qwen/Qwen3.5-9B ; Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B ; GitHub Qwen3.5 (MLX/llama.cpp support): https://github.com/QwenLM/Qwen3.5
- mlx-community/Qwen3.5-9B-MLX-4bit: https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit ; Qwen3.5-4B-MLX-4bit: https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit
- Qwen3-VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ; 4B: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct ; preprocessor_config (patch 16 / merge 2): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/preprocessor_config.json
- mlx-community/Qwen3-VL-8B-Instruct-4bit: https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit (discussion #2 про index.json: https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit/discussions/2) ; 4B: https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit
- Сводка бенчмарков Qwen3-VL (gist по техотчёту): https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3 ; техотчёт: https://arxiv.org/abs/2511.21631
- vllm-mlx paper (латентность картинок на M4 Max, кэш vision-эмбеддингов): https://arxiv.org/html/2601.19139v2
- Incept5 MLX benchmark Gemma 4 / Qwen3.5 (M5 Max): https://github.com/Incept5/gemma4-benchmark
- Qwen 3.5 MLX агрегат (память/скорость): https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide ; Rapid-MLX (M4 Pro decode): https://github.com/raullenchai/Rapid-MLX
- Gemma Terms of Use: https://ai.google.dev/gemma/terms ; Gemma 4 Apache-2.0: https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html ; google/gemma-4-E4B-it: https://huggingface.co/google/gemma-4-E4B-it ; gemma-3-12b-it (PT vision table): https://huggingface.co/google/gemma-3-12b-it
- Llama 3.2 EU-ограничение: https://huggingface.co/blog/llama32 ; https://developer.meta.com/ai/faq/
- InternVL3_5-8B лицензия: https://huggingface.co/OpenGVLab/InternVL3_5-8B
- MiniCPM-V-4_5 лицензия: https://huggingface.co/openbmb/MiniCPM-V-4_5 (контраст: https://docs.voxel51.com/plugins/plugins_ecosystem/minicpm_v.html)
- Kimi-VL-A3B-Instruct: https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct ; GLM-4.1V-9B-Thinking: https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking
- WADE (CHAIR у Qwen3-VL-2B): https://arxiv.org/pdf/2608.22950
- Списки репозиториев mlx-community получены через HF API `https://huggingface.co/api/models?author=mlx-community&search=…`
