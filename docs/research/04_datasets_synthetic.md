# 04. Открытые датасеты и синтетика для сценария «кассир / наличные / сейф» (трек data)

Дата: 2026-10-17. Агент #4. Все лицензии проверены на первичных страницах (ссылки в таблицах), если не сказано иное.
Ограничение: лимит WebSearch в сессии был исчерпан после ~12 запросов; дальнейшая проверка шла только через прямые
WebFetch первичных страниц. Roboflow Universe (поиск), Kaggle и сайт HICO отдают 403/JS — для них указано, что именно
удалось подтвердить.

Условные обозначения вердикта: **OK** — пермиссивная лицензия, можно использовать для коммерческого продукта;
**OK\*** — аннотации пермиссивны, изображения под чужими правами (Flickr/YouTube), т.е. скачивать по URL, не
редистрибутировать; **NO** — NC / research-only / договор; **?** — лицензия не указана (по умолчанию считаем NO).

---

## 1. Кратко (TL;DR)

1. **Ни один открытый датасет не содержит нашей сцены** (overhead-камера, кассир за столом, наличные/сейф/удостоверение
   в контексте) с пермиссивной лицензией. Ближайшее по сцене — MEVA (CC BY 4.0, indoor+outdoor CCTV, активности
   `person_transfers_object`, `hand_interacts_with_person`, `person_opens/closes_facility_door`, `person_talks_on_phone`,
   `person_purchases`, `person_steals_object`) и CAVIAR (CC BY-SA, 384x288, 2004 г., холл/торговый коридор).
2. **По объектам** пермиссивно доступны только «общие» классы: LVIS (`money`, `dollar`, `identity_card`, `passport`,
   `key`, `padlock`, `receipt`, `handbag`, `briefcase`, `cellular_telephone`, `cup`, `mug`, `drawer`, `cabinet`,
   `file_cabinet`, `envelope`, `cash_register`) и Open Images (`Coin`, `Filing cabinet`, `Drawer`, `Door`, `Door handle`,
   `Handbag`, `Briefcase`, `Mobile phone`, `Coffee cup`, `Mug`, `Envelope`, `Calculator`). Класса **«Safe/сейф» нет ни в
   Open Images (601 класс), ни в LVIS (1203), ни в Objects365 (365)**; «Banknote» нет ни в одном; `money`/`dollar` в LVIS
   — редкие/common, в основном макро/ракурс «от первого лица».
3. **Единственный точечный «подарок»**: два датасета **тенге** на Mendeley Data под **CC BY 4.0** (KZ-BD, 4200 фото,
   14 классов, YOLO-разметка; Kazakh Banknote Image Dataset, 6 номиналов + Mix, по 100+ фото на класс). Это фото с телефона
   на «грязном» фоне, не overhead, но это готовые вырезки купюр для cut-and-paste аугментации.
4. **Синтетика на 2 дня**: реалистично только (а) cut-and-paste композит реальных вырезок (купюры тенге из Mendeley,
   собственные фото ключа/сумки/удостоверения) на собственные кадры стола и (б) text-to-image / image-edit под Apache-2.0
   (FLUX.1-schnell 12B, FLUX.2-klein-4B с редактированием, Z-Image-Turbo 6B; Qwen-Image / Qwen-Image-Edit 20B — Apache-2.0,
   но тяжелы для 18 ГБ) для генерации кропов `cash bundle on desk from above`, `key in safe lock`. 3D-пайплайны
   (Unity/Unreal/Blender + аватары) за 2 дня не окупятся: BEDLAM и SMPL-X — **NC**, BlenderProc — GPL-3 (как
   инструмент допустим, но гуманоидов нет), Infinigen Indoors — BSD-3, но без людей и тяжёлый.
5. **Рекомендация**: основной источник — **собственная досъёмка** (сценарии пропущенных типов, 2–3 ракурса, разные
   люди/одежда); внешнее — только (i) купюры тенге CC BY для cut-and-paste и доучивания маленького классификатора
   «деньги на столе / в руках», (ii) MEVA как источник негативов/претрена для hand-interaction детектора при наличии
   времени, (iii) LVIS-предобученные веса детекторов (аннотации CC BY 4.0) — для zero-shot классов `money`, `key`,
   `handbag`, `identity_card` (см. также 01_open_vocab_detectors.md).

---

## 2. Датасеты: лицензии, полезные классы, размеры

### 2.1. Большие детекционные / сегментационные датасеты

| Датасет | Лицензия (первичная страница) | Вердикт | Полезные классы (точные имена) | Размер | Комментарий / ссылка |
|---|---|---|---|---|---|
| **Open Images V7** | Аннотации: CC BY 4.0 (Google). Изображения: «listed as CC BY 2.0», Google «makes no representations» — проверять per-image | **OK\*** | `Coin`, `Filing cabinet`, `Cabinetry`, `Cupboard`, `Drawer`, `Door`, `Door handle`, `Handbag`, `Briefcase`, `Backpack`, `Suitcase`, `Luggage and bags`, `Plastic bag`, `Mobile phone`, `Telephone`, `Corded phone`, `Coffee cup`, `Mug`, `Desk`, `Table`, `Chair`, `Human hand`, `Envelope`, `Box`, `Calculator`, `Computer keyboard`, `Computer monitor`, `Laptop`, `Office supplies`. **НЕТ**: Safe, Banknote, Money, Cash, Key, Lock, Wallet, Cup, Identity card, Passport, Document | 601 boxable-класс, ~1.9M изобр., 16M боксов | [class list CSV](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv), [licenses](https://storage.googleapis.com/openimages/web/factsfigures_v7.html). Сцены — Flickr, ракурс бытовой, не overhead |
| **LVIS v1** | Аннотации CC BY 4.0; изображения = COCO (Flickr, CC-лицензии разные) | **OK\*** | `money` (id 697, c), `dollar` (381, r; синонимы dollar_bill), `coin` (288, c), `identity_card` (211, c), `passport` (770, c), `receipt` (877, c), `key` (605, f), `padlock` (746, c), `combination_lock` (292, r), `deadbolt` (124, c), `handbag` (35, f; purse), `briefcase` (153, c), `tote_bag` (218, f), `shopping_bag` (949, f), `plastic_bag` (898, f), `pouch` (841, c), `satchel` (913, r), `backpack` (34, f), `suitcase` (36, f), `cellular_telephone` (230, f), `telephone` (1072, f), `cup` (344, f), `mug` (708, f), `teacup`, `saucer`, `desk` (361, f), `drawer` (390, f), `cabinet` (181, f), `file_cabinet` (438, c), `cupboard`, `doorknob` (385, f), `envelope` (424, c), `clipboard`, `calculator`, `notebook`, `notepad`, `pen`, `cash_register` (221, c), `stapler_(stapling_machine)`, `monitor_(computer_equipment)`. **НЕТ**: safe/strongbox, banknote, cash, wallet, door | 1203 класса, 164k изобр., ~2M масок | [категории (detectron2)](https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py). Самое ценное — не данные, а **открытые веса LVIS-детекторов** для zero-shot `money/key/identity_card/handbag` |
| **COCO** | Аннотации CC BY 4.0; изображения Flickr (ToU) — страница ToU не отдала текст, лицензия общеизвестна | **OK\*** | `person`, `handbag`, `backpack`, `suitcase`, `cell phone`, `cup`, `bottle`, `laptop`, `keyboard`, `mouse`, `chair`, `book` | 80 классов, 123k изобр. | Уже в пайплайне (YOLOX COCO). Нет денег/сейфа/ключа/удостоверения |
| **Objects365** | Страница download: «available for the academic purpose only»; аннотации — CC BY 4.0 (Consortium), изображения — Flickr ToU, «You will NOT distribute the above images»; регистрация на Biendata | **NO** (academic-only оговорка перекрывает CC BY на аннотации) | `Wallet/Purse` (238), `Key` (251), `Cell Phone` (61), `Cup` (10), `Cabinet/shelf` (12), `Handbag/Satchel` (13), `Briefcase` (194), `Luggage` (120), `Backpack` (38), `Folder` (205), `Notepaper` (281), `Stapler` (306), `Calculator` (316), `Desk` (9). **НЕТ**: safe, money, coin, door, drawer | 365 классов, 2M изобр. | [download/terms](https://www.objects365.org/download.html). Веса, предобученные на O365, тоже тащат этот хвост — избегать |
| **V3Det** | Аннотации/дерево категорий/инструменты: CC BY 4.0 («allows commercial use»); изображения — только URL (Flickr ToU), «we only provide lists of image URLs without redistribution» | **OK\*** | 13 204 категории (6709 base + 6495 novel) — хвостовые классы вероятно включают safe/banknote/key-lock, но список категорий (JSON) не проверялся | train 183k / val 30k / test 30k изобр., HF-бэкап 54 ГБ (только аннотации) | [GitHub](https://github.com/V3Det/V3Det), [HF yhcao/V3Det_Backup](https://huggingface.co/datasets/yhcao/V3Det_Backup). Нужно качать изображения по URL — на 2 дня неоправданно |
| **BigDetection** | Код Apache-2.0; данные = объединение LVIS + Open Images + Objects365 → наследует **academic-only** от O365 | **NO** (из-за O365) | 600 классов | 3.4M изобр. | Веса на BigDetection — не использовать |
| **ADE20K** | Изображения: «Researcher shall use the Database only for non-commercial research and educational purposes»; код/аннотации BSD-3 | **NO** | сцены office/bank — бесполезно при NC | 25k изобр. | [terms](https://ade20k.csail.mit.edu/terms/) |
| **Visual Genome** | CC BY 4.0 («Visual Genome by Ranjay Krishna is licensed under a CC BY 4.0») | **OK\*** (изображения Flickr/COCO) | objects/attributes/relationships — «man holding money», «person handing» и т.п. в графе сцен, но качество разметки шумное | 108k изобр. | [about](https://homes.cs.washington.edu/~ranjay/visualgenome/about.html). Для нашей задачи ценности мало |
| **SA-1B** | «SA-1B Dataset Research License», intended use «Research purposes only» | **NO** | — | 11M изобр. | [Meta page](https://ai.meta.com/datasets/segment-anything/), [README](https://github.com/facebookresearch/segment-anything). Сама модель SAM — Apache-2.0 (это можно) |
| **ImageNet-21k / 1k** | ImageNet terms: non-commercial research | **NO** | — | — | Предобученные на ImageNet бэкбоны — отдельный юридический вопрос (общепринято считать допустимым, но данные качать нельзя) |
| **LAION-400M/5B** | Метаданные CC BY 4.0; изображения — чужие права, только URL | **OK\*** (метаданные) | текстовый поиск по caption «cash on desk», «bank teller counter» возможен через clip-retrieval, но без гарантии прав на картинки | — | На 2 дня: не стоит |

### 2.2. Видео / действия / наблюдение

| Датасет | Лицензия | Вердикт | Что внутри полезного | Размер | Комментарий |
|---|---|---|---|---|---|
| **MEVA (Multiview Extended Video with Activities)** | «All MEVA data is available for use under a CC BY-4.0 license» | **OK** | Стационарные EO-камеры, **indoor и outdoor**, активности ActEV (37): `person_transfers_object`, `hand_interacts_with_person`, `person_purchases`, `person_steals_object`, `person_abandons_package`, `person_picks_up_object`, `person_puts_down_object`, `person_opens_facility_door`, `person_closes_facility_door`, `person_talks_on_phone`, `person_texts_on_phone`, `person_reads_document`, `person_interacts_with_laptop`, `person_sits_down`, `person_stands_up`, `person_carries_heavy_object`, `person_talks_to_person`, `person_embraces_person` + vehicle-классы | 328 ч ground-camera (516 ГБ); размечено 64 ч eval (769 5-мин клипов) + 120 ч train (1443 клипа) | [mevadata.org](https://mevadata.org/), [аннотации (GitLab Kitware)](https://gitlab.kitware.com/meva/meva-data-repo), [список ActEV](https://actev.nist.gov/sdl). Ближайшее к CCTV, но сцена — учебный городок (Muscatatuck), не касса; камеры дальние, объекты крупные (коробки, сумки). Годится как претрен/негативы для детектора передачи предметов между людьми, не как замена своих данных |
| **VIRAT Ground 2.0** | «VIRAT Video Dataset Protection Agreement» (договор, не открытая лицензия) | **NO** (без подписания) | 12 типов событий, outdoor | 8.5 ч | [viratdata.org](https://viratdata.org/) — только outdoor, парковки |
| **CAVIAR (INRIA / Lisbon shopping centre)** | «Usability is Creative Commons BY-SA» + просьба цитировать | **OK с оговоркой** (BY-SA: ShareAlike касается производных данных; модель — спорно, но риска для продукта минимум) | Постановочные сценарии: оставление сумок/коробок, встречи, драки, падение, вход/выход в магазины; 2 синхронных ракурса (вдоль коридора + фронт) | 384x288, 25 fps, MPEG2; десятки клипов | [страница](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/). 2004 г., качество низкое; полезно только как внешняя проверка робастности трекера к ракурсу |
| **AVA Actions / AVA-Kinetics** | Общеизвестно CC BY 4.0 на аннотации (страница загрузки не отдала текст — не подтверждено в этой сессии); видео YouTube-фильмы | **OK\*** (не подтверждено) | атомарные действия: give/serve object to person, take object from person, answer phone, text on phone, drink, open/close door, hand shake, hand wave, write | 430 клипов по 15 мин, 1.62M меток | [AVA](https://research.google.com/ava/). Киноракурсы, не CCTV; перенос на overhead слабый |
| **Kinetics-400/600/700** | Readme K700 лицензии не содержит; общеизвестно CC BY 4.0 на аннотации, видео YouTube | **OK\*** (не подтверждено) | классы `counting money`, `texting`, `drinking`, `shaking hands`, `writing`. **НЕТ** door/keys/handing objects | «counting money» ~1000 клипов | [label map](https://github.com/deepmind/kinetics-i3d/blob/master/data/label_map.txt). «counting money» — руки с купюрами крупным планом; потенциальный источник кропов «купюры в руках» (YouTube-права!) |
| **UCF-Crime** | Страница недоступна (сертификат); лицензия на сайте CRCV не публикуется, видео с YouTube/LiveLeak | **NO/?** | 13 аномалий: Burglary, Robbery, Shoplifting, Stealing — есть indoor-CCTV магазинов и касс | 1900 видео, 128 ч | Права на кадры не чистые — не брать |
| **ShanghaiTech Campus** | Лицензия не указана, только просьба цитировать | **?** → NO | кампус, outdoor | 13 сцен, 270k кадров | [страница](https://svip-lab.github.io/dataset/campus_dataset.html) |
| **CUHK Avenue** | Лицензия не указана | **?** → NO | outdoor аллея | 37 клипов | [страница](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| **UCSD Ped1/Ped2** | Сайт недоступен (ECONNREFUSED) | **?** | outdoor пешеходы | — | Не релевантно |
| **MMPTRACK** (indoor multi-camera, retail/office/lobby) | Страница воркшопа: «will be released to the research community», лицензия не указана | **?** → NO | — | — | [iccv2021-mmp](https://iccv2021-mmp.github.io/) |
| **Ego4D** | Отдельный лицензионный договор (ego4d.dev/request); текст не удалось получить; общеизвестно — research/non-commercial | **NO** | — | — | — |
| **EPIC-Kitchens** | CC BY-NC 4.0 | **NO** | — | — | — |
| **MECCANO** | Лицензия не указана ни на сайте, ни в GitHub | **?** → NO | — | — | [сайт](https://iplab.dmi.unict.it/MECCANO/) |
| **HICO / HICO-DET** | Сайт отдал 403; общеизвестно: изображения Flickr, аннотации без явной лицензии (academic) | **?** → NO | HOI `hold cell_phone`, `carry handbag`, `hold cup` | 47k изобр. | — |

### 2.3. Передача предметов из рук в руки (handover)

| Датасет | Лицензия | Вердикт | Содержимое | Комментарий |
|---|---|---|---|---|
| **Dataset of Bimanual Human-to-Human Object Handovers** (Zenodo 7767535) | CC BY 4.0 | **OK** | 12 пар, 240 бимануальных + 120 одноручных передач, 10 объектов (цветочный горшок, посуда, ноутбук, игрушка, роутер, мяч, одежда, коробка, доска, рюкзак); 2× Kinect v2 RGB 1920x1080 + depth, 30 fps; mocap 120 Гц; **зелёный хромакей, стоя лицом к лицу** | [Zenodo](https://zenodo.org/records/7767535), [статья](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294053/). 8.9 ГБ. Хромакей = можно вырезать и композитить, но ракурс фронтальный/боковой, не overhead, и объекты крупные — не купюры |
| **Multi-sensor Dataset of Sequential Human-to-Human Handovers (shelving)** (Zenodo 7895500) | CC BY 4.0 | **OK** | 12 пар × 30 передач = 1440; 2× RGB-D 30 Гц + mocap | [Zenodo](https://zenodo.org/records/7895500). 58.7 ГБ. Та же лаборатория, лабораторная сцена |
| **HOH (Human-Object-Human)** | CC BY-NC-SA 4.0 | **NO** | 2720 передач, multi-view RGB-D | — |
| **H2O benchmark** (18k клипов) | лицензия/релиз не подтверждены | **?** | — | [arXiv](https://arxiv.org/abs/2104.11466) |
| HF `*handover*` (50 результатов) | Apache-2.0 | OK, но **роботы** (LeRobot/ALOHA/SO-100) | — | Бесполезно для людей |

### 2.4. Банкноты / наличные

| Датасет | Лицензия | Вердикт | Содержимое | Комментарий |
|---|---|---|---|---|
| **KZ-BD: Dataset of Kazakhstan Banknotes with Annotations** (Mendeley `dny3dgvvw8`) | **CC BY 4.0** | **OK** | 4200 фото, **14 классов (1–20 000 KZT, монеты и купюры)**, YOLO-разметка, split 70/30; съёмка телефоном, разное освещение, «cluttered backgrounds», часть купюр сложена | [Mendeley](https://data.mendeley.com/datasets/dny3dgvvw8), [Data in Brief 2024](https://doaj.org/article/8de244d1afd64f4aaa7265f303505bac). **Главная внешняя находка**: готовые боксы купюр тенге → вырезки для cut-and-paste и обучение классификатора «купюры/не купюры» на кропах |
| **Kazakh Banknote Image Dataset** (Mendeley `gyw87k93yb`, авг. 2025) | **CC BY 4.0** | **OK** | классы 500/1000/2000/5000/10000/20000 + Mix (>1000 фото), 5000 разбит на new_single/new/old; ≥100 фото на класс; **в основном несколько купюр в кадре** (пачки/веер), разные фоны; manifest.csv, предзаданные сплиты | [Mendeley](https://data.mendeley.com/datasets/gyw87k93yb). Классификационный (без боксов), но «несколько купюр на фоне» ближе к сцене «пачка на столе» |
| Roboflow Universe: `fyp-leirw/banknote`, `philippine-banknotes`, `kkdoan111/malaysian-banknote`, `currency-note-identification`, `banknotesclasiffication/test_dataset` (PLN) | По ToS Roboflow публичные проекты без указанной лицензии = CC BY 4.0 (п. 23(B)); на страницах показан CC BY 4.0 | **OK** (проверять поле License на каждой странице; прямой фетч страниц отдаёт 403) | чужие валюты, макро/в руке | Тенге на Roboflow поиском не найдено (поиск ограничен). Чужие валюты бесполезны для классификатора тенге, но годятся как «деньги вообще» для бинарного детектора |
| HF: `RichMiguel/Banknotes-Coins` (MIT), `mariamyasir/banknote_dataset` (MIT), `Jakir057/bangladeshi_banknotes_70k` (без лицензии), `paowick/Banknote_thai` (без) | MIT / нет | OK (MIT) / NO | чужие валюты, макро | `farish07/banknote-authentication-dataset` — табличный UCI, не изображения |
| Kinetics «counting money» | см. выше | OK\* | руки с купюрами, видео YouTube | Права на кадры размыты — не для продукта |

### 2.5. Синтетические CCTV-датасеты (готовые)

| Датасет | Лицензия | Вердикт | Содержимое | Комментарий |
|---|---|---|---|---|
| **Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection** (HF) | CC BY 4.0 | OK | 112 синтетических overhead-CCTV кадров, падение/лежание | Метод генерации не раскрыт. Мало, не наша сцена |
| Simuletic/CCTV_Aggressive_Poses_Fight_Detection, cctv-knife/weapon (HF) | CC BY 4.0 | OK | синтетика, оружие/драки | Не релевантно |
| Simuletic/CCTV_ATM_Robbery_Detection, CCTV-Pedestrian-1K | **CC BY-NC 4.0** (на Kaggle противоречиво указаны CC BY и BY-NC-SA) | NO | — | — |
| Kaggle Simuletic «CCTV Shoplifting Detection Dataset — YOLO & VLM» | не удалось прочитать (JS) | ? | retail синтетика | Проверить вручную поле License на Kaggle |
| HF `*shoplifting*` (4 шт.), Ultralytics `cctv` (59 overhead-кадров офиса) | без лицензии / 403 | NO/? | — | — |

### 2.6. Что НЕ существует в открытом виде (проверено)

- Класс «сейф / safe / strongbox» — отсутствует в Open Images, LVIS, Objects365, COCO. Состояние двери сейфа
  (открыт/закрыт) — только своя разметка.
- «Банкнота/пачка купюр на столе сверху» в CCTV-контексте — нет. Есть только макро-/телефонные фото купюр.
- «Предъявление удостоверения клиентом» — нет (`identity_card`/`passport` в LVIS — common-классы, ~сотни экземпляров,
  в основном крупный план).
- «Ключ в замке сейфа» — нет. `key`/`padlock` в LVIS — бытовые планы.
- Overhead indoor офис/касса с людьми под пермиссивной лицензией — нет (MEVA ближе всего, но дальний план; CAVIAR —
  2004 г., 384x288).

---

## 3. Синтетика: варианты, лицензии, реалистичность за 2 дня

### 3.1. Лицензии инструментов и моделей

| Инструмент / модель | Лицензия (первичная) | Можно в продукт? | Примечание |
|---|---|---|---|
| **FLUX.1-schnell** (12B, 1–4 шага) | Apache-2.0 | да | [HF](https://huggingface.co/black-forest-labs/FLUX.1-schnell). FLUX.1-dev / Kontext-dev — NC. Через **mflux** на Apple Silicon с 8-бит квантованием |
| **FLUX.2-klein-4B** (T2I + редактирование по референсам, 4 шага) | Apache-2.0 («Fully open under Apache 2.0») | да | [HF](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). «~13 GB VRAM» — на M3 Pro 18 ГБ пойдёт с квантованием (mflux поддерживает FLUX.2 4B/9B). FLUX.2-dev — коммерческая лицензия, **не брать** |
| **Z-Image-Turbo** (6B, 8 NFE) | Apache-2.0 | да | [HF](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). «fits within 16G VRAM»; поддерживается mflux. Z-Image-Edit — «to be released» на момент проверки |
| **Qwen-Image** (20B) / **Qwen-Image-Edit** (20B) | Apache-2.0 | да | [HF](https://huggingface.co/Qwen/Qwen-Image), [Edit](https://huggingface.co/Qwen/Qwen-Image-Edit). Лучший «вставь объект / замени фон» из открытых, но 20B BF16 = 40 ГБ; на 18 ГБ только 4-бит GGUF и медленно. Не для 2 дней на Mac |
| **SD 1.5 / 2.1** | CreativeML OpenRAIL-M (коммерческое использование допускается с use-restrictions; карточка пишет «intended for research») | да, с оговорками | Быстро на CPU/MPS, но качество рук/купюр слабое |
| **SDXL** | CreativeML Open RAIL++-M | да, с оговорками | — |
| **SD 3.5** | Stability AI Community License: бесплатно только при выручке **< $1M/год** (вы + аффилиаты), иначе Enterprise | **нет для банка** | [license](https://stability.ai/community-license-agreement). Для ForteBank порог заведомо превышен |
| **Wan2.2-TI2V-5B** (text/image-to-video, 720p 24fps 5 с) | Apache-2.0 | да | [HF](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). Требует ≥24 ГБ VRAM, ~9 мин/клип на 4090 — на наших машинах нереально |
| **Kubric** | Apache-2.0 | да | [GitHub](https://github.com/google-research/kubric). Простые объекты (GSO — CC BY 4.0; ShapeNet — NC), нет людей и интерьеров |
| **BlenderProc** | **GPL-3.0** | как офлайн-инструмент генерации — да (GPL не распространяется на отрендеренные данные); в репозиторий продукта не включать | [LICENSE](https://github.com/DLR-RM/BlenderProc/blob/main/LICENSE) |
| **Blender** | GPL (вывод — свободен) | да как инструмент | — |
| **Infinigen / Infinigen Indoors** | BSD-3-Clause | да | [GitHub](https://github.com/princeton-vl/infinigen). Процедурные интерьеры, **без людей**; генерация сцены — часы CPU |
| **Unity Perception** | Apache-2.0 (пакет; проект **discontinued**) | да, но Unity Editor: Personal только при финансах **< $200k/год** | [GitHub](https://github.com/Unity-Technologies/com.unity.perception). **Unity SyntheticHumans — «academic use» лицензия → NC** |
| **Unreal Engine** | Unreal EULA (free для внутренних инструментов/генерации данных, роялти только при продаже продукта с движком) | да как инструмент | Порог входа высокий, 2 дня — нет |
| **SMPL-X** (параметрическое тело) | NC («any use for commercial ... or surveillance purposes is prohibited»); коммерческая — через Meshcapade | **нет** | [license](https://smpl-x.is.tue.mpg.de/modellicense.html). Явный запрет surveillance — прямо наш кейс |
| **BEDLAM** (синтетические люди) | NC, явный запрет «surveillance» | **нет** | [license](https://bedlam.is.tue.mpg.de/license.html) |
| **MakeHuman** | код AGPL, экспортируемые модели CC0 (общеизвестно; страница лицензии не отдала текст) | да как инструмент | Качество аватаров низкое |
| **Mixamo** (персонажи/анимации) | Adobe: royalty-free в проектах (страница JS, не подтверждено в сессии) | вероятно да | Нет анимаций «считает деньги / открывает сейф» — нужна ручная анимация |

### 3.2. Варианты, ранжированные по осуществимости за 2 дня

| # | Вариант | Усилия | Ожидаемый перенос на наши видео | Вердикт |
|---|---|---|---|---|
| 1 | **Cut-and-paste композит** реальных вырезок (купюры тенге из KZ-BD по YOLO-боксам + свои фото ключа, сумки инкассатора, удостоверения, телефона, стакана, снятые сверху) на **собственные кадры стола** из 40 видео (фон с/без объекта, случайный поворот/масштаб/перспектива, blur краёв, цветовая подгонка; плюс сегментация купюр SAM/rembg) | 0.5–1 день (скрипт на numpy/OpenCV/albumentations; 5–20k композитов) | **Высокий** для классификатора кропов «на столе есть деньги / нет», «ключ в замке / нет» и для доучивания детектора на классы `cash`, `key`, `bag`, `id_card`: фон и ракурс — наши настоящие. Риск: «вырезанность» → добавлять 30–50% реальных кропов из своих видео и hard-negatives (бумаги, чеки, визитки) | **Делать** |
| 2 | **Диффузионные кропы** FLUX.2-klein-4B / FLUX.1-schnell / Z-Image-Turbo (Apache-2.0) через mflux на Mac: промпты «top-down view of a bundle of banknotes on a wooden office desk, CCTV camera, 480p», «key inserted in the lock of a steel safe, overhead», «hand holding ID card over counter, security camera»; image-edit режим klein — «add a stack of cash to the desk» на реальных кадрах | 0.5 дня настройка + ~30–90 с/картинка на M3 Pro 18 ГБ (8-бит) → 300–800 картинок/ночь | **Средний**: разнообразие фона/освещения, но тенге нарисует неверно (цвет/дизайн ≈ «доллары/евро»), руки и мелкие объекты — артефакты; при тесте на наших кадрах классификаторы, обученные только на диффузии, обычно теряют 10–30 п.п. без смешивания с реальными. Image-edit на наших кадрах (klein multi-ref: референс купюры + кадр стола) — лучше, чем чистый T2I | **Как добавка** к п.1 (не более 30–50% обучающего набора), только для «общей» формы деньги/ключ/сумка |
| 3 | **Собственная досъёмка** (не синтетика, но главный конкурент) — 2–3 ракурса, 2–3 человека, по 3–5 дублей на каждый пропущенный тип: возврат наличных клиенту, приём без удостоверения, деньги предыдущего клиента остались, высыпание инкассаторской сумки без пересчёта, ключ в замке, брак в пачке | 0.5 дня съёмки + 0.5 дня разметки таймлайнов | **Максимальный** — та же комната/камера/объекты, что у жюри (новый ракурс покрывается сменой положения камеры) | **Приоритет №1** |
| 4 | **MEVA** как претрен детектора «рука-рука/передача объекта» (`person_transfers_object`, `hand_interacts_with_person`) | 1 день (скачать подмножество ~50 ГБ, вырезать клипы по аннотациям, обучить бинарный временной классификатор на позах) | **Низкий–средний**: дальний план, outdoor/коридоры, объекты крупные | Только при избытке времени |
| 5 | 3D-рендер (Blender + BlenderProc/Infinigen Indoors + аватары) | 3–5 дней минимум; аватары с пермиссивной лицензией — слабые (MakeHuman/Mixamo), анимации «считает деньги» делать руками | Низкий без доменной адаптации | **Не делать** за 2 дня |
| 6 | Unity Perception / Unreal | 1–2 недели | — | **Не делать** |
| 7 | Text-to-video (Wan2.2) | нет железа | — | **Не делать** |

### 3.3. Практические заметки к п.1–2

- Купюры тенге из KZ-BD: 14 классов включают монеты и мелкие номиналы — для «пачки на столе» использовать только
  купюры 500–20 000; из Kazakh Banknote Image Dataset брать `Mix`/`5000_new` (несколько купюр в кадре) как шаблон
  «веера/стопки».
- Перспектива: наши камеры смотрят под 45–70° вниз → применять гомографию к вырезке (random tilt), а не только аффинное.
- Hard negatives критичны (false positives «unacceptable» → минимальны): бумаги, чеки, визитки, папки, блокноты —
  вырезать из своих же кадров (LVIS/Open Images `envelope`, `receipt`, `notepad`, `clipboard` можно скачать как
  дополнительные негативы, лицензия OK\*).
- Проверка переноса: обучить классификатор на композитах, валидировать **только** на реальных кропах из 40 видео
  (leave-one-cabinet-out, как в текущем протоколе). Если на реальных < 0.85 balanced acc — синтетику сокращать, не
  увеличивать.
- Лицензионная гигиена: все внешние изображения — только CC BY 4.0 (Mendeley) с указанием авторов в `docs/normative_
  sources.md`/`NOTICE`; веса диффузии и инструменты — Apache-2.0; BlenderProc (GPL) и Unity — не нужны.

---

## 4. Итоговая рекомендация

1. **Основа — собственная досъёмка** пропущенных типов нарушений (список в CONTEXT.md) в той же комнате с 2–3 новыми
   положениями камеры. Ничто внешнее не даёт overhead-сцены кассы.
2. **Внешнее стоит брать только**:
   - **KZ-BD + Kazakh Banknote Image Dataset (CC BY 4.0)** → вырезки купюр тенге для cut-and-paste и обучения
     кроп-классификатора «деньги на столе / в руках» (адресует пропуски: возврат наличных клиенту, деньги предыдущего
     клиента на столе, слияние брака в пачку);
   - **LVIS-предобученные веса / open-vocab детекторы** (аннотации CC BY 4.0) для классов `money`, `key`, `handbag`,
     `identity_card`, `cellular_telephone`, `cup` как сигнал-кандидат (не как детектор нарушения);
   - **MEVA (CC BY 4.0)** — опционально, как претрен/негативы для «передача предмета между людьми», если останется время.
3. **Синтетика**: cut-and-paste на реальных кадрах (0.5–1 день, высокий перенос) + при желании 300–800 диффузионных
   кропов FLUX.2-klein-4B / FLUX.1-schnell / Z-Image-Turbo (Apache-2.0) как добавка ≤ 30–50%. 3D-пайплайны и аватары
   не трогать: BEDLAM/SMPL-X — NC с прямым запретом surveillance, Unity SyntheticHumans — academic, трудоёмкость 3–5+ дней.
4. **Не использовать**: Objects365 и всё на нём предобученное (academic-only), ADE20K, SA-1B данные, ImageNet данные,
   UCF-Crime/ShanghaiTech/Avenue (нет лицензии), HOH (NC), SD 3.5 (порог $1M), FLUX.1-dev/Kontext-dev/FLUX.2-dev (NC/коммерч.).

## 5. Что не удалось проверить (для ручной доработки)

- Тексты лицензий AVA, Kinetics (общеизвестно CC BY 4.0, но страницы не отдали текст), HICO-DET (403), COCO ToU
  (страница без текста), Mixamo FAQ (JS), MakeHuman (сайт недоступен), Kaggle Simuletic shoplifting (JS).
- Поиск по Roboflow Universe на «tenge», «safe open/closed», «key in lock», «cash on table» — поиск не выполнен
  (исчерпан лимит WebSearch; прямые страницы Universe отдают 403). Рекомендуется ручная проверка в браузере с чтением
  поля License на каждой странице (по ToS Roboflow публичный проект без явной лицензии = CC BY 4.0).
- Список 13 204 категорий V3Det на наличие `safe`/`banknote` не просматривался.
