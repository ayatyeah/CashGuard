# 05. Как это делают в индустрии: видеоаналитика кассы/кассира и что из этого взять нам

Агент #5 из 10. Трек: коммерческие и open-source системы контроля кассира, патенты, статьи, хакатоны.
Цель: каталог признаков, которых у нас **нет**, ранжированный по применимости к нашим пропускам;
3–5 идей на уровне псевдокода; реалистичные заявления о точности; ссылки.

Наши пропуски (ml_model.md, 6 типов в 7 видео):
**P1** деньги отданы клиенту (X-1) · **P2** приём денег без документа клиента (X-2) ·
**P3** деньги предыдущего клиента на столе при следующем (X-4, X-38) · **P4** инкассаторская сумка высыпана в сейф
без пересчёта (X-13) · **P5** ключ оставлен в замке сейфа (X-22) · **P6** отбракованные купюры вложены в пачку (X-44).

Оговорка про источники: веб-поиск за эту сессию исчерпал лимит (≈45 запросов + ≈15 прочитанных страниц),
поэтому часть пунктов (Hikvision «absence detection», Kaggle, лицензия RepNet) осталась непроверенной — помечено «⚠ проверить».

---

## 1. TL;DR

1. **Вся индустрия контроля кассира (StopLift, Everseen, Fujitsu, TRASSIR ActivePOS, Macroscop, Ivideon, March Networks
   Searchlight) строится на синхронизации видео с журналом транзакций (POS/TLOG).** Чистое видео без POS — редкость;
   у нас POS нет, значит мы вынуждены воспроизводить «виртуальный журнал транзакций» из визуальных событий (визит клиента,
   руки в зонах, предмет на столе) и искать несогласованность между ними — ровно то, что делает IBM в своих патентах.
2. **Базовый визуальный примитив индустрии — «рука вошла/вышла из зоны» (in/out hand motion через ROI)**, а не детекция
   самого объекта (купюры 12×15 px на потолочной камере не видны — это подтверждает и open-source опыт). IBM/Trinh:
   иерархический конечный автомат на линиях-«растяжках» у границ ROI, цветовая модель кожи, адаптивно обучаемая по
   области клавиатуры; F1 примитивов ≈0.89–0.90, фрод: recall 62 % при FPR 2.5 % на реальных магазинах.
3. **Состояние зоны «до/после» (object-presence в тугом ROI на фоне) — второй примитив**: открытый денежный ящик,
   «объект оставлен», пустая корзина. Правило из патента US7516888: *«ящик открыт дольше, чем присутствует клиент» →
   подозрительно*. Это один-в-один наш P3 (деньги остаются на столе дольше, чем длится визит клиента).
4. **Ложные срабатывания гасят не моделью, а структурой**: (a) второй каскад — 12 эвристик + SVM по временны́м
   отношениям события к соседним (US20130250115); (b) debounce/cooldown/«N кадров подряд»; (c) требование
   полной структуры эпизода (pickup→scan→drop); (d) порог по частоте («≥10 % транзакций нерегулярны»);
   (e) **человек-аналитик в контуре** (StopLift, Veesion, Everseen) — у нас его роль играют evidence-кадры для жюри.
5. **Разные ракурсы**: IBM автоматически калибрует ROI из статистики движения и временно́го порядка относительно
   «якоря» (сканер) — зоны задаются относительно друг друга, а не в пикселях; расстояния между линиями
   пропорциональны разрешению; open-source автор нормирует расстояние «кисть–бедро» на ширину плеч. Наш подход
   «всё в долях роста кассира» — в русле практики; добавить стоит **автокалибровку зон по тепловой карте движения**.
6. **Реалистичная точность**: вендоры публикуют только «85–90 % алертов релевантны» (Veesion), «>80 % non-scan
   закрывается без персонала» (Everseen), «97–99 %» без методики; IBM в рецензируемой статье — 62 %/2.5 % FPR;
   ни один вендор не публикует recall по типам. Наши 0.725 acc / 0 FP в LOCO-CV по кабинетам — честнее большинства
   вендорских цифр; заявлять надо recall по типам + FP на час видео + протокол LOCO, и не обещать 95 %+.

---

## 2. Каталог: кто что детектирует и на каких признаках

| Система / источник | События | Визуальные признаки | POS | Ложные срабатывания | Ракурс | Заявленная точность и как оценивали |
|---|---|---|---|---|---|---|
| **StopLift ScanItAll** (NCR Voyix) [1][2][3] | sweethearting: непробитый товар, закрытый штрихкод, стопка товаров, товар мимо сканера, товар в тележке после оплаты | надпотолочная камера, движение предметов через сканер vs POS-журнал; «MilliSync» синхронизация мс | да, обязателен | первый проход — локальный сервер, затем **офсайт-аналитики** отсеивают ложные | камера строго над сканером; работает на низком разрешении аналог | только кейсы ($10k→$1k потерь/мес Piggly Wiggly); IPVM: метрик нет, рекомендуют пилот |
| **Everseen Evercheck** [4][5][6] | non-scan на кассах самообслуживания и обычных | сравнение корзины с «пустой» моделью, присутствие предметов в зонах транзакции/упаковки, подсчёт предметов vs сканер | да | **каскад верификации**: отложенный промпт покупателю с вариантами ответов, повторный прогон детектора, второй уровень — сотрудник; уровни тревоги yellow/red | — | «>80 % событий закрыто без персонала», «−30 % вызовов персонала»; сотрудники Walmart жалуются на FP (анекдот) |
| **Fujitsu Cashier Fraud Monitoring / HMA for Trust** [7][8] | scan avoidance, banana trick, label switching, забытый товар в корзине | **детектор «предмет в руке»** (SOTA на HICO-DET), счёт предметов vs POS, пустая корзина в начале чека | да | — | камера над кассой | методики нет; manual 2024: в поставке только scan avoidance, офлайн |
| **IBM Research** (Fan, Trinh et al., патенты US8761517, US8345101, US9299229, US20130250115) [9]–[14] | fake scan (pickup→scan→drop без штрихкода), nonscan | **hand motion map** (frame diff × адаптивная цветовая модель кожи), **иерархический FSM по линиям у границ ROI**, пики/впадины энергии движения в ROI, STIP+HoG/HoF, MIL-SVM по перекрывающимся ROI | да; венгерский алгоритм сопоставления визуальных сканов с TLOG | 2-я стадия: 12 эвристик (расстояние до key-in, до соседних fakescan, позиция в транзакции, число примитивов внутри) + RBF-SVM + ранжирование по уверенности | **автокалибровка ROI** по тепловой карте движения и временно́му порядку относительно сканера; пересчёт по расписанию; алерт «проверьте камеру» при несогласованной карте | примитивы F1 0.88–0.90; фрод recall 62 % @ FPR 2.5 % (vs 46 %/3.8 % у предыдущего метода), датасет — 2 магазина, 1 день; 20 потоков 15 fps 360×244 на Quad Core 2.4 GHz |
| **Avigilon / ObjectVideo патент US20080018738** [15] | рука в денежный ящик, N захватов из зоны, void/refund без клиента, override без менеджера, basket run, tailgating | зоны (enter/exit/loiter/dwell/count), tripwire, вставка/удаление статичного объекта, blob-признаки (skin-tone mask, motion histogram), комбинаторы AND/OR/**MATCH в окне времени** | да (для части правил) | правило = комбинатор событий + окно | ROI на камеру | нет |
| **Патент US7516888 (аудит транзакций по видео)** [16] | **ящик открыт после ухода клиента**, no-sale без клиента, счёт предметов | object-presence в тугом ROI по разности с фоном (адаптивный модальный фон ~10 мин), доля кадров с присутствием > порога; клиент отличается от кассира по цвету униформы | опционально | порог по доле кадров, морфология, адаптивный фон, одновременная проверка двух условий | ROI рисует оператор; 2D/3D-зоны | нет |
| **Avigilon US9218580 (behavioural analytics)** [17] | нерегулярное поведение: рукопожатие/передача между кассиром и клиентом, движение вне траекторных границ, сканирование вне частотного диапазона, уход с места во время транзакции | траектории тела и рук, частота pickup/scan, «**handing of items**», положение относительно элементов POS | да | уверенность + взвешенное среднее; порог «нерегулярно в ≥10 % транзакций»; затем ревью | ROI + контекст сцены | нет |
| **Everseen US11482082** [6] | non-scan на SCO | см. выше | да | 10-секундный countdown без ответа → стоп терминала | — | нет |
| **TRASSIR ActivePOS** [18] | 100+ типов: аннулирование, фиктивный возврат, своя скидочная карта, **неправомерное открытие денежного ящика**, операции с крупными купюрами, алкоголь без документа | события ККТ и весов синхронизированы с видео; **чисто видеодетекторов нет** | да | оператор | — | «−70 % мошенничества, −50 % потерь, расследование 10 мин» — маркетинг |
| **Macroscop «Контроль кассовых операций»** [19] | ошибки персонала и правонарушения на кассе | наложение чека на видео, сопоставление учётной системы и записи | да | оператор | — | нет; заявлено применение в **банках** |
| **Ivideon Cash Control** [20] | удаление позиций, продажа без чека + отмена, скидки, фиктивный возврат, аномалии выручки | видео + онлайн-касса (Эвотор) | да | не описано | — | нет |
| **Veesion** [21][22] | жесты сокрытия товара покупателем; **касса — «в разработке»** | только жесты тела, без биометрии; клип ≤30 с | нет | **охранник смотрит клип** перед вмешательством; FAQ с перечнем того, что не детектируется | не указано | «>85 % алертов релевантны» (CEO), цель 99 %; критики: 10+ алертов/день, 90 % FP (без источника) |
| **Scylla** [23] | сокрытие товара в сумку; **зона кассы исключается** (покупатель упаковывает после оплаты) | зоны; клип на телефон | нет | — | — | «до 97 %» без методики |
| **Agrex.ai (банки)** [24] | **неучтённая наличность на стойке, работа с наличными без dual-control, доступ к хранилищу, SOP от открытия хранилища до передачи инкассации**, кассир отсутствует на месте, клиент без внимания, лоитеринг у ATM >5 мин | зоны, dwell-time, «поведенческие модели»; про руки — ничего | — | — | edge/cloud | «99 % несанкционированных доступов», «0 ложных на нормальном трафике» — маркетинг без методики |
| **March Networks Searchlight** [25] | расследование по транзакциям ATM/кассира | видео, синхронизированное с журналом ATM и теллера, поиск по сумме/счёту | да | человек | — | нет |
| **Open-source sweethearting-detection (MIT)** [26] | рука кассира в карман после операции | YOLOv8-Pose, **отношение «кисть–бедро / ширина плеч»** (инвариант к росту и дистанции), кастомный детектор кисти для проверки «кисть исчезла в кармане», ByteTrack; AlarmManager с **debounce + cooldown** | нет | debounce/cooldown; одна тревога на эпизод | **провал на 90° сверху**: «потянулся вперёд» и «опустил к карману» различаются на 8 px; требует 40–45°, 2.4–2.8 м, 300+ люкс | «95 %+, почти 0 FP» без тест-сета; автор сам разоблачил свои 0.995 mAP как утечку по соседним кадрам. Провал классификатора «деньги есть/нет»: купюры 12×15 px, кулак закрывает |
| **door-open-closed-alert, YOLO-TASK** [27][28] | дверь открыта > 20 с | кастомный YOLO open/closed; подтверждение состояния **несколькими последовательными детекциями**; дедуп near-duplicate кадров перед сплитом | — | N подряд + таймер | — | — |
| **AI City Challenge Track 4 (DACNet, Adaptive RoI, VISTA)** [29][30][31] | счёт товаров в руках у кассы | **маскирование рук** цветом фона (EgoHOS), адаптивный ROI от YOLO-person + MediaPipe hands, фильтрация кадров по «цветности» | — | — | синтетика→реальное видео: разрыв закрывают сегментацией рук | F1 0.83 (DACNet, top-3), 0.66 (Adaptive RoI, 6-е) |

Хакатоны: по Decentrathon/Astana Hub/HackNU/Forte публичных write-up'ов про кассу и CCTV-комплаенс **не найдено**
(треки 4.0–5.0 — фото авто для inDrive, геотреки) [32][33]; казахстанские стартапы SENTRA («статистика нарушений персонала у
кассы»), Protector AI, Kepler Vision — без технических деталей [34]. Российские write-up'ы по кассе — только вендорские
обзоры (Crystals, Малахит, МАЙПЛ) с тем же тезисом: «без чека по видео нарушения кассира не видны» [35].
Академических работ именно по **теллеру банка** не нашлось; ближайшие — ATM (лицо закрыто/шлем) и surveillance-events
TRECVid SED (ObjectPut, PeopleMeet) [36]. Датасеты: кассовые CCTV не публикуются; есть купюры на Roboflow (CC BY 4.0,
крупные планы) [37], MIDV-2020 (документы, источники Wikimedia — лицензия датасета не ясна) [38], H2O/HOH — передача
предметов из рук в руки, но лабораторные RGB-D [39].

---

## 3. Каталог признаков, которых у нас нет — ранжирование по применимости к пропускам

Оценка: ● прямо решает, ◐ сильный вспомогательный сигнал, ○ слабый. Стоимость — для CPU/M3, в нашем стеке.

| # | Признак / приём (откуда) | P1 отдача денег | P2 без документа | P3 деньги на столе | P4 сумка без пересчёта | P5 ключ в замке | P6 отбраковка | Стоимость | Что нужно |
|---|---|---|---|---|---|---|---|---|---|
| F1 | **In/out FSM руки по зонам** (IBM/Trinh): последовательность «кисть вошла в зону Z_i → вышла» как токены; эпизод = регулярное выражение по токенам | ● | ● | ◐ | ● | ○ | ◐ | низкая | у нас уже есть RTMPose-кисти; добавить зоны «клиентская половина стола», «ящик/счётная машинка», «сейф», «сумка», автоматически от места кассира |
| F2 | **Object-presence «состояние зоны vs присутствие человека»** (US7516888): предмет/изменение в ROI сохраняется дольше, чем длится визит клиента | ○ | ○ | ● | ○ | ● | ○ | низкая | уже делаем для двери/сейфа; обобщить на **стол между визитами** и на **дверцу сейфа после отхода** |
| F3 | **Детектор «предмет в руке / контакт кисти с портативным объектом»** (Fujitsu HICO-DET; 100DOH `hand_object_detector`, MIT, Faster R-CNN R101) — состояние кисти N/S/O/P/F + bbox объекта | ● | ◐ | ○ | ◐ | ○ | ◐ | средняя (≈1–2 с/кадр CPU, только в ключевые моменты) | ⚠ веса обучены на YouTube-кадрах 100DOH — лицензия данных не указана, код MIT; альтернатива — RTMW wholebody (Apache-2.0, 133 точки с пальцами, 13.5 мс CPU) + «раскрытость кисти» как прокси захвата |
| F4 | **Периодичность движения = «пересчёт»** (повторяющиеся движения: Fan CVPR'09 «repetitive sequential activity», RepNet/TransRAC) — автокорреляция энергии движения кисти в зоне стола | ○ | ◐ | ○ | ● | ○ | ◐ | низкая (без нейросети: автокорреляция сигналов, которые уже есть) | ничего нового; TransRAC (Apache-2.0 по листингу, ⚠ бейдж 996.ICU, VideoSwin — тяжёлый) не нужен |
| F5 | **Комбинатор MATCH в окне времени** (ObjectVideo): событие A, за ним B в ≤T с, иначе нарушение («swipe → вход ≤20 с») | ● | ● | ◐ | ● | ◐ | ◐ | низкая | формализовать правила как `A then B within T` над нашими кандидатами |
| F6 | **Второй каскад анти-FP по временны́м отношениям** (IBM US20130250115): расстояние события до начала/конца визита, до соседних событий, число примитивов внутри, регулярность | — | — | — | — | — | — | низкая | подходит ко всем типам; у нас 1–3 позитива на тип → не SVM, а 2–3 ручных правила |
| F7 | **Автокалибровка ROI по тепловой карте движения и порядку** (IBM US8345101): зона «сканер» = где движение совпадает с событиями; «подача» — до, «выдача» — после; кассир — компактное движение, проход — рассеянное | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | низкая | строим heat-map кистей кассира за видео → кластеры = зоны (ящик/машинка, клиентская кромка, монитор/клавиатура) без ручной разметки под ракурс |
| F8 | **Детектор кисти для подтверждения «кисть исчезла» / «кисть что-то держит»** (sweethearting repo) | ◐ | ○ | ○ | ◐ | ○ | ○ | средняя | у нас уверенность точек RTMPose по запястью — дешёвая замена |
| F9 | **Zero-shot open-vocabulary детектор** (OWLv2 / OWL-ViT / Grounding DINO — все Apache-2.0) для «key», «banknote», «ID card», «money bundle» в кропе зоны в ключевые моменты | ◐ | ◐ | ◐ | ○ | ● | ◐ | высокая (≈1–3 с/кадр CPU на 960²; только 5–20 кадров на видео) | ⚠ проверить работу на мелких объектах при 464×848; использовать как **подтверждение** кандидата, не как детектор |
| F10 | **Маскирование рук цветом фона перед анализом стола** (DACNet) | ○ | ○ | ● | ○ | ◐ | ◐ | низкая | убирает ложные «предметы на столе» от рук кассира при разностях «до/после» |
| F11 | **Дедупликация соседних кадров при обучении/оценке** (YOLO-TASK) и honest-eval: автор sweethearting-репо получил 0.995 mAP из-за утечки по соседним кадрам | — | — | — | — | — | — | — | мы уже делим по видео/кабинетам — закрепить в описании методики |
| F12 | **Debounce + cooldown + «N детекций подряд»** (door repo, sweethearting repo) | — | — | — | — | — | — | низкая | частично есть (пороги по времени); оформить единообразно |
| F13 | **«Отключение детектора в зоне, где поведение легитимно»** (Scylla исключает зону кассы для детектора сокрытия) | ◐ | ○ | ○ | ○ | ○ | ○ | низкая | P1: «деньги через стол» нормально для *выдачи* — ограничить правило операциями *приёма* (клиент дал → кассир вернул) |
| F14 | **Траекторные границы / частотный диапазон нормальной работы** (Avigilon US9218580): нормальная транзакция имеет типовую траекторию рук и темп | ○ | ◐ | ○ | ◐ | ○ | ◐ | средняя | требует больше данных нормы, у нас 19 нормальных видео — можно как soft-признак |

**Итог по применимости:** F1 + F5 (FSM руки по зонам + MATCH-правила) покрывают P1, P2, P4 и дают сигнал для P6; F2 (+F10) закрывают
P3 и P5; F4 — специфичный признак «пересчёт был/не был» для P4 (и косвенно P2/P6); F3/F9 — дорогие подтверждающие проверки
в ключевые моменты. F6/F7/F12 — инфраструктура против FP и под новый ракурс.

---

## 4. Пять конкретных идей (псевдокод)

Общие обозначения: `wristL/wristR(t)` — запястья кассира (RTMPose, в долях роста H относительно места кассира),
`zones` — словарь полигонов в тех же координатах, `visits = [(t_in, t_out, seat_id)]` — визиты клиентов (уже есть),
`frame(t)` — кадр 2 fps, `conf_wrist(t)` — уверенность точки.

### Идея A. Зоны вокруг места кассира + FSM «рука вошла/вышла» → токены → правила (P1, P2, P4, P6)

Берём из IBM/Trinh принцип «каждый примитив — in/out руки относительно ROI», но вместо линий-растяжек — зоны,
выученные автоматически из тепловой карты кистей (F7), чтобы не зависеть от ракурса.

```
# 1. Автокалибровка зон (один раз на видео)
heat = zeros(grid)                       # сетка в координатах "доли роста кассира" относительно её места
for t in frames_where_cashier_seated:
    for w in (wristL, wristR): heat[cell(w(t))] += conf_wrist(t)
clusters = meanshift(heat, bandwidth=0.08*H)      # 3–6 кластеров
# семантика кластеров по ГЕОМЕТРИИ, а не по пикселям:
Z_client  = кластер, ближайший к центроиду сидящего посетителя (усреднён по visits)   # "кромка стола к клиенту"
Z_work    = кластер с max суммарной энергией, ближе к кассиру                         # "перед собой": счётная машинка/клавиатура
Z_drawer  = кластер ниже уровня стола (y_wrist > y_hip кассира) и сбоку                # ящик/тумба
Z_safe    = область, где кассир стоит неподвижно вдали от места (уже есть в safe-детекторе)
Z_bag     = bbox объекта (backpack/handbag/suitcase по YOLOX) в момент его появления в комнате

# 2. Токенизация движения кисти (FSM с гистерезисом: вход = 2 кадра внутри, выход = 2 кадра снаружи)
tokens = []  # (t, hand, 'IN'/'OUT', zone)
for each hand: state = None
    for t: z = zone_of(wrist(t)) if conf_wrist(t) > 0.5 else state
        if z != state and stable_for(2 frames): tokens += (t, hand, 'OUT', state), (t, hand, 'IN', z); state = z

# 3. Эпизоды как регулярные выражения над токенами внутри визита
EP_RECEIVE = IN(Z_client) → OUT(Z_client) → IN(Z_work) ... [≥1 c]        # взяла у клиента, перенесла к себе
EP_RETURN  = IN(Z_work|Z_drawer) → OUT → IN(Z_client) → OUT(Z_client)       # перенесла что-то клиенту
EP_DOC     = IN(Z_client) → IN(Z_work) с длительностью в Z_work < 1.5 с и БЕЗ последующего EP_COUNT  # взяла тонкий предмет (документ) и положила
EP_COUNT   = периодичность в Z_work (идея C)
EP_DRAWER  = IN(Z_drawer) → OUT(Z_drawer)

# 4. Правила-комбинаторы MATCH (ObjectVideo), окно T в секундах
P1 cash_returned:   EP_RECEIVE ∧ EP_COUNT ∧ затем EP_RETURN в течение визита ∧ ¬EP_DRAWER после EP_RETURN
                    (деньги пересчитали и вернули клиенту, в ящик не убрали)   → evidence: кадр max(y-протяжённость руки к клиенту)
P2 no_id:           EP_RECEIVE ∧ EP_COUNT ∧ EP_DRAWER ∧ ¬(EP_DOC до EP_COUNT) ∧ ¬(кисть у монитора/сканера ≥1 с до EP_DRAWER)
P4 bag_no_count:    bag появился (курьер) ∧ Z_bag→Z_safe перенос (кассир несёт сумку: bbox сумки движется вместе с ней)
                    ∧ ¬EP_COUNT между приходом сумки и IN(Z_safe)   # сравнить с X-16 (корректная сдача: есть EP_COUNT/ID)
P6 rejects_merged:  EP_COUNT ∧ ≥1 «откладывание» (кисть OUT(Z_work) → IN(Z_aside) где Z_aside — новый кластер появившийся во время счёта)
                    ∧ затем IN(Z_aside)→IN(Z_work)/Z_drawer без EP_RETURN   # отложенное вернулось в общую массу
```

Анти-FP (F6): эпизод засчитывается только если (1) кисть видна с conf>0.5 в ≥70 % кадров эпизода, (2) эпизод не в первые/последние
3 с визита, (3) не более одного срабатывания на визит (cooldown), (4) для P1 — требуем *оба* сигнала: EP_RETURN **и** отсутствие
EP_DRAWER после него (согласие двух признаков — уже наш приём).

### Идея B. «Состояние стола/сейфа переживает присутствие человека» (P3, P5) — обобщение US7516888

```
# Фон стола берём в момент "никого нет за столом и кассир руки убрала" (начало видео / между визитами)
desk_roi = polygon(Z_client ∪ Z_work)                              # из идеи A
def desk_state(t):                                                # маска "что лежит на столе"
    img = frame(t); mask_hands = dilate(skeleton_hands_mask(t), 0.05*H)   # F10: маскируем руки/предплечья
    diff = |gray(img) - gray(bg)| > thr_adapt(img)                # порог относит. контраста кадра (как в safe-детекторе)
    return diff & desk_roi & ~mask_hands

for i, (t_in, t_out) in enumerate(visits[:-1]):
    S_end   = median(desk_state(t) for t in [t_out-2, t_out+3])        # что осталось после ухода клиента i
    S_base  = median(desk_state(t) for t in [t_in-3, t_in])             # что было до его прихода
    leftover = S_end & ~S_base                                           # появилось во время визита и не исчезло
    t_next  = visits[i+1].t_in
    S_next  = median(desk_state(t) for t in [t_next, t_next+3])
    if area(leftover & S_next) > 0.004*H²  and  components(leftover) contain compact blob (0.02H..0.15H):
        P3 candidate at t_next, evidence = frame(t_out+2), frame(t_next+1)
        # подтверждение (опц., F9): OWLv2 на кропе blob с запросами ["banknotes","stack of cash","paper document","pen"]:
        # засчитать только если top-1 ∈ {banknotes, stack of cash} и score > s*
```

P5 (ключ в замке): тот же шаблон на `Z_safe_door`: `S_base` — вид дверцы до подхода кассира, `S_after` — после того как кассир
отошла ≥1.5 с и до конца видео; `leftover` маленький (0.01–0.05 H), **локализован у края дверцы/ручки**, сохраняется до конца
видео. Отличие от уже сделанного `safe_left_open`: там измеряется крупное изменение (открытая дверь), здесь — малое стабильное
изменение при закрытой двери; подтверждение — OWLv2 «key», «keychain» на кропе (один кадр, ≈2 с CPU).

### Идея C. Детектор «пересчёт» через периодичность (P4, вспомогательно P2/P6)

Банкноты не видны, но пересчёт — ритмичное движение одной кисти/пальцев (или укладка в счётную машинку) с периодом 0.2–1.5 с.
Это ровно наблюдение IBM про «repetitive and periodic activities». Нам хватит сигнала, который уже есть.

```
sig(t) = |wristR(t) - wristR(t-1)| + |wristL(t) - wristL(t-1)|            # при 2 fps — энергия движения кистей
       + motion_energy(frame(t), frame(t-1), Z_work)                       # frame-diff в рабочей зоне (норм. на площадь)
# при 2 fps период 0.2–0.5 с не разрешится → для окна-кандидата (кисти в Z_work ≥ 4 с) пересэмплировать видео 8–10 fps
for window W (6 с, шаг 2 с) where wrists in Z_work:
    s = detrend(sig at 10 fps); r = autocorr(s)
    peak_lag, peak_val = argmax(r[lag ∈ 0.2..1.5 с])
    periodic[W] = peak_val > 0.4 and spectral_flatness(s) < 0.6
EP_COUNT = объединение ≥2 подряд периодических окон (≥ 6 с)
```

Для P4: `bag_arrived ∧ cashier_at_safe_with_bag ∧ ¬EP_COUNT[t_bag, t_safe]` → нарушение, таймкод = момент IN(Z_safe).
Для X-16 (норма) EP_COUNT/проверка пломбы должны присутствовать — проверить разделимость на этих двух видео прежде чем
включать. Если TransRAC/RepNet захочется всё же — лицензии ⚠ проверить (TransRAC: Apache-2.0 по листингу, бейдж 996.ICU; RepNet в
google-research — обычно Apache-2.0, не подтверждено в этой сессии).

### Идея D. Контакт кисти с объектом в ключевой момент (P1, P2) — подтверждающий каскад

```
key_moments = [t for (t,hand,'IN',Z_client) in tokens]            # момент пересечения кромки стола к клиенту
for t in key_moments (≤ 8 на видео):
    crop = frame(t) around wrist, size 0.35*H
    # вариант 1 (MIT код, ⚠ лицензия данных): hand_object_detector → state ∈ {N,S,O,P,F}, obj_bbox
    # вариант 2 (Apache-2.0): RTMW wholebody → 21 точка кисти → "раскрытость" = mean dist(fingertips, palm)/hand_len;
    #   захват (<0.45) vs пустая/плоская ладонь (>0.7)
    holding[t] = state == 'P' or openness < 0.45
# P1: holding в момент IN(Z_client) после EP_COUNT (несёт что-то клиенту) ∧ obj_bbox тонкий/плоский (купюры), а не крупный (сумка)
# P2: ¬holding ни в одном IN(Z_client)→IN(Z_work) переходе ДО EP_COUNT (ничего тонкого не взяла → документа не было)
```
Это именно то, что Fujitsu называет «object held in hand» и строит на нём весь продукт; у нас — только как
подтверждение кандидата из идеи A (каскад снижает FP, как у Everseen).

### Идея E. Единый анти-FP слой и оценка (F6, F11, F12)

```
for cand in candidates:
    cand.score_structure = all([ cand.has_full_episode,                 # pickup→...→drop аналог
                                 cand.t - visit.t_in > 3 and visit.t_out - cand.t > 2,
                                 cand.n_primitive_repeats <= 2 ])      # "много примитивов внутри" = шум (IBM)
    cand.score_agree = количество независимых признаков (руки/стол/VLM) > = 2
    emit if score_structure and score_agree and cooldown_ok(type, visit)
```
Оценка: leave-one-cabinet-out (уже), **плюс** отчёт «FP на час видео» и recall по типам; никакой случайный сплит кадров.

---

## 5. Как индустрия справляется с разными ракурсами (и что взять)

- **IBM US8345101**: ROI не в пикселях, а относительно «якоря» (сканер = зона, чья активность совпадает с событиями), остальные —
  по порядку во времени и по компактности движения; перекалибровка по расписанию; **alert «проверьте камеру», если карта движения
  не похожа на кассу** — для нас: если автокалибровка не нашла Z_client/Z_work, деградировать в консервативный режим (без правил A), не
  выдавать нарушение.
- **Trinh**: расстояние между линиями FSM пропорционально разрешению; цветовая модель кожи переобучается по области клавиатуры при
  смене освещения/кассира.
- **sweethearting-detection**: нормировка на ширину плеч; прямое признание, что строго надпотолочный ракурс делает различение
  «вперёд/вниз» невозможным (8 px) — при новом ракурсе жюри часть правил по глубине руки стоит отключать по условию
  «камера почти вертикальная» (оценивается по отношению длины торса к ширине плеч в кадре).
- **DACNet**: маскирование рук цветом фона перед анализом объектов — убирает главный источник FP при сравнении состояний стола.
- Все вендоры задают требования к монтажу (StopLift — строго над сканером; sweethearting repo — 40–45°, 2.4–2.8 м, 300+ люкс); у нас
  монтаж не контролируем, поэтому единственный путь — относительная геометрия + автокалибровка + отказ от правил, которые
  в данном ракурсе физически не проверяемы (как уже сделано с дверью).

---

## 6. Какие заявления о точности реалистичны

| Источник | Цифра | Как получена |
|---|---|---|
| IBM Trinh et al. (рецензируемая) | примитивы F1 0.88–0.90; фрод recall **62 %** @ FPR **2.5 %** | 2 магазина, 1 рабочий день, ручная разметка; авторы прямо пишут, что часть «истинных» событий не размечена и засчитана как FP |
| Veesion (вендор) | «>85 % алертов релевантны» | отзывы операторов, без протокола |
| Everseen (вендор/Forrester) | «>80 % non-scan закрыто без персонала» | композитный клиент Forrester TEI |
| Agrex.ai (вендор) | «99 % доступов», «0 FP на нормальном трафике» | кейс без методики |
| Scylla (вендор) | «до 97 %» | нет |
| sweethearting repo (open-source) | «95 %+, почти 0 FP» | без тест-сета; автор сам разоблачил 0.995 mAP как утечку |
| IPVM о StopLift | метрик нет; «проводите пилот» | независимый обзор |
| **Мы (LOCO по кабинетам, 40 видео)** | acc 0.725, FP 0, тип верен 0.80; финальная 0.825/0 FP | честная кросс-валидация по ракурсу |

Выводы для документации/продукта:
1. **Не заявлять** «95–99 %». Реалистичный и защищаемый формат: «recall по типам X/Y при 0 FP на N часов видео,
   leave-one-camera-out», плюс список типов, которые система **не** детектирует (как делает Veesion в FAQ) — жюри это
   воспринимает как зрелость, а не слабость.
2. Индустриальная норма — **человек подтверждает алерт** (StopLift-аналитики, охранник Veesion, сотрудник Everseen).
   Наши 1–3 evidence-кадра + описание — наш «аналитик в контуре»; стоит явно подать это как design-choice в продуктовой части.
3. Признак без POS/видимого объекта (руки, зоны, периодичность, состояние стола) в литературе даёт recall ~0.6 при FPR ~2–4 %
   на *реальных* потоках. На постановочных 40 видео можно получить лучше, но при 1–3 позитивах на тип любой порог — догадка;
   говорить «обнаруживает» только про типы с ≥2 позитивами и проверенным разделением на нормальных видео.

---

## 7. Open-source ресурсы и лицензии (для STACK.md)

| Ресурс | Лицензия | Применимость | Замечание |
|---|---|---|---|
| RTMW / RTMPose-wholebody через rtmlib (133 точки, кисти) | Apache-2.0 | F3/идея D: пальцы → «захват» | RTMPose-m wholebody 13.5 мс CPU (i7-11700, ONNX) [40] |
| ddshan/hand_object_detector (100DOH) | код **MIT**; веса — Faster R-CNN R101, обучены на YouTube-кадрах | детектор «кисть держит объект» | ⚠ лицензия данных 100DOH не заявлена; тяжёлый на CPU [41] |
| OWLv2 / OWL-ViT (Google), Grounding DINO (IDEA) | Apache-2.0 | F9: zero-shot «key/banknote/ID card» | ⚠ мелкие объекты на 464×848 — проверить; YOLO-World — GPL, не брать [42] |
| TransRAC | Apache-2.0 (листинг), бейдж 996.ICU | счёт повторений | ⚠ не нужен: автокорреляции хватает [43] |
| RepNet (google-research) | ⚠ не проверено в сессии (репо google-research обычно Apache-2.0) | — | TF, тяжёлый |
| MuhamedEminKrd/sweethearting-detection | MIT | опыт/паттерны (debounce, нормировка) | модели Ultralytics YOLOv8 — **AGPL**, код копировать нельзя, идеи — можно [26] |
| door-open-closed-alert, YOLO-TASK | не проверено | паттерн «N подряд + таймер» | Ultralytics → AGPL |
| Roboflow: Banknote (Workkk), Banknote detection (BIN, ~1.6k) и др. | CC BY 4.0 | крупные планы купюр — для CLIP/OWL-проб, не для CCTV | тенге нет [37] |
| MIDV-2020 | источники Wikimedia Commons; лицензия датасета не указана | документы | ⚠ [38] |

---

## 8. Ссылки

1. StopLift / Harps — https://progressivegrocer.com/harps-using-stoplift-tech-catch-crafty-cashiers
2. StopLift (Wikipedia, MilliSync, кейс Piggly Wiggly) — https://en.wikipedia.org/wiki/StopLift ; http://dev.stoplift.com/
3. IPVM «StopLift Retail Video Analytics Examined» — https://ipvm.com/reports/1501
4. Everseen Evercheck — https://everseen.com/solutions/evercheck ; блог — https://everseen.com/blog/stop-skip-scanning
5. Forrester TEI of Evercheck — https://tei.forrester.com/go/everseen/evercheck/docs/TEI_of_Evercheck.pdf ; VentureBeat о FP у Walmart — https://venturebeat.com/ai/cashierless-tech-could-detect-shoplifting-but-bias-concerns-abound
6. Патент Everseen US11482082 «Non-scan loss verification at self-checkout terminal» — https://patents.google.com/patent/US11482082B2/en
7. Fujitsu Cashier Fraud Monitoring — https://en-portal.research.global.fujitsu.com/cashier-fraud-monitoring ; manual — https://en-portal.research.global.fujitsu.com/content/CashierFraudMonitoring_User'sManual.pdf
8. Fujitsu HMA for Trust — https://www.fujitsu.com/global/themes/data-driven/data-intelligence-paas/hma-for-trust
9. Trinh, Fan et al., «Detecting human activities in retail surveillance using hierarchical FSM», ICASSP — https://home.ttic.edu/~ntrinh/papers/HFSM_ICASSP.pdf
10. Fan et al., «Detecting sweethearting in retail surveillance videos», ICASSP 2009 — https://ieeexplore.ieee.org/abstract/document/4959867
11. Fan et al., «Recognition of repetitive sequential human activity», CVPR 2009 — https://mlanthology.org/cvpr/2009/fan2009cvpr-recognition
12. IBM US8761517 «Human activity determination from video» — https://patents.google.com/patent/US8761517B2/en
13. IBM US8345101 «Automatically calibrating regions of interest for video surveillance» — https://patents.google.com/patent/US8345101B2/en
14. IBM US20130250115 «False alarm reduction during event detection» — https://patents.google.com/patent/US20130250115A1/en ; US9299229 «Detecting primitive events at checkout» — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9299229
15. ObjectVideo US20080018738 «Video analytics for retail business process monitoring» — https://patents.google.com/patent/US20080018738A1/en
16. US7516888 «Auditing transaction activity ... using visual recognition» — https://patents.google.com/patent/US7516888B1/en
17. Avigilon US9218580 «Detecting retail shrinkage using behavioral analytics» — https://patents.google.com/patent/US9218580B2/en
18. TRASSIR контроль кассовых операций — https://www.dssl.ru/l/solutions_trassir_activepos/ ; кейс в банке — https://www.dssl.ru/publications/cases/videoanalitika-trassir-v-vedushchem-rossiyskom-banke/
19. Macroscop контроль кассовых операций — https://macroscop.com/novosti/kontrol-kassovyh-operatsiy-ot-macroscop-sokraschenie-poter-na-kassah-i-effektivnyy-kontrol-nad-obektom
20. Ivideon Cash Control — https://ru.ivideon.com/cash-registers/
21. Veesion — https://veesion.io/en/sectors/cctv-retail-stores/ ; Newsweek («>85 % relevant») — https://www.newsweek.com/shoplifting-cameras-artifical-intelligence-2089221
22. Критика Veesion — https://www.arcadian.ai/blogs/blogs/shoplifting-detection-2025-why-ai-solutions-fail-and-how-retailers-must-think-differently
23. Scylla — https://www.scylla.ai/suspicious-shopping-behavior-detection/
24. Agrex.ai Banking Video Analytics — https://www.agrexai.com/banking-video-analytics/
25. March Networks Searchlight (банки) — https://go.marchnetworks.com/banking-video-surveillance ; IPVM обсуждение CCTV в банках — https://ipvm.com/discussions/what-is-the-best-cctv-system-for-banks
26. MuhamedEminKrd/sweethearting-detection (MIT) — https://github.com/MuhamedEminKrd/sweethearting-detection
27. Anangsha072/door-open-closed-alert — https://github.com/Anangsha072/door-open-closed-alert
28. tanukusaitejesh-prog/YOLO-TASK (door open/closed, overhead CCTV) — https://github.com/tanukusaitejesh-prog/YOLO-TASK
29. DACNet (CVPRW 2023, маскирование рук) — https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Cai_DACNet_A_Deep_Automated_Checkout_Network_With_Selective_Deblurring_CVPRW_2023_paper.pdf
30. Adaptive RoI (CVPRW 2023) — https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Dhonde_Adaptive_RoI_With_Pretrained_Models_for_Automated_Retail_Checkout_CVPRW_2023_paper.pdf
31. 7th AI City Challenge — https://arxiv.org/pdf/2304.07500 ; VISTA — https://arxiv.org/pdf/2204.11024
32. Decentrathon 4.0 итоги — https://www.gov.kz/memleket/entities/maidd/press/news/details/1086593?lang=ru ; 6.0 — https://decentrathon.ai/en
33. Decentrathon 5.0 кейсы inDrive — https://digitalbusiness.kz/2026-03-26/kak-s-pomoshchyu-ai-nahodit-liderov-i-raspredelyat-resursi-keysi-indrive-dlya-uchastnikov-decentrathon-5-0/
34. SENTRA (Астана) — https://digitalbusiness.kz/2026-08-31/shkolnik-iz-kazahstana-sozdal-ii-platformu-kotoraya-predotvrashchaet-krazhi-v-magazinah/ ; Protector AI — https://kapital.kz/tehnology/151051/ne-prosto-kamery-kak-kazahstanskij-startap-menyaet-rynok-sistem-bezopasnosti.html ; Startup Radar — https://raem.kz/startup-radar/
35. Crystals: контроль кассовой зоны СБ — https://crystals.ru/infocenter/articles/kontrol-i-analitika-kassovoy-zony-magazina-sluzhboy-bezopasnosti-metody-varianty-ispolneniya-plyusy-/ ; hi-tech.mail.ru о FP на СКО — https://hi-tech.mail.ru/articles/151106-kassa-zapodozrila-vas-v-krazhe-kak-ii-na-samoobsluzhivanii-zlit-lyudej/
36. TRECVid SED (PKU, ObjectPut/PeopleMeet) — https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/pku-sed.pdf ; ATM theft CNN — https://idr.nitk.ac.in/handle/123456789/15157
37. Roboflow banknote datasets — https://universe.roboflow.com/workkk-quf76/banknote-br9ax ; https://universe.roboflow.com/bin-mibxd/banknote-detection-2amdt
38. MIDV-500/2020 — https://arxiv.org/abs/1807.05786 ; https://arxiv.org/pdf/2407.20662
39. H2O handover benchmark — https://arxiv.org/pdf/2104.11466 ; HOH — https://arxiv.org/html/2310.00723v5
40. rtmlib (RTMW wholebody) — https://github.com/Tau-J/rtmlib ; RTMPose benchmark — https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/benchmark
41. 100DOH / hand_object_detector — https://github.com/ddshan/hand_object_detector ; статья — https://arxiv.org/pdf/2006.06669
42. Grounding DINO vs OWLv2 (лицензии) — https://roboflow.com/compare/grounding-dino-vs-owlv2
43. TransRAC — https://github.com/SvipRepetitionCounting/TransRAC ; RepNet eval note — https://arxiv.org/pdf/2411.08878
44. Abandoned-object detection (dual background, Porikli MERL) — https://www.merl.com/publications/docs/TR2007-040.pdf ; Eocortex обзор — https://eocortex.com/company/blog/algorithms-for-solving-abandoned-objects-detection-issues
45. US11100331 (skin-tone в зоне сканера против пустых рук) — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11100331
