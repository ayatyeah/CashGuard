# ForteWin — AI-сервис выявления нарушений кассовой дисциплины по видео

Datathon 26, кейс ForteBank. Сервис принимает видео (.mov) работы кассира, локально (без облачных AI)
находит визуально наблюдаемые признаки возможных нарушений и возвращает тип, таймкод, кадры-доказательства,
описание и ссылку на норматив РК. План стека и API — `STACK.md`.

> Статус: в разработке. Готово: анализ данных, таксономия нарушений, справочник нормативов,
> ML-анализатор (`app/analyzers/ml.py`, обученные модели в `models/*.json`). Подробно — `docs/ml_model.md`.

## Быстрый запуск ML-анализатора
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash scripts/download_models.sh                       # веса (Apache-2.0/MIT), один раз
.venv/bin/python scripts/analyze_video.py video.mov    # JSON в формате API жюри
```
В сервисе: `ANALYZER=ml` → `from app.analyzers.ml import MLAnalyzer` (интерфейс STACK.md §4).

## Данные
Датасет **не хранится в репозитории** (условия хакатона: данные только для использования в рамках хакатона).
Распакуйте `open.zip` и разложите по `configs/split.json`:
`data/train/videos/` (30 видео, 32 файла с дубликатами) и `data/test/videos/` (10 видео — **только для финальной оценки**,
не использовать для обучения и подбора порогов).

## Что есть
| Файл | Содержание |
|---|---|
| `configs/split.json` | разбиение train/test (стратифицировано 5+5, дубликаты Х-4/Х-47, Х-5/Х-48 — в train) |
| `configs/train_annotations.json` | наша разметка типов и таймкодов по train-видео |
| `app/rules/norms.yaml` | 13 типов нарушений → пункт норматива РК + дословная цитата (сверено с adilet) |
| `docs/violations_train.md` | сводка типов нарушений и наблюдения о разметке |
| `docs/train_timelines.md` | покадровые таймлайны всех train-видео и признаки «норма vs нарушение» |
| `docs/normative_sources.md` | документы НБРК/АРРФР, сопоставление тип → пункт, пробелы в нормативке |
| `docs/ml_model.md` | архитектура ML, модели и лицензии, обучение, метрики, ограничения |
| `app/ml/` | восприятие, сцена, кандидаты событий, скорер, пайплайн |
| `app/analyzers/ml.py` | `MLAnalyzer` — точка подключения к сервису |
| `models/track_model.json`, `models/scorer.json` | обученные модели (веса детекторов — `scripts/download_models.sh`) |
| `scripts/` | извлечение признаков, обучение (`train_tracks.py`, `train_scorer.py`), оценка, CLI |



# CashGuard

Локальное приложение для проверки видео кассовых операций. React + TypeScript + Tailwind + shadcn/ui; сервер Go + Gin.

## Docker Compose (основной запуск)

Требуется запущенный Docker Desktop с Linux-контейнерами. Все команды выполняются из корня проекта.

1. Если файла .env ещё нет, скопируйте .env.example в .env. Замените POSTGRES_PASSWORD и API_KEY случайными значениями. Для пароля PostgreSQL используйте буквы и цифры (пароль входит в URL подключения).
2. Запустите:

```powershell
docker compose up -d --build
```

Интерфейс: http://127.0.0.1:3000. API: http://127.0.0.1:8081/api. PostgreSQL: 127.0.0.1:55432, база и пользователь cashguard, пароль из .env. Порты изменяются через FRONTEND_PORT, BACKEND_PORT и POSTGRES_PORT.

В Compose три основных сервиса: db (PostgreSQL 17), backend (Go Gin), frontend (nginx со статической React-сборкой). Healthcheck базы и API определяют порядок запуска. База и видео сохраняются в именованных томах postgres_data и uploads. Все опубликованные порты привязаны только к 127.0.0.1.

```powershell
docker compose ps
docker compose logs --tail 100 backend
docker compose stop
docker compose start
```

Обычный docker compose down сохраняет тома. Не используйте down -v, если данные нужны: этот флаг удаляет тома базы и видео.

## Перенос существующих данных из SQLite

SQLite больше не используется сервером. Драйвер оставлен только для переноса старых данных и автономных тестов. Для переноса сначала остановите старый Go-сервер, чтобы в нём не появлялись новые данные. Сохраните резервную копию src/backend/data.

До первого использования новой PostgreSQL-базы выполните:

```powershell
docker compose up -d db
docker compose build backend
docker compose --profile tools run --rm migrate
docker compose up -d backend frontend
```

Команда читает src/backend/data/cashguard.db и исходные видео, копирует пользователей, хеши паролей, сессии, настройки, задания и записи в PostgreSQL/том uploads. Исходная SQLite-база и файлы не меняются. Если целевая база уже содержит пользователей или задания, перенос откажется перезаписывать её. При ошибке транзакция откатывается; новые скопированные файлы удаляются. Старые общие JSON-результаты при обычном запуске импортируются в пространство жюри, если присутствуют в DATA_DIR/jobs.

## Разработка без контейнеризации приложения

Требуются Go 1.25+ и Node.js 20+, PostgreSQL можно оставить в Docker:

```powershell
docker compose up -d db
$env:DATABASE_URL = 'postgres://cashguard:YOUR_PASSWORD@127.0.0.1:55432/cashguard?sslmode=disable'
$env:API_KEY = 'replace-with-your-random-secret-key'
$env:DEMO_MODE = 'true'
go -C src/backend run ./cmd/server
```

Во втором терминале:

```powershell
npm.cmd --prefix src/frontend install
npm.cmd --prefix src/frontend run dev
```

Откройте http://127.0.0.1:5173 и подключите API-ключ из .env (API_KEY). Для использования Vite с API из Compose задайте перед запуском Vite API_PROXY_TARGET=http://127.0.0.1:8081. Все запросы видео используют X-API-Key; история общая. Видео: MP4, MOV или WebM до 500 МБ. После перезагрузки запись доступна через кнопку «Открыть видео».

**Деморежим не анализирует видео.** Он возвращает маркированное синтетическое событие и иллюстрацию для проверки интеграции. Без деморежима и подключённого анализатора задание честно заканчивается ошибкой, а не выдуманным вердиктом.

## Структура

```text
src/frontend/
  src/components/ui/    адаптированный App1 и компоненты из промпта
  src/lib/              типы, API-клиент, утилиты
  src/index.css         стили и адаптивная раскладка
src/backend/
  cmd/server/           сборка зависимостей и запуск
  internal/domain/      сущности, интерфейсы Repository и Analyzer
  internal/usecase/     очередь, обработка, восстановление после перезапуска
  internal/infrastructure/  PostgreSQL, миграция SQLite и локальный анализатор
  internal/transport/   Gin HTTP API, авторизация, проверка загрузки
```

`@/components/ui` настроен в `components.json` и alias TypeScript/Vite: компоненты shadcn используют этот единый путь. Исходный App1 адаптирован к банковскому кейсу; отсутствующие в промпте sidebar/data заменены меню и реальными данными API.

## API

Маршруты видео требуют `X-API-Key`. Регистрация, вход по паролю и Bearer-сессии отключены. `/api/health` и `/api/ready` доступны без ключа.

| Метод | Маршрут | Назначение |
|---|---|---|
| GET | /api/health | Доступность и признак demo |
| GET | /api/ready | Готовность подключения к PostgreSQL |
| POST | /api/jobs | multipart/form-data, поле video; ответ 202 с job_id |
| GET | /api/jobs/:id | Статус и результат |
| GET | /api/jobs | Личная история (новые первыми) |
| GET | /api/jobs/:id/video | Исходное видео, только для владельца |

Статусы: `queued`, `processing`, `done`, `failed`. При ошибке возвращается `error`. Вердикты: `violation`, `no_violation`.

```powershell
curl.exe -H "X-API-Key: YOUR_KEY" -F "video=@video.mp4" http://127.0.0.1:8080/api/jobs
curl.exe -H "X-API-Key: YOUR_KEY" http://127.0.0.1:8080/api/jobs/JOB_ID
```

Результат сохраняет контракт `job_id`, `status`, `verdict`, `violations[].type`, `timecode_sec`, `description`, `evidence_frames[].timestamp_sec`, `image_base64`. Дополнительные поля: `filename`, `created_at`, `demo`; необязательный `violations[].rule = {title, section, url}`. Изображения передаются как чистый base64 PNG/JPEG без data URI.

## Подключение локальной модели

```powershell
$env:DEMO_MODE = 'false'
$env:ANALYZER_COMMAND = 'C:\models\cash-analyzer.exe'
go -C src/backend run ./cmd/server
```

Адаптер запускает исполняемый файл напрямую, без shell, с единственным аргументом — абсолютным путём видео. Анализатор пишет завершённый JSON в stdout (не более 32 МБ), логи — в stderr; код выхода 0. Максимальное время — 15 минут. Он должен возвращать `status: done`, вердикт и список нарушений по контракту пользователя. `job_id` и метаданные назначает Go-сервер. Python-модель можно обернуть в локальный исполняемый launcher. Модель и все веса должны находиться локально; облачные API использовать нельзя.

Модель, веса, обучающий датасет и нормативные правила не предоставлены и в репозиторий не включены. Распознавание реальных нарушений и нормативное обоснование ещё нужно реализовать в анализаторе; интерфейс не выдумывает ссылки и показывает отсутствие источника. Это готовый слой приложения и интеграции, а не обученная AI-модель.

## Хранение и ограничения

Сервер требует DATABASE_URL для PostgreSQL. Схема создаётся транзакционно. Новые задания сохраняются в общем пространстве API-ключа (jury). Старые аккаунты и их данные сохранены для совместимости миграции, но недоступны через API. Ключ хранится в sessionStorage текущей вкладки и проверяется сервером.

DATA_DIR хранит исходные видео, в Compose это отдельный том /data. Очередь обработки пока находится в памяти одного Go-сервера: 1 worker и до 20 ожидающих задач. Не запускайте несколько экземпляров backend без общей очереди. Прерванные задания после перезапуска помечаются failed. Для резервирования нужны и дамп PostgreSQL, и копия тома видео. Исходная SQLite-база сохраняется после миграции как резервная копия.

Светлая/тёмная тема и русский, английский, казахский языки переключаются в шапке. Настройки сохраняются в текущем браузере. Тексты и описания, возвращаемые моделью, показываются на языке оригинала: интерфейс не переводит и не меняет доказательства.

Ограничения: пока нет ролей администратора, автоматического удаления старых записей и пагинации. Для production нужны HTTPS, политика хранения и резервное копирование. Проверка MIME/сигнатуры не заменяет декодирование видео в модели.

`ADDR` по умолчанию `127.0.0.1:8080`; для доступа жюри разверните в разрешённом контуре с HTTPS reverse proxy. Не выставляйте Vite dev server как production. Статическую сборку frontend/dist обслуживает веб-сервер, который проксирует /api в Go.

## Проверки и сборка

```powershell
npm.cmd --prefix src/frontend run build
go -C src/backend test ./...
go -C src/backend vet ./...
go -C src/backend build -o cashguard.exe ./cmd/server
```

Все ресурсы интерфейса поставляются локально, внешних шрифтов или облачных AI-запросов нет. Интернет нужен для первоначальной установки зависимостей; после сборки — не нужен.

## Интеграционные тесты PostgreSQL

```powershell
$env:TEST_DATABASE_URL = 'postgres://cashguard:YOUR_PASSWORD@127.0.0.1:55432/cashguard?sslmode=disable'
go -C src/backend test ./internal/infrastructure -run TestPostgres -v
```

Тесты создают собственную временную схему и удаляют только её. Проверяют запись/чтение, изоляцию владельцев, сессии, настройки, повторное подключение и перенос SQLite с видео. Без TEST_DATABASE_URL эти интеграционные тесты пропускаются.

Образ backend не содержит AI-модель. Для реального анализа расширьте Dockerfile локальной моделью и её зависимостями, смонтируйте веса и задайте ANALYZER_COMMAND внутри контейнера. Включённый по умолчанию деморежим остаётся явно маркированным.


Полный smoke-тест работающего Compose (PowerShell 7): `./smoke-compose.ps1`. Он использует API_KEY из .env, загружает синтетический тестовый файл в деморежиме, проверяет результат и запрет неверного ключа, затем удаляет только своё тестовое задание. Это проверка интеграции, а не качества AI или декодирования настоящего видео.
