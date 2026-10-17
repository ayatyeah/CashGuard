# ⚠️ Контракт API: расхождения с ТЗ (раздел 10) — закрыть до сдачи

Жюри гоняет тестовые видео **автоматически** по API из ТЗ. Если пути/поля другие — все решения
получат `0` по критериям 1–3 (55 баллов из 100), независимо от качества модели.

| Что | ТЗ §10 (жюри) | Текущий Go-бэкенд (README) | Действие |
|---|---|---|---|
| Загрузка | `POST /api/v1/analyze`, multipart, поле **`file`** | `POST /api/jobs`, поле `video` | добавить маршрут `/api/v1/analyze`, принимать поле `file` (и `video` для UI) |
| Статус | `GET /api/v1/jobs/{job_id}` | `GET /api/jobs/:id` | добавить алиас `/api/v1/jobs/:id` |
| Ответ на загрузку | `{"job_id": "...", "status": "processing"}` | 202 с `job_id` | статус в ответе — `processing` |
| Статусы | только `processing` / `done` / `failed` | `queued`, `processing`, `done`, `failed` | наружу `queued` отдавать как `processing` |
| Ключ | заголовок `X-API-Key`, без ключа — отказ | ✅ так и есть | — |
| Формат | `.mov`, ~29 МБ | MP4/MOV/WebM до 500 МБ | ✅ ок |
| Результат | `verdict`, `violations[]` с `type`, `timecode_sec`, `description`, `evidence_frames[]{timestamp_sec, image_base64}` | ✅ совпадает (+ доп. поля `filename`, `created_at`, `demo`, `rule`) | доп. поля допустимы; `image_base64` без префикса `data:` ✅ |
| Таймаут | нет результата за **5 минут** от загрузки → видео не засчитано | worker: 15 минут, 1 воркер, очередь до 20 | анализатор сам считает бюджет от mtime файла (`scripts/cashguard_analyzer.py`); бэкенд должен сохранять загрузку на диск сразу и не трогать mtime |
| `DEMO_MODE` | — | по умолчанию `true`: возвращает **выдуманное** событие | на проверке жюри строго `DEMO_MODE=false`, `ANALYZER_COMMAND=<repo>/bin/cashguard-analyzer` |

Проверка одной командой (как у жюри):
```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video.mov"
curl https://<host>/api/v1/jobs/<job_id> -H "X-API-Key: $KEY"
```
Ожидаемо: сначала `{"job_id":..., "status":"processing"}`, затем `status: done` с `verdict` и `violations` (пустой список при `no_violation`).
