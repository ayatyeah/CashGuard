# Деплой на Railway (CPU-режим, без VLM)

Один контейнер (`Dockerfile` в корне): React-интерфейс (статика) + Go API + Python-анализатор (YOLOX, RTMPose, CLIP, OWLv2).
Postgres — плагин Railway, том `/data` — загруженные видео.

Проект: `cashguard` (Railway), сервис `app`, публичный адрес `https://app-production-d11c.up.railway.app`.

## Переменные сервиса `app`
`API_KEY` (ключ для жюри), `DATABASE_URL=${{Postgres.DATABASE_URL}}`, `DEMO_MODE=false`, `FORTE_USE_VLM=0`,
`FORTE_USE_OWL=1`, `ANALYZER_JURY_LIMIT_SEC=255`, `OMP_NUM_THREADS=6`, `DATA_DIR=/data`.
Остальное задано в Dockerfile (`ANALYZER_COMMAND`, `STATIC_DIR`, `HF_HUB_OFFLINE=1`).

## Команды
```bash
railway link            # проект cashguard, сервис app
railway up --detach     # сборка из Dockerfile и деплой
railway logs            # логи
railway variables       # переменные
```

## Проверка «как жюри»
```bash
BASE=https://app-production-d11c.up.railway.app KEY=<API_KEY> bash scripts/smoke_jury.sh video1.mov video2.mov
```
Проверяет: `POST /api/v1/analyze` (поле `file`) → `{"job_id","status":"processing"}`; `GET /api/v1/jobs/{id}` →
`done` с `verdict`/`violations[]`, JPEG base64 без префикса `data:`, 1–3 кадра; время от загрузки до `done` ≤ 300 с;
без ключа — 401. Обязательно прогнать самое длинное видео (~145 с) и 3–5 видео подряд.

## Ограничения
- CPU Railway разделяемый: анализ медленнее ноутбука; авто-бюджет снижает частоту кадров, ответ приходит всегда.
- VLM выключена (нет GPU/Metal). OWLv2 включён (признаки «предмет на столе между клиентами», «ключ в замке»).
- Один воркер в Go: при 5 одновременных загрузках хвост очереди получает урезанный бюджет (CV-результат без OWL).
