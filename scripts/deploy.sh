#!/usr/bin/env bash
# Start CashGuard on an always-on machine: backend + ML analyzer + frontend + public tunnel,
# then print the public URL. Safe to re-run; containers restart by themselves (restart: unless-stopped).
#   usage: scripts/deploy.sh            (reads .env)
#          ENV_FILE=_env scripts/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE="${ENV_FILE:-.env}"

[ -f "$ENV_FILE" ] || { echo "Нет $ENV_FILE: cp .env.example .env и задайте POSTGRES_PASSWORD и API_KEY" >&2; exit 1; }
grep -q '^DEMO_MODE=false' "$ENV_FILE" || echo "Внимание: в $ENV_FILE не DEMO_MODE=false — сервис будет в демо-режиме" >&2
docker info >/dev/null 2>&1 || { echo "Docker не запущен" >&2; exit 1; }

docker compose --env-file "$ENV_FILE" --profile public up -d --build

url=""
for _ in $(seq 1 60); do
  url=$(docker compose --env-file "$ENV_FILE" logs tunnel 2>/dev/null | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)
  [ -n "$url" ] && curl -sf "$url/api/health" >/dev/null 2>&1 && break
  sleep 3
done
docker compose --env-file "$ENV_FILE" ps --format '{{.Service}}: {{.Status}}'
echo "Публичный адрес: ${url:-не получен, смотрите docker compose logs tunnel}"
echo "API для жюри:    ${url:-<адрес>}/api/v1/analyze"
