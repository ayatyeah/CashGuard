# Windows version of scripts/deploy.sh: start backend + ML analyzer + frontend + public tunnel
# and print the public URL. Safe to re-run.
#   usage: powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
# 'Continue': Windows PowerShell 5.1 turns docker's stderr progress output into terminating errors under 'Stop'
$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')
$envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { '.env' }

if (-not (Test-Path $envFile)) { throw "Нет ${envFile}: скопируйте .env.example в .env и задайте POSTGRES_PASSWORD и API_KEY" }
if (-not (Select-String -Path $envFile -Pattern '^DEMO_MODE=false' -Quiet)) {
  Write-Warning "В $envFile не DEMO_MODE=false — сервис будет в демо-режиме"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker не запущен' }

docker compose --env-file $envFile --profile public up -d --build
if ($LASTEXITCODE -ne 0) { throw 'docker compose завершился с ошибкой' }

$url = $null
for ($i = 0; $i -lt 60; $i++) {
  $found = docker compose --env-file $envFile logs tunnel 2>$null | Select-String -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches
  if ($found) {
    $url = $found[-1].Matches[-1].Value
    try { Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 5 | Out-Null; break } catch {}
  }
  Start-Sleep -Seconds 3
}
docker compose --env-file $envFile ps --format '{{.Service}}: {{.Status}}'
if ($url) { Write-Output "Публичный адрес: $url"; Write-Output "API для жюри:    $url/api/v1/analyze" }
else { Write-Output 'Публичный адрес не получен, смотрите: docker compose logs tunnel' }
