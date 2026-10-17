# CashGuard / ForteWin — single container: React UI (static) + Go API + Python ML analyzer (CPU only).
# Jury contract: POST /api/v1/analyze (field file), GET /api/v1/jobs/{id}; X-API-Key. See docs/api_contract_gap.md.

# ---------- 1. frontend ----------
FROM node:22-alpine AS web
WORKDIR /app
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci
COPY src/frontend/ .
RUN npm run build

# ---------- 2. backend ----------
FROM golang:1.25-alpine AS api
WORKDIR /src
COPY src/backend/go.mod src/backend/go.sum ./
RUN go mod download
COPY src/backend/ .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server

# ---------- 3. runtime: python + models + go server + static ----------
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/hf OMP_NUM_THREADS=6 FORTE_USE_VLM=0 FORTE_USE_OWL=1 FORTE_MODELS_DIR=/app/models \
    ADDR=0.0.0.0:8080 DATA_DIR=/data STATIC_DIR=/app/static GIN_MODE=release \
    DEMO_MODE=false ANALYZER_COMMAND=/app/bin/cashguard-analyzer CASHGUARD_PYTHON=/usr/local/bin/python \
    ANALYZER_JURY_LIMIT_SEC=255
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-ml.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
# weights: YOLOX / RTMPose (GitHub, OpenMMLab) + CLIP (HF cache under HF_HOME)
COPY scripts/download_models.sh scripts/download_models.sh
RUN mkdir -p models && HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 bash scripts/download_models.sh \
    && HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as s; s('google/owlv2-base-patch16-ensemble', allow_patterns=['*.json','model.safetensors','*.txt'])"
COPY app/ app/
COPY scripts/ scripts/
COPY bin/ bin/
COPY models/*.json models/
COPY --from=api /out/server /app/server
COPY --from=web /app/dist /app/static
RUN chmod +x bin/cashguard-analyzer && mkdir -p /data/uploads
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fsS http://127.0.0.1:8080/api/ready || exit 1
CMD ["/app/server"]
