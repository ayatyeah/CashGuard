# Backend + local ML analyzer in one image (build context: repository root).
# Weights are downloaded once at build time; at runtime the analysis needs no internet (ТЗ §7).
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY src/backend/go.mod src/backend/go.sum ./
RUN go mod download
COPY src/backend/ .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip wget ca-certificates libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ml
COPY requirements-ml.txt .
# CPU-only torch; rtmlib without deps, otherwise it pulls GUI OpenCV next to the headless build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^rtmlib' requirements-ml.txt | pip install --no-cache-dir -r /dev/stdin \
 && pip install --no-cache-dir --no-deps rtmlib && pip install --no-cache-dir tqdm
ENV HF_HOME=/opt/hf
COPY scripts/download_models.sh scripts/
RUN mkdir -p models && bash scripts/download_models.sh
COPY app/ app/
COPY scripts/ scripts/
COPY bin/ bin/
COPY models/*.json models/
COPY --from=build /out/server /app/server
# uid/gid 100:101 match the alpine image (src/backend/Dockerfile), so existing uploads volumes stay writable
RUN groupadd --system --gid 101 app && useradd --system --uid 100 --gid 101 app && mkdir -p /data/uploads && chown -R app:app /data
USER app
# FORTE_COST_PER_FRAME: measured ~1.1 s per sampled frame on Apple M3 8 GB inside Docker (6 threads)
ENV ADDR=0.0.0.0:8080 DATA_DIR=/data GIN_MODE=release \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    FORTE_COST_PER_FRAME=1.2 \
    CASHGUARD_PYTHON=/usr/local/bin/python ANALYZER_COMMAND=/opt/ml/bin/cashguard-analyzer
EXPOSE 8080
CMD ["/app/server"]
