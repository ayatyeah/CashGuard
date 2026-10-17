#!/usr/bin/env bash
# Download pretrained weights (all Apache-2.0 / MIT) into models/. Run once; afterwards no internet is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/download_models.sh [--vlm MODEL_ID [REVISION]]"
  echo "Default: YOLOX, RTMPose and CLIP. --vlm: download only the selected Qwen3-VL 4-bit snapshot into models/vlm/."
  exit 0
fi
if [[ "${1:-}" == "--vlm" ]]; then
  case "${2:-}" in
    mlx-community/Qwen3-VL-2B-Instruct-4bit|mlx-community/Qwen3-VL-4B-Instruct-4bit) ;;
    *) echo "Expected an audited mlx-community/Qwen3-VL-{2B,4B}-Instruct-4bit model ID" >&2; exit 2 ;;
  esac
  if (( $# > 3 )); then echo "Too many arguments" >&2; exit 2; fi
  # Explicit optional download: never run by default. Larger snapshots exceed 2 GB;
  # consult docs/mac_18gb.md before choosing 4B. No dataset is uploaded.
  python - "$2" "${3:-main}" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

model_id, revision = sys.argv[1:]
info = HfApi().model_info(model_id, revision=revision)
destination = Path("models/vlm") / model_id.replace("/", "--")
snapshot_download(model_id, revision=info.sha, local_dir=str(destination))
(destination / "forte_download.json").write_text(json.dumps(
    {"model_id": model_id, "revision": info.sha}, indent=2) + "\n")
print(f"Local model: {destination.resolve()} (revision {info.sha})")
PY
  exit 0
fi
if (( $# )); then echo "Unknown arguments; use --help" >&2; exit 2; fi
mkdir -p models
cd models
dl(){ for i in $(seq 1 20); do curl -sL -C - --speed-limit 20000 --speed-time 15 --connect-timeout 15 -o "$1" "$2" && return 0; sleep 2; done; return 1; }
# YOLOX-M / YOLOX-S, COCO (Apache-2.0, Megvii)
dl yolox_m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
dl yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
# RTMPose-m body7 (Apache-2.0, OpenMMLab)
if [ ! -d rtmpose_m ]; then
  dl rtmpose_m.zip https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
  unzip -o -q rtmpose_m.zip -d rtmpose_m && rm rtmpose_m.zip
fi
# CLIP ViT-B/32 (MIT, OpenAI) → HuggingFace cache
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', allow_patterns=['*.json','*.txt','pytorch_model.bin'])"
echo "models ready"
