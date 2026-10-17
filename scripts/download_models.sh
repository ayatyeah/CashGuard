#!/usr/bin/env bash
# Download pretrained weights (all Apache-2.0 / MIT) into models/. Run once; afterwards no internet is needed.
set -euo pipefail
cd "$(dirname "$0")/../models"
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
