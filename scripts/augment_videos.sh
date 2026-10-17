#!/usr/bin/env bash
# Geometric/photometric augmentation of the dataset videos (labels unchanged, timestamps unchanged).
# hflip = camera on the other side; zoom = different distance/framing; dark = different lighting.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/aug/videos; mkdir -p "$OUT"
for f in data/train/videos/*.mov data/test/videos/*.mov; do
  n=$(basename "${f%.mov}")
  case "$n" in Х-47|Х-48) continue;; esac   # duplicates of Х-4 / Х-5
  [ -f "$OUT/${n}__hflip.mov" ] || ffmpeg -v error -y -i "$f" -vf "hflip" -c:v libx264 -preset veryfast -crf 20 -an "$OUT/${n}__hflip.mov"
  [ -f "$OUT/${n}__zoom.mov" ]  || ffmpeg -v error -y -i "$f" -vf "crop=iw/1.12:ih/1.12:(iw-iw/1.12)*0.6:(ih-ih/1.12)*0.3,scale=464:848" -c:v libx264 -preset veryfast -crf 20 -an "$OUT/${n}__zoom.mov"
  [ -f "$OUT/${n}__dark.mov" ]  || ffmpeg -v error -y -i "$f" -vf "eq=brightness=-0.12:contrast=0.9:saturation=0.8" -c:v libx264 -preset veryfast -crf 20 -an "$OUT/${n}__dark.mov"
done
ls "$OUT" | wc -l
