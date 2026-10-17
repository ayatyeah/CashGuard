# Context for research agents (read first)

Project: ForteBank Datathon — local AI service that flags cash-discipline violations by a bank cashier from a single
overhead CCTV-style camera (.mov, 464x848, 30 fps, 10–145 s). Output per video: verdict (violation / no_violation),
list of events with type, timecode, 1–3 evidence JPEG frames, description, KZ regulatory reference.
Scoring (100): detection 25, localization 15, classification 15, regulatory grounding 15, robustness to a NEW camera
angle 10, product 10, technical 10. Jury runs 5+ unseen videos through our API; 5-minute limit per video; test may
contain violation types not in training data. Misses are "unacceptable", false positives must be minimal.

Hard constraints: fully local inference (no cloud AI APIs — disqualification), models/code with licenses allowing
commercial use (Apache-2.0 / MIT / BSD; NOT GPL/AGPL/CC-BY-NC/research-only), no face recognition, runtime Mac M3
Pro 18 GB unified memory (MLX / CPU / CoreML), dev box Linux CPU only.

Data: 40 staged videos, 21 with a violation (1–3 examples per type), one room, several camera positions ("cabinets").
Current pipeline (Python 3.11): YOLOX-M/S (COCO, ONNX) + RTMPose-m + CLIP ViT-B/32 → person tracking → cashier/seat
detection → camera-agnostic features (relative to cashier height/seat) → candidate events (cashier leaves, door state
after entry, standing visitor, short client visit, drink, phone, safe zone change) → learned decision thresholds →
norms.yaml. Optional Qwen3-VL-2B (transformers/MLX) answers yes/no JSON about key moments with 2-of-3 self-consistency.
Metrics: leave-one-cabinet-out CV 0.725 acc / 0 FP; final model on all 40: 0.825 / 0 FP / type correct 100%.
Misses: cash handed back to client; client cash taken without ID; cash of previous client left on desk when next
enters; collection bag emptied into safe without counting; key left in safe lock; rejected notes merged into bundle.
Tried and rejected (hurt cabinet CV): 6-feature logistic regression; VLM-2B as detector of new types; CLIP linear probe
on desk crops; video augmentation (hflip/zoom/dark). Open datasets reviewed: none with permissive license fit.
Repo docs worth reading: docs/ml_model.md, docs/train_timelines.md, docs/normative_sources.md, STACK.md.
