# Context for "killer feature" research agents (read first, then docs/research/CONTEXT.md and 00_SUMMARY.md)

State 2026-10-18: production stand https://fortecashguard.up.railway.app (CPU only, Railway). Pipeline: YOLOX+RTMPose+CLIP+OWLv2
→ camera-agnostic features → learned thresholds → KZ norm citations. Metrics on 40 dev videos: leave-one-cabinet-out
0.775 acc / 0 FP (recall 0.57), all-40 fit 0.90 / 0 FP, type correct 100%, timecode median 1.2–2 s. Jury: 5+ unseen
videos via API, 5-min limit, one video shot from a NEW camera angle, possibly NEW violation types. Scoring (100):
detection 25, localization 15, classification 15, regulatory grounding 15, robustness 10, product 10, technical 10.
Shortlist of ~30 out of ~300 teams is picked automatically from API results, then a 5-min pitch.
Our estimate: ~75 points; strongest = grounding/product/technical; weakest = recall (each missed video −5…−11 pts).
Known misses (need hand-action understanding): cash handed back to client, cash taken without ID, collection bag emptied
into safe without counting, rejected notes merged into bundle. Already tried & rejected: VLM-2B as detector, CLIP probes
on desk crops, video augmentation, DINOv2 zone anomaly, multi-feature logistic regression, dense-pose handover direction.
Hard constraints: local inference only (no cloud AI), permissive licenses (Apache/MIT/BSD), no face recognition,
runtime CPU on Railway (≈1.15 s per analysed frame) or Mac M3 Pro 18 GB (MLX) as a second stand. Remaining time: ~1 day.
A "killer feature" here = something that (a) raises expected jury score or shortlist odds measurably, (b) is implementable
in ≤ 8–12 hours by 2–3 people, (c) is defensible on the pitch (no fabricated claims). It can be a model, a product
capability, an evaluation trick, a data move, or a presentation asset — but it must be concrete and checkable.
