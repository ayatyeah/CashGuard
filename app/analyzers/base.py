# Interface from STACK.md §4 (shared with the service part).
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Detection:
    type_id: str                  # ключ из norms.yaml
    frame_id: int                 # главный кадр события (абсолютный индекс кадра в видео)
    evidence_frame_ids: list[int] # 1–3 кадра-доказательства
    description: str              # что видно на видео, без домыслов
    confidence: float             # 0..1


@dataclass
class AnalysisResult:
    detections: list[Detection]
    covered: list[tuple[float, float]]           # реально проверенные интервалы, сек
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # decode, detect, vlm, ...
    mode: str = "full"                           # full | cheap (мало бюджета)


class Analyzer(Protocol):
    def analyze(self, video_path: str, meta: dict, budget_sec: float) -> AnalysisResult: ...
