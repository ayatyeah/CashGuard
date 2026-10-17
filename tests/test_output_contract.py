"""Output contract of scripts/cashguard_analyzer.py as enforced by the Go backend
(src/backend/internal/infrastructure/analyzer.go, CommandAnalyzer.Analyze).

`go_validate` mirrors the Go checks one-to-one:
  * stdout must be JSON that unmarshals into domain.Job (string fields must be strings, numbers numbers);
  * status == "done" and verdict in {violation, no_violation};
  * verdict == violation  ⇔  violations non-empty;
  * every violation: non-empty type and description, timecode_sec >= 0, >= 1 evidence frame;
  * every frame: timestamp_sec >= 0, image_base64 is standard base64 (padded) that decodes to an image
    in a format registered in the Go binary (PNG via analyzer.go, JPEG via cmd/server/main.go).
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
import sys
import textwrap

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ContractError(ValueError):
    pass


def _str(d: dict, k: str) -> str:
    v = d.get(k)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ContractError(f"{k}: not a string (json.Unmarshal fails)")
    return v


def _num(d: dict, k: str) -> float:
    v = d.get(k)
    if v is None:
        return 0.0
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ContractError(f"{k}: not a number (json.Unmarshal fails)")
    return float(v)


def _is_go_image(b: bytes) -> bool:
    if not (b.startswith(b"\x89PNG\r\n\x1a\n") or b.startswith(b"\xff\xd8")):
        return False  # only PNG and JPEG decoders are registered in the Go binary
    # full decode: slightly stricter than Go's header-only image.DecodeConfig
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED) is not None


def go_validate(payload) -> dict:
    """Validate analyzer stdout (bytes/str) or an already-built dict exactly like the Go backend."""
    if isinstance(payload, (bytes, str)):
        try:
            result = json.loads(payload)
        except ValueError as e:
            raise ContractError("анализатор вернул некорректный JSON") from e
    else:
        result = json.loads(json.dumps(payload, ensure_ascii=False))  # what actually goes over the pipe
    if not isinstance(result, dict):
        raise ContractError("анализатор вернул некорректный JSON")
    violations = result.get("violations")
    if violations is None:
        violations = []
    if not isinstance(violations, list):
        raise ContractError("violations: not a list")
    if _str(result, "status") != "done" or _str(result, "verdict") not in ("violation", "no_violation"):
        raise ContractError("некорректный статус или вердикт анализатора")
    if result["verdict"] == "violation" and not violations:
        raise ContractError("нет доказательств нарушения")
    if result["verdict"] == "no_violation" and violations:
        raise ContractError("вердикт противоречит списку нарушений")
    for v in violations:
        if not isinstance(v, dict):
            raise ContractError("violation: not an object")
        frames = v.get("evidence_frames") or []
        if not isinstance(frames, list):
            raise ContractError("evidence_frames: not a list")
        if not _str(v, "type") or not _str(v, "description") or _num(v, "timecode_sec") < 0 or not frames:
            raise ContractError("неполные данные нарушения")
        for f in frames:
            if _num(f, "timestamp_sec") < 0:
                raise ContractError("некорректный таймкод")
            try:
                raw = base64.b64decode(_str(f, "image_base64"), validate=True)
            except (binascii.Error, ValueError) as e:
                raise ContractError("некорректный кадр") from e
            if not _is_go_image(raw):
                raise ContractError("кадр не является изображением")
    return result


# ---------------------------------------------------------------- fixtures
def synthetic_frame(h=90, w=160) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (40, 80, 120)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (200, 220, 30), -1)
    return img


def jpeg_b64(img=None) -> str:
    """Same encoding as scripts/analyze_video.frame_b64."""
    ok, buf = cv2.imencode(".jpg", synthetic_frame() if img is None else img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return base64.b64encode(buf).decode()


def png_b64() -> str:
    ok, buf = cv2.imencode(".png", synthetic_frame())
    assert ok
    return base64.b64encode(buf).decode()


def good_result() -> dict:
    return {"status": "done", "verdict": "violation", "violations": [{
        "type": "Мобильный телефон у кассира", "timecode_sec": 12.5,
        "description": "Около 12 с в руках кассира виден мобильный телефон.",
        "evidence_frames": [{"timestamp_sec": 12.5, "image_base64": jpeg_b64()},
                            {"timestamp_sec": 13.0, "image_base64": png_b64()}],
        "rule": {"title": "Doc", "section": "п. 12", "url": "https://adilet.zan.kz/"},
    }]}


# ---------------------------------------------------------------- validator self-tests
def test_valid_violation_result_passes():
    go_validate(good_result())
    go_validate(json.dumps(good_result(), ensure_ascii=False).encode())


@pytest.mark.parametrize("violations", [[], None])
def test_valid_no_violation_result_passes(violations):
    go_validate({"status": "done", "verdict": "no_violation", "violations": violations})


def test_synthetic_jpeg_roundtrip():
    img = synthetic_frame()
    raw = base64.b64decode(jpeg_b64(img), validate=True)
    assert raw[:2] == b"\xff\xd8"
    dec = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert dec.shape == img.shape
    assert np.abs(dec.astype(int) - img.astype(int)).mean() < 5


def _set(path, value):
    def mut(r):
        obj = r
        for k in path[:-1]:
            obj = obj[k]
        obj[path[-1]] = value
    return mut


def _bmp(r):
    ok, buf = cv2.imencode(".bmp", synthetic_frame())
    r["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(buf).decode()


V0, F0 = ("violations", 0), ("violations", 0, "evidence_frames", 0)


@pytest.mark.parametrize("mutate", [
    _set(("status",), "processing"),
    _set(("verdict",), "maybe"),
    _set(("verdict",), "no_violation"),                   # contradicts non-empty violations
    _set(("violations",), []),                            # violation verdict without evidence
    _set(V0 + ("type",), ""),
    _set(V0 + ("description",), ""),
    _set(V0 + ("type",), 5),                              # wrong JSON type → Unmarshal error
    _set(V0 + ("timecode_sec",), -0.1),
    _set(V0 + ("timecode_sec",), "12.5"),
    _set(V0 + ("evidence_frames",), []),
    _set(F0 + ("timestamp_sec",), -1),
    _set(F0 + ("image_base64",), ""),
    _set(F0 + ("image_base64",), "not base64!!"),
    _set(F0 + ("image_base64",), base64.b64encode(b"hello, not an image").decode()),
    _set(F0 + ("image_base64",), jpeg_b64()[:-1]),         # truncated → bad base64 length/padding
    _bmp,                                                 # real image, but no BMP decoder in the Go binary
])
def test_invalid_results_are_rejected(mutate):
    r = good_result()
    mutate(r)
    with pytest.raises(ContractError):
        go_validate(r)


def test_invalid_json_rejected():
    with pytest.raises(ContractError):
        go_validate(b"timings={} warnings=[]\n{\"status\": \"done\"}")  # log line leaked into stdout


# ---------------------------------------------------------------- real frame_b64 on a tiny synthetic video
def test_frame_b64_on_synthetic_video(tmp_path):
    from scripts.analyze_video import frame_b64

    path = str(tmp_path / "tiny.avi")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    if not wr.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for k in range(5):
        wr.write(np.full((48, 64, 3), 40 * k, np.uint8))
    wr.release()
    b = frame_b64(path, 2)
    assert b
    raw = base64.b64decode(b, validate=True)
    assert _is_go_image(raw) and raw[:2] == b"\xff\xd8"
    assert frame_b64(path, 10_000) == ""  # unreadable frame → "" (the analyzer then drops it)


# ---------------------------------------------------------------- end-to-end: the real launcher, faked models
RUNNER = textwrap.dedent(r'''
    import base64, importlib.abc, runpy, sys
    root, scenario = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("torch", "transformers", "mlx_vlm", "mlx"):
                raise ImportError("blocked in tests: " + name)
    sys.meta_path.insert(0, Block())

    import cv2, numpy as np, yaml
    import onnxruntime
    def _no_session(*a, **k):
        raise RuntimeError("ONNX session blocked in tests")
    onnxruntime.InferenceSession = _no_session

    import app.analyzers.ml as A
    import app.ml.perception as P
    import scripts.analyze_video as AV
    from app.analyzers.base import AnalysisResult, Detection
    from app.ml import pipeline

    norms = yaml.safe_load(open(root + "/app/rules/norms.yaml", encoding="utf-8"))

    def frame_b64(path, fid):
        if fid < 0:
            return ""  # simulate an unreadable frame
        img = np.full((72, 128, 3), fid % 255, np.uint8)
        return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()

    def ev(type_id, t):
        return pipeline.describe(pipeline.Event(type_id, t, 0, [0], 0.9, "x", {}), norms)

    DETS = {
        "mixed": [Detection("phone_use", 300, [300, 310, 320, 330], ev("phone_use", 12.0), 0.9),
                  Detection("unknown_review", 50, [50], ev("unknown_review", 2.0), 0.6)],
        "all_dropped": [Detection("food_drink", 100, [-1], ev("food_drink", 4.0), 0.7)],
        "none": [],
    }

    class FakeAnalyzer:
        def __init__(self, *a, **k):
            self.models = type("M", (), {"norms": norms})()
        def analyze(self, path, meta, budget):
            print("library noise on stdout")  # must be redirected to stderr by the launcher
            return AnalysisResult(detections=DETS[scenario], covered=[(0.0, 60.0)],
                                  warnings=["w"], timings={"perception": 0.1})

    A.MLAnalyzer = FakeAnalyzer
    P.video_meta = lambda path: {"fps": 25.0, "frames": 1500, "duration": 60.0, "width": 1280, "height": 720}
    AV.frame_b64 = frame_b64

    script = root + "/scripts/cashguard_analyzer.py"
    sys.argv = [script, "/nonexistent/video.mp4"]
    runpy.run_path(script, run_name="__main__")
''')


def run_launcher(tmp_path, scenario, success=True):
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, str(runner), ROOT, scenario], capture_output=True, timeout=60, env=env)
    assert (p.returncode == 0) == success, p.stderr.decode(errors="replace")[-3000:]
    return p


def test_launcher_end_to_end_mixed(tmp_path, norms):
    p = run_launcher(tmp_path, "mixed")
    res = go_validate(p.stdout)          # stdout must be exactly one JSON document
    assert b"library noise" in p.stderr and b"library noise" not in p.stdout
    assert res["verdict"] == "violation"
    vs = res["violations"]
    assert len(vs) == 2
    assert res["warnings"] == ["w"]
    phone, unknown = vs
    assert phone["type"] == norms["phone_use"]["title"]
    assert phone["timecode_sec"] == 12.0                       # frame 300 / 25 fps
    assert [f["timestamp_sec"] for f in phone["evidence_frames"]] == [12.0, 12.4, 12.8]  # max 3 frames
    assert phone["rule"] == {"title": norms["phone_use"]["document"], "section": norms["phone_use"]["clause"],
                             "url": norms["phone_use"]["url"]}
    assert "rule" not in unknown                                # source_verified: false → no rule
    assert unknown["type"] == norms["unknown_review"]["title"]


def test_launcher_end_to_end_no_violation(tmp_path):
    res = go_validate(run_launcher(tmp_path, "none").stdout)
    assert res == {"status": "done", "verdict": "no_violation", "violations": [], "warnings": ["w"]}


def test_launcher_missing_evidence_fails(tmp_path):
    p = run_launcher(tmp_path, "all_dropped", success=False)
    assert not p.stdout  # Never turn a lost detection into a clean verdict.
