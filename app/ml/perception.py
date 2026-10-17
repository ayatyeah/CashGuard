"""Frame-level perception: COCO object detection (YOLOX, ONNX) + body pose (RTMPose, ONNX).

Both models are Apache-2.0 and run on CPU through onnxruntime (Linux / macOS arm64), no CUDA needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.environ.get("FORTE_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# classes we keep (everything else is noise for a cash room)
KEEP = {
    "person", "backpack", "handbag", "suitcase", "umbrella", "bottle", "wine glass", "cup", "bowl", "banana",
    "apple", "sandwich", "donut", "cake", "chair", "dining table", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "book", "scissors", "printer", "microwave", "toaster", "refrigerator", "clock",
}
KEEP_IDS = np.array([i for i, n in enumerate(COCO) if n in KEEP])


def _session(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class YoloxDetector:
    """Official Megvii YOLOX ONNX export (raw head output, decoded here)."""

    def __init__(self, path: str, input_size: int = 640, threads: int = 4):
        self.sess = _session(path, threads)
        self.inp = self.sess.get_inputs()[0].name
        self.size = input_size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def __call__(self, img: np.ndarray, conf: float = 0.25, nms: float = 0.5):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        sc = scores[np.arange(len(cls)), cls]
        m = (sc >= conf) & np.isin(cls, KEEP_IDS)
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        xy, wh, sc, cls = xy[m], wh[m], sc[m], cls[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = cv2.dnn.NMSBoxesBatched(
            [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
            sc.tolist(), cls.tolist(), conf, nms)
        keep = np.array(keep, dtype=np.int64).reshape(-1)
        return boxes[keep].astype(np.float32), sc[keep].astype(np.float32), cls[keep]


class PoseEstimator:
    """RTMPose-m body7 (17 COCO keypoints), top-down on given person boxes."""

    def __init__(self, path: str, threads: int = 4):
        from rtmlib import RTMPose
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        self.model = RTMPose(onnx_model=path, model_input_size=(192, 256), backend="onnxruntime", device="cpu")

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        if len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        kp, sc = self.model(img, bboxes=[b.tolist() for b in boxes])
        return kp.astype(np.float32), sc.astype(np.float32)


@dataclass
class FrameObs:
    t: float                      # seconds
    frame_id: int
    boxes: np.ndarray             # (N,4) xyxy all kept detections
    scores: np.ndarray            # (N,)
    classes: np.ndarray           # (N,) COCO ids
    kpts: np.ndarray              # (P,17,2) for person detections (same order as person rows)
    kpt_scores: np.ndarray        # (P,17)
    brightness: float
    motion: float                 # mean abs diff vs previous sampled frame (gray, downscaled)
    extra: dict = field(default_factory=dict)


def find_model(*names: str) -> str:
    for n in names:
        for root, _, files in os.walk(MODELS_DIR):
            if n in files:
                return os.path.join(root, n)
    raise FileNotFoundError(f"model not found in {MODELS_DIR}: {names}")


def iter_frames(path: str, fps: float, max_side: int = 960):
    """Yield (frame_id, t, bgr) sampled at ~fps. Decodes sequentially (grab) to keep exact frame indices."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                s = max_side / max(h, w)
                if s < 1:
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                yield i, i / src_fps, frame
        i += 1
    cap.release()


def video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "nb_frames": n, "duration": n / fps if fps else 0.0, "width": w, "height": h}


def work_area(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, shape, prev=None):
    """Region around people + laptop (where cash handling happens); kept from previous frame when room is empty."""
    H, W = shape[:2]
    m = ((classes == 0) & (scores > 0.4)) | (classes == COCO.index("laptop"))
    if not m.any():
        return prev
    b = boxes[m]
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * 1.3 + 40, (y2 - y1) * 1.3 + 40
    s = max(w, h)
    box = np.array([max(0, cx - s / 2), max(0, cy - s / 2), min(W, cx + s / 2), min(H, cy + s / 2)], np.float32)
    if prev is not None:  # smooth to avoid jitter
        box = 0.5 * box + 0.5 * prev
    return box


class Perception:
    def __init__(self, threads: int = 6, det_model: str = "yolox_m.onnx", crop_model: str | None = "yolox_s.onnx"):
        self.det = YoloxDetector(find_model(det_model), threads=threads)
        self.det_crop = YoloxDetector(find_model(crop_model), threads=threads) if crop_model else None
        self.pose = PoseEstimator(find_model("end2end.onnx"), threads=threads)

    def frame(self, fid: int, t: float, img: np.ndarray, state: dict) -> FrameObs:
        boxes, scores, classes = self.det(img)
        pm = classes == 0
        kp, ks = self.pose(img, boxes[pm])
        g = cv2.cvtColor(cv2.resize(img, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev = state.get("gray")
        motion = float(np.abs(g - prev).mean()) if prev is not None else 0.0
        state["gray"] = g
        extra = {}
        wa = work_area(boxes, scores, classes, img.shape, state.get("wa"))
        state["wa"] = wa
        if wa is not None:
            extra["work_area"] = wa
            if self.det_crop is not None:
                x1, y1, x2, y2 = wa.astype(int)
                cb, cs, cc = self.det_crop(img[y1:y2, x1:x2], conf=0.15)
                cb[:, [0, 2]] += x1
                cb[:, [1, 3]] += y1
                extra["crop_det"] = (cb, cs, cc)
        return FrameObs(t, fid, boxes, scores, classes, kp, ks, float(g.mean()), motion, extra)

    def run(self, path: str, fps: float = 2.0, on_frame=None):
        obs: list[FrameObs] = []
        state: dict = {}
        for fid, t, img in iter_frames(path, fps):
            o = self.frame(fid, t, img, state)
            obs.append(o)
            if on_frame:
                on_frame(o, img)
        return obs
