"""Detection backends.

Two backends share one output format so the tracker never cares which ran:

- ``color``: demo detector for the synthetic sequence in
  ``make_synthetic_sequence.py``. Finds saturated object blobs by color
  segmentation. Exists so the full pipeline runs offline, end-to-end,
  before any dataset download or model weights are involved.
- ``yolo``: real detector for KITTI / dashcam frames via Ultralytics
  YOLO. Used once a real sequence is dropped into ``data/``.
  Requires ``pip install ultralytics`` and downloads weights on first run.
"""

from __future__ import annotations

import numpy as np

# Synthetic-scene palette: label -> RGB center color used by the demo scene.
PALETTE = {
    "car": (220, 60, 60),
    "truck": (60, 90, 220),
    "pedestrian": (240, 180, 40),
}


def detect_color(frame: np.ndarray, min_area: int = 200) -> list[dict]:
    """Find objects in a synthetic frame by nearest-palette-color segmentation."""
    detections = []
    img = frame.astype(int)
    for label, (r, g, b) in PALETTE.items():
        dist = np.abs(img[:, :, 0] - r) + np.abs(img[:, :, 1] - g) + np.abs(img[:, :, 2] - b)
        mask = dist < 90
        ys, xs = np.nonzero(mask)
        if len(xs) < min_area:
            continue
        # The synthetic scene has at most one object per class per frame in v1;
        # cluster by connected x-runs if two same-class objects appear.
        order = np.argsort(xs)
        xs_s, ys_s = xs[order], ys[order]
        splits = np.where(np.diff(xs_s) > 30)[0]
        groups = np.split(np.arange(len(xs_s)), splits + 1)
        for grp in groups:
            if len(grp) < min_area:
                continue
            gx, gy = xs_s[grp], ys_s[grp]
            detections.append(
                {
                    "box": np.array([gx.min(), gy.min(), gx.max(), gy.max()], dtype=float),
                    "label": label,
                    "score": 0.99,
                }
            )
    return detections


_ONNX_SESSION = None
_ONNX_INPUT_NAME = None

# COCO class ids used by YOLO11 -> project labels (same keep-set as detect_yolo).
_ONNX_LABELS = {
    0: "pedestrian",  # person
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _onnx_session(weights: str | None = None):
    """Load (once) the YOLO11n ONNX session. Weights: $YOLO_ONNX_PATH, else
    ``yolo11n.onnx`` next to the project root / cwd — the ultralytics release
    asset, ~11 MB, so this backend needs no torch install."""
    global _ONNX_SESSION, _ONNX_INPUT_NAME
    if _ONNX_SESSION is None:
        import os
        from pathlib import Path

        import onnxruntime as ort

        candidates = [
            weights,
            os.environ.get("YOLO_ONNX_PATH"),
            Path(__file__).resolve().parent.parent / "yolo11n.onnx",
            Path.cwd() / "yolo11n.onnx",
        ]
        path = next((c for c in candidates if c and Path(c).exists()), None)
        if path is None:
            raise FileNotFoundError(
                "yolo11n.onnx not found — download the ultralytics release asset "
                "or set YOLO_ONNX_PATH"
            )
        _ONNX_SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _ONNX_INPUT_NAME = _ONNX_SESSION.get_inputs()[0].name
    return _ONNX_SESSION, _ONNX_INPUT_NAME


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """Greedy non-max suppression; boxes are [x1, y1, x2, y2]."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i][0], boxes[rest, 0])
        y1 = np.maximum(boxes[i][1], boxes[rest, 1])
        x2 = np.minimum(boxes[i][2], boxes[rest, 2])
        y2 = np.minimum(boxes[i][3], boxes[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(1e-9, area_i + area_r - inter)
        order = rest[iou <= iou_thr]
    return keep


def detect_yolo_onnx(frame: np.ndarray, conf: float = 0.35) -> list[dict]:
    """Run YOLO11n on a real frame via ONNX Runtime (no torch / ultralytics).

    Same weights family and same output format as ``detect_yolo``:
    letterbox to 640, forward pass, per-class NMS, boxes un-letterboxed
    back to frame coordinates.
    """
    from PIL import Image

    session, input_name = _onnx_session()
    h, w = frame.shape[:2]
    size = 640
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = np.asarray(Image.fromarray(frame).resize((new_w, new_h)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    out = session.run(None, {input_name: tensor})[0][0].T  # [8400, 84]
    boxes_cxcywh, class_scores = out[:, :4], out[:, 4:]
    cls = class_scores.argmax(axis=1)
    score = class_scores[np.arange(len(out)), cls]
    mask = (score >= conf) & np.isin(cls, list(_ONNX_LABELS))
    if not mask.any():
        return []
    boxes_cxcywh, cls, score = boxes_cxcywh[mask], cls[mask], score[mask]
    boxes = np.stack(
        [
            boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
            boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
            boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2,
            boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2,
        ],
        axis=1,
    )

    detections = []
    for label_id in np.unique(cls):
        idx = np.where(cls == label_id)[0]
        for k in _nms(boxes[idx], score[idx]):
            i = int(idx[k])
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                {
                    "box": np.array(
                        [
                            np.clip((x1 - pad_x) / scale, 0, w),
                            np.clip((y1 - pad_y) / scale, 0, h),
                            np.clip((x2 - pad_x) / scale, 0, w),
                            np.clip((y2 - pad_y) / scale, 0, h),
                        ],
                        dtype=float,
                    ),
                    "label": _ONNX_LABELS[int(label_id)],
                    "score": float(score[i]),
                }
            )
    return detections


def detect_yolo(frame: np.ndarray, model=None, conf: float = 0.35) -> list[dict]:
    """Run YOLO on a real frame. Lazily loads the model on first call."""
    from ultralytics import YOLO  # imported here so the demo path needs no torch

    if model is None:
        model = YOLO("yolo11n.pt")
    keep = {"car", "truck", "bus", "person", "bicycle", "motorcycle"}
    detections = []
    for result in model.predict(frame, conf=conf, verbose=False):
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls)]
            if label not in keep:
                continue
            detections.append(
                {
                    "box": box.xyxy[0].cpu().numpy().astype(float),
                    "label": "pedestrian" if label == "person" else label,
                    "score": float(box.conf),
                }
            )
    return detections
