"""Kalman-filter multi-object tracker (SORT-style) — Week 2 upgrade.

Drop-in alternative to tracker.IoUTracker with the same interface:
    tracker = KalmanTracker(iou_threshold=0.3, max_age=10)
    tracks = tracker.update(detections)   # [{'box','label','score'}] -> [Track]

Design (mirrors SORT, Bewley et al. 2016):
- Each track carries a constant-velocity Kalman filter over the box
  centre, width and height: state [cx, cy, w, h, vx, vy, vw, vh],
  measurement [cx, cy, w, h]. numpy only — no scipy/filterpy needed.
- Association is greedy best-IoU, same as the IoU tracker, except each
  track's box is first *predicted forward* one frame. That prediction is
  what fixes the IoU tracker's known failure case: an object that moves
  more than its own width per frame has ~zero IoU with its last box, but
  its predicted box lands on the new detection, so the ID survives.
- Unmatched tracks coast on their prediction for up to `max_age` frames
  (same occlusion behaviour as the IoU tracker); matched tracks are
  corrected with the detection.

Kept as a separate module so the before/after ID-switch comparison the
metrics note needs can run both trackers on the same detections.
"""

from __future__ import annotations

import numpy as np

from tracker import Track, iou


class _KalmanBox:
    """Constant-velocity Kalman filter for one [x1, y1, x2, y2] box."""

    def __init__(self, box: np.ndarray):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # velocity starts very uncertain
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] *= 10.0  # allow acceleration between frames
        self.R = np.eye(4) * 1.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box: np.ndarray) -> None:
        z = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0,
                      box[2] - box[0], box[3] - box[1]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class KalmanTrack(Track):
    def __init__(self, box: np.ndarray, label: str, score: float):
        super().__init__(box, label, score)
        self.kf = _KalmanBox(box)


class KalmanTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[KalmanTrack] = []

    def update(self, detections: list[dict]) -> list[KalmanTrack]:
        # 1. Predict every track one frame forward; associate against the
        #    predicted boxes, not last frame's boxes.
        predicted = [t.kf.predict() for t in self.tracks]

        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det["label"] != track.label:
                    continue
                pairs.append((iou(predicted[ti], det["box"]), ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        def _apply_match(ti: int, di: int) -> None:
            track = self.tracks[ti]
            track.kf.update(detections[di]["box"].astype(float))
            track.box = track.kf.box()
            track.score = detections[di]["score"]
            track.hits += 1
            track.missed = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            _apply_match(ti, di)

        # Second pass (ByteTrack-style): a cold-start track has velocity 0,
        # so a >1-box-width jump on its second frame scores IoU 0 and the
        # filter never learns the velocity. Gate the leftovers by centre
        # distance to the predicted box (within 2x the box's larger side),
        # nearest first, same label only.
        leftovers = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_tracks:
                continue
            pc = ((predicted[ti][0] + predicted[ti][2]) / 2.0,
                  (predicted[ti][1] + predicted[ti][3]) / 2.0)
            gate = 2.0 * max(predicted[ti][2] - predicted[ti][0],
                             predicted[ti][3] - predicted[ti][1])
            for di, det in enumerate(detections):
                if di in matched_dets or det["label"] != track.label:
                    continue
                dc = ((det["box"][0] + det["box"][2]) / 2.0,
                      (det["box"][1] + det["box"][3]) / 2.0)
                dist = float(np.hypot(pc[0] - dc[0], pc[1] - dc[1]))
                if dist <= gate:
                    leftovers.append((dist, ti, di))
        leftovers.sort()
        for _dist, ti, di in leftovers:
            if ti in matched_tracks or di in matched_dets:
                continue
            _apply_match(ti, di)

        # 2. Age unmatched tracks; they coast on the Kalman prediction.
        survivors = []
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missed += 1
                track.box = predicted[ti]
            if track.missed <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        # 3. Spawn new tracks for unmatched detections.
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(
                    KalmanTrack(det["box"], det["label"], det["score"]))

        return list(self.tracks)
