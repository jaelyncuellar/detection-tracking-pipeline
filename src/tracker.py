"""Minimal IoU multi-object tracker (SORT-style, no Kalman filter yet).

Design notes for interview prep:
- Each track stores its last box, a persistent integer ID, age, and a
  hit/miss counter. Association is greedy by IoU (highest pair first).
- A track survives up to `max_age` consecutive missed frames, which is
  what lets IDs persist through a short occlusion.
- Known limitation (Week 2 upgrade): no motion model, so a fast object
  that moves more than its own width per frame breaks IoU association.
  SORT/ByteTrack fix this with a Kalman filter + a second low-score
  association pass. That upgrade is Project 1's Week 2 polish item.
"""

from __future__ import annotations

import numpy as np


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union for boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


class Track:
    _next_id = 1

    def __init__(self, box: np.ndarray, label: str, score: float):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box.astype(float)
        self.label = label
        self.score = score
        self.hits = 1
        self.missed = 0

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 1


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []

    def update(self, detections: list[dict]) -> list[Track]:
        """detections: [{'box': np.array([x1,y1,x2,y2]), 'label': str, 'score': float}]"""
        # Score every (track, detection) pair, match greedily best-first.
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det["label"] != track.label:
                    continue
                pairs.append((iou(track.box, det["box"]), ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            track = self.tracks[ti]
            track.box = detections[di]["box"].astype(float)
            track.score = detections[di]["score"]
            track.hits += 1
            track.missed = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Age unmatched tracks; retire ones unseen too long.
        survivors = []
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missed += 1
            if track.missed <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        # Spawn new tracks for unmatched detections.
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(Track(det["box"], det["label"], det["score"]))

        return list(self.tracks)
