"""Generate a synthetic 'driving scene' sequence so the pipeline has data
on day one — no KITTI download required.

Scene: a gray road band, three objects (car, truck, pedestrian) moving at
different speeds, sensor noise (per-pixel jitter), and one scripted
occlusion: a gray 'pole' covers the car for frames 30-36. A good tracker
keeps the car's ID through that gap; a naive per-frame detector cannot.

Frames are saved as PNGs in the output directory plus a ground-truth CSV
(object, frame, x1, y1, x2, y2) for later metrics work.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H, N_FRAMES = 640, 360, 60
rng = np.random.default_rng(7)

OBJECTS = [
    # label, color, w, h, y, x_start, speed(px/frame)
    ("car", (220, 60, 60), 90, 44, 210, -100, 9.0),
    ("truck", (60, 90, 220), 120, 60, 120, 700, -6.0),
    ("pedestrian", (240, 180, 40), 26, 52, 260, 60, 3.5),
]
POLE_X = (330, 356)  # occlusion pole covering the car, frames 30-36
OCCLUDE_FRAMES = range(30, 37)


def make_frame(idx: int):
    frame = np.full((H, W, 3), 38, dtype=np.uint8)  # dark background
    frame[100:300, :, :] = 92  # road band
    frame[196:204, :, :] = 200  # lane line
    truth = []
    for label, color, w, h, y, x0, speed in OBJECTS:
        x = int(x0 + speed * idx)
        if x + w < 0 or x > W:
            continue
        x1, x2 = max(x, 0), min(x + w, W)
        frame[y : y + h, x1:x2] = color
        truth.append((label, idx, x, y, x + w, y + h))
    if idx in OCCLUDE_FRAMES:
        frame[:, POLE_X[0] : POLE_X[1]] = 150  # gray pole occludes whatever is behind it
    noise = rng.normal(0, 7, frame.shape)
    frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    return frame, truth


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_FRAMES):
        frame, truth = make_frame(i)
        Image.fromarray(frame).save(out / f"frame_{i:03d}.png")
        rows.extend(truth)
    with open(out / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "frame", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"wrote {N_FRAMES} frames + ground_truth.csv ({len(rows)} boxes) to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
