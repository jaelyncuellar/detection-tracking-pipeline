"""Run the detection + tracking pipeline end-to-end.

Usage:
    python src/make_synthetic_sequence.py data/synthetic   # once, makes demo data
    python src/run_pipeline.py data/synthetic out          # demo detector (offline)
    python src/run_pipeline.py data/kitti_sample out --detector yolo   # real frames

Outputs in the out dir:
    annotated.gif  - frames with boxes + persistent track IDs (README demo)
    tracks.csv     - frame, track_id, label, x1, y1, x2, y2 per tracked box
    summary.txt    - track count, lengths, whether IDs survived the occlusion
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from detect import detect_color, detect_yolo, detect_yolo_onnx  # noqa: E402
from tracker import IoUTracker, Track  # noqa: E402

ID_COLORS = [(255, 80, 80), (80, 160, 255), (255, 200, 60), (120, 255, 140), (255, 120, 255)]


def main(frames_dir: str, out_dir: str, detector_name: str = "color") -> None:
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.png")) or sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames found in {frames_dir}")

    Track.reset_ids()
    tracker = IoUTracker(iou_threshold=0.3, max_age=10)
    annotated, rows = [], []
    seen_ids: dict[str, set[int]] = {}

    for idx, path in enumerate(frame_paths):
        frame = np.array(Image.open(path).convert("RGB"))
        if detector_name == "color":
            detections = detect_color(frame)
        elif detector_name == "yolo_onnx":
            detections = detect_yolo_onnx(frame)
        else:
            detections = detect_yolo(frame)
        tracks = tracker.update(detections)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for t in tracks:
            if t.missed > 0:
                continue  # don't draw boxes for frames the detector missed
            x1, y1, x2, y2 = t.box
            color = ID_COLORS[t.id % len(ID_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{t.label} #{t.id}", fill=color)
            rows.append([idx, t.id, t.label, int(x1), int(y1), int(x2), int(y2)])
            seen_ids.setdefault(t.label, set()).add(t.id)
        annotated.append(img)

    annotated[0].save(out_dir / "annotated.gif", save_all=True, append_images=annotated[1:], duration=90, loop=0)
    with open(out_dir / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "label", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    lines = [f"frames processed: {len(frame_paths)}", f"detector: {detector_name}"]
    for label, ids in sorted(seen_ids.items()):
        lines.append(f"{label}: {len(ids)} unique track ID(s) -> {sorted(ids)}")
    car_ids = seen_ids.get("car", set())
    lines.append(
        "occlusion check (frames 30-36, car behind pole): "
        + ("PASS - car kept a single ID through the occlusion" if len(car_ids) == 1 else f"CHECK - car used IDs {sorted(car_ids)}")
    )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"outputs in {out_dir}: annotated.gif, tracks.csv, summary.txt")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    det = sys.argv[sys.argv.index("--detector") + 1] if "--detector" in sys.argv else "color"
    main(args[0] if args else "data/synthetic", args[1] if len(args) > 1 else "out", det)
