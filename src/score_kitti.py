"""Score tracker output against KITTI tracking ground truth (label_02).

Usage:
    python src/score_kitti.py data/kitti_sample/label_02/0000.txt out_kitti/tracks.csv
    python src/score_kitti.py <label_file> <tracks.csv> --iou 0.5 --out metrics.txt

Method (deliberately simple and honest, CLEAR-MOT style matching):
- Per frame, each ground-truth box is matched to the predicted track box
  with the highest IoU, provided IoU >= threshold and the class maps
  (KITTI Car/Van -> car, Truck -> truck, Pedestrian/Person_sitting ->
  pedestrian, Cyclist -> cyclist). Each predicted box matches at most
  one GT box per frame (greedy, highest IoU first).
- DontCare / Misc / Tram GT rows are skipped.
- An ID switch is counted when a GT track's matched predicted ID differs
  from the predicted ID it was last matched to (consecutive matched
  frames or after a gap — both count, and the report separates them
  into switches_after_gap so the number can be read honestly).
- Also reported: GT tracks, matched frames per GT track (coverage),
  fragmentations (a GT track matched, then unmatched for >=1 frame,
  then matched again), and per-class totals.

Output: printed report, and written to --out path if given.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

CLASS_MAP = {
    "car": "car", "van": "car",
    "truck": "truck",
    "pedestrian": "pedestrian", "person_sitting": "pedestrian",
    "cyclist": "cyclist",
}
SKIP_TYPES = {"dontcare", "misc", "tram"}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_gt(path: Path):
    gt = defaultdict(list)  # frame -> [(gt_id, label, box)]
    tracks = set()
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 10:
                continue
            typ = p[2].lower()
            if typ in SKIP_TYPES or typ not in CLASS_MAP:
                continue
            frame, gt_id = int(p[0]), int(p[1])
            box = tuple(float(x) for x in p[6:10])
            gt[frame].append((gt_id, CLASS_MAP[typ], box))
            tracks.add(gt_id)
    return gt, tracks


def load_pred(path: Path):
    pred = defaultdict(list)  # frame -> [(track_id, label, box)]
    with open(path) as f:
        for r in csv.DictReader(f):
            box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            pred[int(r["frame"])].append((int(r["track_id"]), r["label"].lower(), box))
    return pred


def main(label_path: str, tracks_path: str, iou_thr: float = 0.5, out_path: str | None = None) -> None:
    gt, gt_tracks = load_gt(Path(label_path))
    pred = load_pred(Path(tracks_path))

    last_pred: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    matched_frames: dict[int, int] = defaultdict(int)
    total_frames: dict[int, int] = defaultdict(int)
    switches = switches_after_gap = fragmentations = 0
    per_class = defaultdict(lambda: [0, 0])  # label -> [gt_boxes, matched]
    unmatched_streak: dict[int, bool] = defaultdict(bool)

    for frame in sorted(set(gt) | set(pred)):
        gt_rows = gt.get(frame, [])
        for gt_id, _label, _box in gt_rows:
            total_frames[gt_id] += 1
        # greedy matching, highest IoU first
        pairs = []
        for gi, (gt_id, label, gbox) in enumerate(gt_rows):
            for pi, (pid, plabel, pbox) in enumerate(pred.get(frame, [])):
                if plabel == label:
                    v = iou(gbox, pbox)
                    if v >= iou_thr:
                        pairs.append((v, gi, pi))
        pairs.sort(reverse=True)
        used_g, used_p, match = set(), set(), {}
        for _v, gi, pi in pairs:
            if gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi)
                match[gt_rows[gi][0]] = pred[frame][pi][0]
        for gt_id, label, _box in gt_rows:
            per_class[label][0] += 1
            if gt_id in match:
                per_class[label][1] += 1
                matched_frames[gt_id] += 1
                pid = match[gt_id]
                if gt_id in last_pred and last_pred[gt_id] != pid:
                    if frame == last_frame[gt_id] + 1:
                        switches += 1
                    else:
                        switches_after_gap += 1
                if unmatched_streak[gt_id]:
                    fragmentations += 1
                    unmatched_streak[gt_id] = False
                last_pred[gt_id] = pid
                last_frame[gt_id] = frame
            elif gt_id in last_pred:
                unmatched_streak[gt_id] = True

    lines = [
        f"ground truth: {label_path}",
        f"predictions:  {tracks_path}",
        f"IoU threshold: {iou_thr}",
        f"GT tracks scored: {len(gt_tracks)}",
        f"ID switches (consecutive frames): {switches}",
        f"ID switches (after a gap): {switches_after_gap}",
        f"ID switches (total): {switches + switches_after_gap}",
        f"fragmentations: {fragmentations}",
    ]
    for label in sorted(per_class):
        tot, hit = per_class[label]
        lines.append(f"{label}: {hit}/{tot} GT boxes matched ({100 * hit / tot:.1f}%)")
    for gt_id in sorted(total_frames):
        lines.append(f"GT track {gt_id}: matched {matched_frames[gt_id]}/{total_frames[gt_id]} frames")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if out_path:
        Path(out_path).write_text(report)
        print(f"metrics written to {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    thr = float(sys.argv[sys.argv.index("--iou") + 1]) if "--iou" in sys.argv else 0.5
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if len(args) < 2:
        raise SystemExit(__doc__)
    main(args[0], args[1], thr, out)
