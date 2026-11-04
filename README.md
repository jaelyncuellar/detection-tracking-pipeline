# Project 1 — Detection + Tracking Pipeline

Perception Project 1 for the NY robotics end-of-year plan (ship by **Nov 1, 2026**).
**Status Oct 28, 2026: baseline, real-data YOLO pass, Kalman upgrade, and KITTI scorer
all run end-to-end** — Kalman upgrade done Oct 26, scorer built Oct 26, real-data
YOLO11n (ONNX) pass verified Oct 25. KITTI evaluation is wired and pending the
dataset download (see Not done below).

## What it does

Frames in → detections (boxes + labels) → multi-object tracking (persistent IDs) →
annotated demo + per-frame track CSV + a plain-language summary.

Detector backends behind one interface (`src/detect.py`):

| Backend | Use | Needs |
|---|---|---|
| `color` (default) | Synthetic demo sequence — runs offline today, no downloads | numpy, pillow |
| `yolo_onnx` | Real frames (KITTI tracking sample / dashcam) via YOLO11n ONNX — the backend that actually ran on real frames | onnxruntime, `yolo11n.onnx` in project root |
| `yolo` | Real frames via Ultralytics YOLO11n (legacy path, not used for the verified real-data pass) | + ultralytics (torch), weights download on first run |

Two trackers behind the same interface:
- `src/tracker.py` — SORT-style IoU tracker written from scratch: greedy best-IoU
  association, per-label matching, and `max_age` coasting so an ID survives a
  short occlusion instead of respawning as a new object.
- `src/tracker_kalman.py` — constant-velocity Kalman tracker (added Oct 26):
  greedy best-IoU against each track's *predicted* box, plus a ByteTrack-style
  distance-gated second pass for leftover tracks/detections.

## Run it (verified Oct 18, 2026)

```bash
pip install -r requirements.txt
python src/make_synthetic_sequence.py data/synthetic   # 60 demo frames + ground truth
python src/run_pipeline.py data/synthetic out          # demo detector
# Real data, once downloaded:
python src/run_pipeline.py data/kitti_sample out_kitti --detector yolo_onnx
python src/score_kitti.py   # ID-switch count vs KITTI ground truth
```

Verified output on the synthetic sequence (60 frames, 3 objects):

```
car: 1 unique track ID | pedestrian: 1 unique track ID | truck: 1 unique track ID
occlusion check (frames 30–36, car behind pole): PASS — car kept a single ID
```

Outputs: `out/annotated.gif` (the demo), `out/tracks.csv`, `out/summary.txt`.

## Dataset choice — the sensor trade-study (Robotics 101 Lecture 4 tie-in)

Camera frames first because they are the cheapest, highest-information sensor and
public driving datasets (KITTI, nuScenes-mini) ship labeled camera sequences with
ground truth — no hardware, no calibration rig. Known camera failure modes this
project will actually hit: occlusion (handled above via track coasting), motion
blur at speed, and objects leaving the frame. Lidar would fix depth/occlusion
ambiguity but no public mini-dataset keeps Week 1 this small; adding it is a
Project 3 concern, not a Project 1 one.

## Known failure case — fixed Oct 26 (honest, for the README/interviews)

The original IoU tracker has **no motion model**: an object that moves more than
its own width between frames breaks IoU association and gets a new ID. Verified
on a fast-object stress case (50px box moving 90px/frame for 12 frames,
consecutive-frame IoU = 0):

- IoU tracker: **12 unique IDs in 12 frames** (a new ID every frame)
- Kalman tracker (`src/tracker_kalman.py`): **1 unique ID**

The Kalman fix needed a cold-start repair: with velocity starting at 0, a
>1-box-width jump on a track's second frame scores IoU 0 and the filter never
learns the velocity — so leftover tracks/detections get a distance-gated second
pass, like ByteTrack. On the synthetic 60-frame sequence the Kalman tracker is
identical to the IoU tracker (1 ID per object, occlusion PASS) — no regression.
This motion-model material is also exactly Project 2's EKF prep, so the two
projects reinforce each other.

## Week 1 map (from the Phase 1 starter)

- ~~Mon Oct 19: repo created, environment runs, sample sequence loads~~ — **done Oct 18**
- Tue Oct 20: detector on frames, boxes drawn, output saved — **demo backend done; YOLO-on-KITTI is the remaining Tue step**
- Wed Oct 21: OFF (MADD panel)
- Thu Oct 22: tracker added, IDs persist — **done Oct 18 for the demo; re-verify on KITTI**
- Fri–Sun Oct 23–25: README (this file), raw demo clip (`out/annotated.gif`)
- Week 2 (Oct 26–Nov 1): Kalman upgrade — **done Oct 26**; KITTI scorer built — **done Oct 26** (reproduces 0 ID switches on synthetic ground truth); KITTI metrics note (ID switches vs ground truth) — pending KITTI download; final demo, resume v2

## Not done / needs Jaelyn

- Published Nov 4, 2026: https://github.com/jaelyncuellar/detection-tracking-pipeline (pushed at Jaelyn's go; job applications remain a separate step).
- KITTI tracking sample download (free, registration on the KITTI site; sequence 0000, image_02 + label_02 into `data/kitti_sample`) is the one manual step before KITTI scoring runs. Scored on synthetic ground truth so far (0 ID switches, scorer verified); KITTI evaluation wired and pending dataset download — do not claim KITTI metrics until that run lands.
- Real-data pass so far is the YOLO11n ONNX run over a 31-frame traffic clip (`out_yolo/`: detections in every frame, 177 track rows) — a busy 2-second demo clip with no ground truth, so its many track IDs mix genuine entries/exits with real ID switches and are not a scored result.
