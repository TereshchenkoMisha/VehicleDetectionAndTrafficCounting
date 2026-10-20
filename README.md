# Vehicle Detection and Traffic Counting

Controlled 2x2 comparison of two detectors (YOLOv8n, YOLOv8s) and two trackers (ByteTrack, BoT-SORT)
on UA-DETRAC, with line-based vehicle counting (classes: car, bus, truck). All models are COCO-pretrained,
no fine-tuning. Question: which component (detector or tracker) has the larger effect on counting accuracy,
and at what computational cost?

Team: Mikhail Tereshchenko, Kamil Gainutdinov, Artem Salakhutdinov, Danis Abdulov.

## Status

| Stage | State |
|---|---|
| Data pipeline (parse, YOLO conversion, split) | done |
| Detector evaluation (mAP, P/R) on dev | done |
| Detection + tracking, 4 configurations, timing, on dev | done (`run_tracking`) |
| Results table and plot (`scripts/results.ipynb` -> `reports/charts/`) | done (dev) |
| Ground-truth counts, counting line, line counter | done |
| Counting metrics tables (MAE, WAPE, SCE) | done (`eval_counting` -> `counting_eval_*.md`) |
| Tracking metrics (TrackEval: IDF1, IDSW) | todo |
| Confidence-threshold ablation | todo |
| Failure analysis | todo |

## 1. Setup

Python 3.12+ (reference runs on 3.13). Run every command **from the repository root**
(`python -m src....` does not work from other directories).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`pip install` may pull a CPU-only PyTorch build. With an NVIDIA GPU, reinstall the CUDA build and check it:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Reference environment of the reported runs: Ultralytics 8.4.175, torch 2.14.1+cu126, Python 3.13.14,
NVIDIA GTX 1070 Ti (OpenCV 5.0.0 was installed, but older `timing.json` files do not record it).
Re-running on Ultralytics 8.4.176 / CUDA 13.2 / RTX 2080 Super reproduced the metrics and tracks up to GPU
numerical noise (see "Reproducibility" below); timings differ strongly between machines. Pretrained weights (`yolov8n.pt`, `yolov8s.pt`) are downloaded automatically on first
use into `models/` (git-ignored), so the repository root stays clean.

Latest run (2026-10-19; the counting results in sections 7-8 come from it): Python 3.13.15,
torch 2.11.0+cu128, ultralytics 8.4.182, OpenCV 5.0.0, NVIDIA RTX 5060 Ti. Two GPU notes from that setup:

- RTX 50xx (Blackwell, sm_120) requires a cu128+ torch build; the `cu126` index above does not support it.
- Windows, fresh venv: if `import torch` fails with `OSError: [WinError 1114] ... c10.dll`, the system
  `msvcp140.dll` is too old. Update the Microsoft Visual C++ Redistributable, or copy a newer
  `msvcp140.dll` (e.g. from `.venv\Lib\site-packages\numpy.libs\`) into
  `.venv\Lib\site-packages\torch\lib\`.

## 2. Dataset

UA-DETRAC is not stored in git (`/data/` is ignored). The official site is no longer available, so we use a
Kaggle mirror (version 2, ~9.5 GB): https://www.kaggle.com/datasets/bratjay/ua-detrac-orig .

Download **DETRAC-Images, DETRAC-Train-Annotations-XML, DETRAC-Test-Annotations-XML, DETRAC-MOT-toolkit**
and unpack them so that the layout is exactly (note the double `DETRAC-Images`, specific to this mirror):

```
data/raw/UA-DETRAC/
  DETRAC-Images/DETRAC-Images/MVI_20011/img00001.jpg ...
  DETRAC-Train-Annotations-XML/MVI_*.xml
  DETRAC-Test-Annotations-XML/MVI_*.xml
  DETRAC-MOT-toolkit/
```

The mirror has no `DETRAC-MetaData` (weather attributes). The weather stratification therefore uses a
brightness proxy: mean brightness of 3 frames per sequence, `< 90/255` means night, else day. So the strata
are **day/night only**, not cloudy/sunny/rainy/night as originally planned.

## 3. Data preparation

```bash
python -m src.scripts.prepare_dataset --frame-step 5 --copy
```

- `--frame-step 5` keeps every 5th frame for the **detector evaluation** store (25 fps, neighbouring
  frames are near-duplicates). Tracking does not use it, see section 5.
- `--copy` copies images instead of symlinking (symlinks need Developer Mode on Windows).
- Never pass `--force` after the split is fixed: it re-draws the sequences. The committed
  `reports/splits.json` is the reference split.

Output in `data/processed/detrac_yolo/`: `images/{dev,test}`, `labels/{dev,test}` (YOLO format),
`detrac.yaml`, `splits.json`, `ignored_regions.json` (absolute xyxy pixels), `stats.json`.
Copies of `splits.json` and `ignored_regions.json` are kept in `reports/`. The split is deterministic
(seed 42), so a fresh `prepare_dataset` run reproduces `reports/splits.json` exactly (checked). `run_tracking`
falls back to the copies in `reports/`; `evaluate_detectors` needs them in `data/processed/detrac_yolo/`,
so copy them there if they are missing:

```bash
cp reports/splits.json reports/ignored_regions.json data/processed/detrac_yolo/
```

And for Windows (PowerShell):
```powershell
cp reports/splits.json data/processed/detrac_yolo/
cp reports/ignored_regions.json data/processed/detrac_yolo/
```

Visual check of the converted GT boxes: `python scripts/peek_labels.py` (writes to `reports/sanity/`).

### Split (fixed, seed 42)

| Split | Sequences |
|---|---|
| dev (5) | MVI_20033, MVI_40141, MVI_40171, MVI_40773 (night), MVI_63553 |
| test (7) | MVI_20064, MVI_39031, MVI_39051, MVI_39311, MVI_39821, MVI_40792 (night), MVI_40902 |

The split is by sequence, never by frame. Dev is used for debugging and tuning (confidence, counting line);
test is used only for the final evaluation.

### Class mapping and ignored regions

- `car` -> car, `van` -> car (COCO has no van), `bus` -> bus, `others` -> truck (heterogeneous, a known
  limitation). No motorcycle class exists in UA-DETRAC and it is out of scope.
- COCO ids used by the detectors: `{2: car, 5: bus, 7: truck}` mapped to local ids `{0, 1, 2}`.
- Detections whose **centre** lies in an UA-DETRAC ignored region are dropped, both in the detector
  evaluation and in tracking, so that all stages are evaluated under the same conditions.

Annotation counts (boxes, from `stats.json`):

| | frames | car | bus | truck |
|---|---|---|---|---|
| dev | 1184 | 7199 | 521 | 47 |
| test | 1704 | 13784 | 1139 | 100 |

The `truck` class is rare (test: two sequences, MVI_40792 and MVI_40902), so per-class truck numbers are noisy.

## 4. Detector evaluation (mAP)

```bash
python -m src.scripts.evaluate_detectors --models yolov8n.pt yolov8s.pt --split dev --device 0
```

Writes `reports/detection_<model>_<split>.json`. Settings: `imgsz=960`, `conf=0.001` for mAP, P/R at
`conf=0.25`, IoU 0.5. Dev results:

| | YOLOv8n | YOLOv8s |
|---|---|---|
| mAP@0.5 | 0.5829 | 0.6132 |
| mAP@0.5:0.95 | 0.4290 | 0.4707 |
| car P / R @0.25 | 0.827 / 0.738 | 0.829 / 0.798 |
| bus P / R @0.25 | 0.744 / 0.914 | 0.755 / 0.921 |
| truck P / R @0.25 | 0.065 / 0.277 | 0.082 / 0.447 |

The test split is reserved for the final run (`--split test`).

The `ms_per_frame` field in these reports includes image loading and has no warm-up; do not use it as a speed
measure (it varied 2-3x between runs). Speed comes only from `run_tracking`.

## 5. Detection + tracking (4 configurations)

```bash
python -m src.scripts.run_tracking --split dev            # all 4 configs on the dev sequences
python -m src.scripts.run_tracking --split dev --seqs MVI_20033 --max-frames 100 --vis-dir outputs/vis   # quick check
```

Runs YOLOv8n/s x ByteTrack/BoT-SORT with a progress bar. Key options: `--conf` (default 0.25), `--imgsz`
(960), `--device`, `--models`, `--trackers`, `--seqs`, `--max-frames`, `--vis-dir` (annotated mp4).

How it works:

- Tracking runs on the **full-rate** frames in `data/raw/.../DETRAC-Images` (not on the every-5th-frame store).
- The detector (`model.predict`) and the tracker (`tracker.update`) are called separately, which gives separate
  ms/frame for each, and lets ignored-region filtering happen before the tracker.
- Tracker settings are in `configs/bytetrack.yaml` and `configs/botsort.yaml`. The association thresholds
  (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`) are
  identical. BoT-SORT keeps its default camera motion compensation, ReID is off, so both trackers are purely
  motion-based. Input size, confidence and counting rules are the same for all four configurations.
- The detector `--conf` filters detections before the tracker. Above `track_low_thresh` (0.1) it switches off
  the low-score association step of both trackers; keep this in mind when interpreting the ablation.
- Timing excludes image reading, a warm-up is run first. Detector time = preprocess + inference + NMS +
  GPU->CPU copy; tracker time = ignored-region mask + `tracker.update`. Compare timings only between runs on
  the same machine.

Output, per configuration, in `outputs/tracks/<split>/<det>_<trk>_conf<conf>/`:

- `<seq>.txt`: MOTChallenge rows `frame,id,x,y,w,h,conf,cls,-1,-1` (frame is 1-based, `cls`: 0 car, 1 bus,
  2 truck, local ids);
- `timing.json`: ms/frame for the detector and the tracker, FPS, per-sequence stats, environment.

Speed (single runs on the reference machine, GTX 1070 Ti; dev = 5924 frames; latency =
detector + tracker, ms/frame; FPS = 1000 / latency):

| Config | FPS | Total (det / trk) ms |
|---|---|---|
| YOLOv8n + ByteTrack | 109.9 | 9.10 (8.27 / 0.83) |
| YOLOv8n + BoT-SORT | 67.2 | 14.87 (8.59 / 6.29) |
| YOLOv8s + ByteTrack | 70.4 | 14.20 (13.39 / 0.81) |
| YOLOv8s + BoT-SORT | 48.1 | 20.79 (14.50 / 6.28) |

The tracker cost of BoT-SORT (about 6 ms vs 0.8 ms) is the camera motion compensation overhead.

Run `--split test` only after the configurations, `conf` and the counting line are frozen on dev.

## 6. Results table and plot

`scripts/results.ipynb` builds the comparison table and the plot from `reports/detection_<det>_<SPLIT>.json`
and `outputs/tracks/<SPLIT>/*/timing.json`. It finds the repository root by itself, so it can be run from the
repository root in one command (with the virtual environment activated, so that the `python` kernel is the
project one):

```bash
jupyter nbconvert --to notebook --execute --inplace scripts/results.ipynb             # dev (default)
SPLIT=test jupyter nbconvert --to notebook --execute --inplace scripts/results.ipynb  # test
```

On Windows PowerShell set the split with `$env:SPLIT = "test"` first. Outputs go to `reports/charts/` and carry
the split in the name: `metrics_table_<split>.md`, `metrics_table_<split>.tex`,
`models_comparison_charts_<split>.png`. mAP is a detector property, so it is identical for both trackers of one
detector; "Latency_ms" is detector + tracker time from `run_tracking`.

## 7. Counting line and ground-truth counts

The counting line is placed per sequence to count vehicles crossing it. Lines are stored in
`reports/counting_lines.json` (auto-placed from the GT flow field, then manually corrected).
Each target is counted once, at its first crossing of the line (src/eval/line_crossing.py).

### Placing the line

Auto-placement (`scripts/place_lines.py`) uses the GT trajectories to find the dominant traffic stream
and builds a line perpendicular to it. The resulting preview is in `reports/line_previews/<seq>.png`.
Manual correction is done with `src/scripts/annotate_line.py`:

```bash
# interactive mode
python -m src.scripts.annotate_line --sequence MVI_20033

# headless mode (already placed)
python -m src.scripts.annotate_line --sequence MVI_20033 --p1 140,300 --p2 950,300
```

All 12 sequences of the reference split have lines now: 10 of them achieve 100% coverage of
potential crossings, 39311 — 97%, 63553 — 84%. Coverage = the share of moving GT trajectories whose
endpoints lie on opposite sides of the line and that are actually counted; measured against the GT
trajectories during placement (scripts/place_lines.py + the coverage check in the session log).

### Ground-truth counts

GT counts (with ignored regions excluded) are computed from the raw XML annotations:

```bash
python -m src.scripts.count_gt_crossings --vis-dir outputs/vis
```

Output:
- `reports/gt_counts.json` — per-sequence counts per class and direction.
- `reports/gt_counts.md` — human-readable tables.
- `outputs/vis/gt_line_*.mp4` — check videos with the counting line, GT boxes, and running counters.

GT totals (exclude_ignored):
- dev: 251 (245 car, 5 bus, 1 truck)
- test: 250 (226 car, 22 bus, 2 truck)
- total: 501

### Counting from tracks

Counts from the saved MOT tracks (same line, same crossing rule):

```bash
python -m src.scripts.count_tracks_crossings --splits dev
python -m src.scripts.count_tracks_crossings --splits test
```

Output: `reports/track_counts.json`.

Example dev results (yolov8s_bytetrack_conf0.25):
- MVI_20033: car=18 bus=0 truck=1 (all=19)
- MVI_40171: car=34 bus=3 truck=0 (all=37)
- MVI_63553: car=132 bus=6 truck=0 (all=138)

## 8. Counting evaluation (MAE, WAPE, SCE)

Compares the four detector x tracker configurations against GT using
src/eval/counting_metrics.py. The GT variant `exclude_ignored` is used (targets with centres in ignored
regions are dropped before tracking, so this is the comparable protocol).

```bash
python -m src.scripts.eval_counting --splits dev
python -m src.scripts.eval_counting --splits test
```

Output:
- `reports/counting_eval_<split>.md` — MAE / WAPE per class, SCE per sequence, and summary.
- `reports/counting_eval_<split>.json` — full numerical results.

Example dev results (all classes):

| Config | MAE | WAPE | meanSCE |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 5.60 | 11.16% | -5.20 |
| yolov8n_bytetrack_conf0.25 | 5.40 | 10.76% | -5.00 |
| yolov8s_botsort_conf0.25 | 4.00 | 7.97% | -2.80 |
| yolov8s_bytetrack_conf0.25 | 3.80 | 7.57% | -2.20 |

Example test results:

| Config | MAE | WAPE | meanSCE |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 1.57 | 4.40% | +1.00 |
| yolov8n_bytetrack_conf0.25 | 1.57 | 4.40% | +1.00 |
| yolov8s_botsort_conf0.25 | 1.86 | 5.20% | +1.00 |
| yolov8s_bytetrack_conf0.25 | 1.57 | 4.40% | +1.00 |

The test results are consistently better (MAE ~1.5–1.9, WAPE ~4–5%), which is expected because the test split
has fewer night sequences and more stable traffic flows.

## 9. Reproducibility

Re-run on another machine (RTX 2080 Super, Ultralytics 8.4.176, torch 2.14.1+cu132) from `main`:

- split, ignored regions and all box counts: identical;
- detector mAP/P/R: identical on dev;
- dev tracks (141 858 rows): same frames, rows, classes and track IDs, boxes within 0.27 px (files are not
  byte-identical: GPU numerics, no seeds are set in tracking);
- timings: **not** reproducible across machines (2-3x lower FPS on the laptop GPU, same ordering of the four
  configurations). Compare timings only on one machine.

## Repository layout

```
configs/            tracker configs (shared thresholds)
models/             pretrained weights, downloaded automatically (git-ignored)
src/data/           UA-DETRAC parser, YOLO conversion, split, brightness proxy
src/eval/           detector evaluation, counting metrics (MAE, WAPE, SCE)
src/tracking/       detector + tracker loop
src/scripts/        prepare_dataset, evaluate_detectors, run_tracking,
                   annotate_line, place_lines, count_gt_crossings,
                   count_tracks_crossings, eval_counting
scripts/            peek_labels.py, results.ipynb
reports/            detector metrics, split and ignored-region copies,
                   counting_lines.json, line_previews/, gt_counts.*,
                   track_counts.json, counting_eval_*.md/json
outputs/tracks/     MOTChallenge tracks + timing.json per split and configuration
outputs/vis/        annotated example videos, GT line check videos
reports/charts/     results tables and plots (generated by the notebook)
```

## Known pitfalls

- Run `python -m src.scripts.*` from the repository root.
- `data/` and `*.pt` are not in git; a fresh clone needs sections 1-3 first.
- `--iou 0.7` (detector NMS) is fixed for all configurations and recorded in `timing.json`.
- Detector mAP uses the COCO protocol with at most 100 detections per frame (measured effect on mAP@0.5/0.75
  versus a limit of 300: none).
