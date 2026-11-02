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
| Detection + tracking, 4 configurations, timing, on dev | done on dev (run `run_20261102_agnostic`); **test must be re-run** with class-agnostic NMS |
| Results table and plot (`notebooks/results.ipynb` -> `outputs/reports/<split>/`) | done (dev) |
| Ground-truth counts, counting line, line counter | done |
| Counting metrics tables (MAE, WAPE, SCE) | done on dev (run `run_20261102_agnostic`); test numbers are from the earlier run |
| Tracking metrics (TrackEval: HOTA, IDF1, IDSW) | done on dev (run `run_20261102_agnostic`); test must be re-run |
| Code review fixes and migration to `src/` + `scripts/` | done (see `.nocommit/code_review_it-0.md` for open items) |
| Confidence-threshold ablation | code done (`run_ablation`, `notebooks/ablation.ipynb`); not run yet |
| Failure analysis | todo |

## 1. Setup

Python 3.12+ (reference runs on 3.13). All paths are resolved from the repository root, so commands
work from any directory, but `pip install -e .` is needed once so that `scripts/*.py` can import `src`.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .            # add ".[notebook]" for Jupyter, ".[dev]" for pytest, ruff, mypy
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

Environment of the dev results in sections 5-8 (2026-11-02): Python 3.13.16, torch 2.14.1+cu126,
ultralytics 8.4.197, OpenCV 5.0.0, NVIDIA GTX 1070 Ti. An earlier setup (2026-10-19) used Python 3.13.15,
torch 2.11.0+cu128, ultralytics 8.4.182, NVIDIA RTX 5060 Ti. Two GPU notes from that setup:

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
python scripts/prepare_dataset.py --frame-step 5 --copy
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
falls back to the copies in `reports/`; `evaluate_detectors` needs `ignored_regions.json` in
`data/processed/detrac_yolo/` (`prepare_dataset` writes it; copy it there if it is missing):

```bash
cp reports/ignored_regions.json data/processed/detrac_yolo/       # PowerShell: same command (cp is an alias)
```


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
| test | 1718 | 13784 | 1139 | 100 |

The `truck` class is rare (test: two sequences, MVI_40792 and MVI_40902), so per-class truck numbers are noisy.

## 4. Detector evaluation (mAP)

```bash
python scripts/evaluate_detectors.py --models yolov8n.pt yolov8s.pt --split dev --device 0
```

Writes `reports/detection_<model>_<split>.json`. Settings: `imgsz=960`, `conf=0.001` for mAP, P/R at
`conf=0.25`, IoU 0.5. Dev results:

| | YOLOv8n | YOLOv8s |
|---|---|---|
| mAP@0.5 | 0.6123 | 0.6445 |
| mAP@0.5:0.95 | 0.4516 | 0.4956 |
| car P / R @0.25 | 0.825 / 0.763 | 0.827 / 0.825 |
| bus P / R @0.25 | 0.738 / 0.963 | 0.748 / 0.971 |
| truck P / R @0.25 | 0.065 / 0.342 | 0.082 / 0.553 |

(Run `run_20261102_134333`.) mAP is the mean over car, bus and truck, and `truck` is the heterogeneous
`others` class (47 dev boxes; AP@0.5:0.95 0.08 on dev against 0.33 on test), so it is noisy and drags the
mean. Read the car and bus rows separately. Vans are labelled `car`, but COCO detectors often call them
`truck`, which is the main source of truck false positives (precision 0.07).
Detector mAP uses per-class NMS (standard protocol); only the tracking stage uses class-agnostic NMS.

The test split is reserved for the final run (`--split test`).

The `ms_per_frame` field in these reports includes image loading and has no warm-up; do not use it as a speed
measure (it varied 2-3x between runs). Speed comes only from `run_tracking`.

## 5. Detection + tracking (4 configurations)

```bash
python scripts/run_tracking.py --split dev            # all 4 configs on the dev sequences
python scripts/run_tracking.py --split dev --seqs MVI_20033 --max-frames 100 --vis-dir outputs/vis   # quick check
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
  **Camera motion compensation is not the only difference between the two:** in Ultralytics ByteTrack uses a
  Kalman filter over (x, y, aspect, h) and BoT-SORT over (x, y, w, h). A BoT-SORT gain in ID switches or AssA
  therefore cannot be credited to camera motion compensation alone; that needs a run with
  `gmc_method: none` (not done yet).
- NMS before the tracker is **class-agnostic** (`TRACK_AGNOSTIC_NMS` in `src/config.py`). With per-class NMS
  a vehicle can arrive as two boxes of different classes (e.g. car 0.6 and truck 0.3); both pass the tracker's
  thresholds and the second starts a duplicate track. Measured on the run of 2026-11-02, 0.9-1.4% of the output
  rows were such duplicate pairs (IoU > 0.8, different class), and e.g. MVI_39311 had 206 tracks against 142
  without them. `timing.json` records `agnostic_nms`; runs made before this change do not have it and must be
  re-generated.
- `--conf` is the tracker's high-score threshold (`track_high_thresh`, `new_track_thresh`). The detector itself
  runs at `track_low_thresh` (0.1), so the low-score association step of both trackers is active. Runs made
  before this change (detector at `conf` 0.25) had that step switched off and are not comparable.
- Seeds are always set. `--deterministic` additionally makes GPU results repeatable but is slower; do not use it
  for speed measurements. `timing.json` records which one was used.
- A run is written to `<config>.partial/` and renamed when finished, so a crash does not block the next run.
- Timing excludes image reading, a warm-up is run first. Detector time = preprocess + inference + NMS +
  GPU->CPU copy; tracker time = ignored-region mask + `tracker.update`. Compare timings only between runs on
  the same machine.

Output, per configuration, in `outputs/tracks/<split>/<det>_<trk>_conf<conf>/`:

- `<seq>.txt`: MOTChallenge rows `frame,id,x,y,w,h,conf,cls,-1,-1` (frame is 1-based, `cls`: 0 car, 1 bus,
  2 truck, local ids);
- `timing.json`: ms/frame for the detector and the tracker, FPS, per-sequence stats, environment.

**The dev numbers in sections 5, 7 and 8 come from run `run_20261102_agnostic`** (2026-11-02, Ultralytics
8.4.197, GTX 1070 Ti, not deterministic mode; `reports/run_20261102_agnostic/`, tracks in
`outputs/tracks_run_20261102_agnostic/`, git-ignored). That run has the low-score step active and
class-agnostic NMS. **Test numbers have not been re-generated** with class-agnostic NMS; the test tables below
come from the earlier run `run_20261102_134333` and are marked as such. Old runs in `outputs/tracks/` have
the same directory names, so use a new `--out` or delete them.

Speed (single runs on the reference machine, GTX 1070 Ti; dev = 5924 frames; latency =
detector + tracker, ms/frame; FPS = 1000 / latency; image reading and drawing excluded, no repeats, so
differences of a few percent are within run-to-run noise):

| Config | FPS | Total (det / trk) ms |
|---|---|---|
| YOLOv8n + ByteTrack | 109.6 | 9.13 (8.27 / 0.86) |
| YOLOv8n + BoT-SORT | 66.7 | 14.98 (8.66 / 6.32) |
| YOLOv8s + ByteTrack | 68.7 | 14.56 (13.71 / 0.85) |
| YOLOv8s + BoT-SORT | 49.7 | 20.10 (13.80 / 6.31) |

BoT-SORT's tracker step costs about 6.3 ms against 0.9 ms for ByteTrack. Camera motion compensation (sparse
optical flow) is the likely cause, but it has not been isolated (no run with `gmc_method: none`).

### Tracking evaluation (HOTA, MOTA, IDF1, IDSW)

```bash
python scripts/evaluate_tracking.py --splits dev     # then test; writes reports/tracking_eval_<split>.md/json
```

Dev results (TrackEval combination over the 5 dev sequences, run `run_20261102_agnostic`):

| Config | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| yolov8n_bytetrack | 61.48 | 54.88 | 69.29 | 61.65 | 75.46 | 91 |
| yolov8n_botsort | 65.41 | 58.07 | 74.06 | 65.72 | 79.08 | 66 |
| yolov8s_bytetrack | 64.91 | 58.19 | 73.05 | 64.25 | 79.68 | 56 |
| yolov8s_botsort | 68.74 | 61.67 | 77.16 | 69.32 | 82.09 | 48 |

Class-agnostic NMS raised HOTA by 0.4-0.6 and IDF1 by 0.7-0.9 points and removed 5-9% of the false positives
compared with the previous run (one run per configuration, so treat the size of the gain with care).
BoT-SORT is ahead of ByteTrack by 3.9 HOTA points for both detectors, and YOLOv8s ahead of YOLOv8n by
3.3-3.4; see the note on the Kalman filter above before attributing the BoT-SORT gain to camera motion compensation.

GT in MOT format (`reports/gt_mot/<seq>.txt`) is created on the first run and needs the raw dataset.
Metrics are class-agnostic (TrackEval HOTA, CLEAR, Identity), reported as a macro average over sequences and as
a TrackEval combination over the whole split. GT and tracker boxes whose centre is in an ignored region are dropped
before scoring (`--ignored`, default `reports/ignored_regions.json`); a run made with other regions is rejected.

Run `--split test` only after the configurations, `conf` and the counting line are frozen on dev.

## 6. Results table and plot

`notebooks/results.ipynb` builds the comparison table and the plot from `reports/detection_<det>_<SPLIT>.json`
and `outputs/tracks/<SPLIT>/*/timing.json`. It finds the repository root by itself, so it can be run from the
repository root in one command (with the virtual environment activated, so that the `python` kernel is the
project one):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb             # dev (default)
SPLIT=test jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb  # test
```

On Windows PowerShell set the split with `$env:SPLIT = "test"` first. Outputs go to `outputs/reports/<split>/` and carry
the split in the name: `metrics_table_<split>.md`, `metrics_table_<split>.tex`,
`models_comparison_charts_<split>.png`. mAP is a detector property, so it is identical for both trackers of one
detector; "Latency_ms" is detector + tracker time from `run_tracking`.

## 7. Counting line and ground-truth counts

The counting line is placed per sequence to count vehicles crossing it. Lines are stored in
`reports/counting_lines.json` (auto-placed from the GT flow field, then manually corrected).
Each target is counted once, at its first crossing of the line (`src/counting/line_crossing.py`).

### Placing the line

Auto-placement (`scripts/place_lines.py`, default split `dev`) uses the GT trajectories to find the dominant traffic stream
and builds a line perpendicular to it. The resulting preview is in `reports/line_previews/<seq>.png`.
Manual correction is done with `scripts/annotate_line.py`:

```bash
# interactive mode
python scripts/annotate_line.py --sequence MVI_20033

# headless mode (already placed)
python scripts/annotate_line.py --sequence MVI_20033 --p1 140,300 --p2 950,300
```

**Caveat:** lines of the test sequences were also placed and corrected using test GT trajectories, so the
test split is not blind for the counting protocol and absolute test counting errors are optimistic. This does
not affect the comparison of detectors and trackers (they never see the lines), but state it next to any test
number.

All 12 sequences of the reference split have lines now: 10 of them achieve 100% coverage of
potential crossings, 39311 — 97%, 63553 — 84%. Coverage = the share of moving GT trajectories whose
endpoints lie on opposite sides of the line and that are actually counted; measured against the GT
trajectories during placement (scripts/place_lines.py + the coverage check in the session log).

### Ground-truth counts

GT counts (with ignored regions excluded) are computed from the raw XML annotations:

```bash
python scripts/count_gt.py --vis-dir outputs/vis
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
python scripts/count_tracks.py --splits dev
python scripts/count_tracks.py --splits test
```

Output: `reports/track_counts.json`.

Example dev results (yolov8s_bytetrack_conf0.25, run `run_20261102_agnostic`; GT in brackets):
- MVI_20033: car=18 bus=0 truck=1 (all=19, GT 19)
- MVI_40171: car=34 bus=3 truck=0 (all=37, GT 38)
- MVI_63553: car=128 bus=5 truck=0 (all=133, GT 152)

## 8. Counting evaluation (MAE, WAPE, SCE)

Compares the four detector x tracker configurations against GT using
`src/counting/metrics.py`. The GT variant `exclude_ignored` is used (targets with centres in ignored
regions are dropped before tracking, so this is the comparable protocol).

```bash
python scripts/evaluate_counting.py --splits dev
python scripts/evaluate_counting.py --splits test
```

Output:
- `reports/counting_eval_<split>.md` — MAE / WAPE per class, SCE per sequence, and summary.
- `reports/counting_eval_<split>.json` — full numerical results.

Dev results (all classes, run `run_20261102_agnostic`, GT total 251):

| Config | MAE | WAPE | meanSCE |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 7.00 | 13.94% | -7.00 |
| yolov8n_bytetrack_conf0.25 | 6.80 | 13.55% | -6.80 |
| yolov8s_botsort_conf0.25 | 4.20 | 8.37% | -3.80 |
| yolov8s_bytetrack_conf0.25 | 4.20 | 8.37% | -3.80 |

Test results (all classes, GT total 250). **Run `run_20261102_134333`, before class-agnostic NMS;
re-run `--split test` once the configuration is frozen:**

| Config | MAE | WAPE | meanSCE |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 2.43 | 6.80% | +1.57 |
| yolov8n_bytetrack_conf0.25 | 1.71 | 4.80% | +0.86 |
| yolov8s_botsort_conf0.25 | 1.86 | 5.20% | +1.00 |
| yolov8s_bytetrack_conf0.25 | 2.00 | 5.60% | +0.57 |

On dev the counting error did not improve with class-agnostic NMS (previous run: MAE 6.8 / 6.4 / 4.2 / 4.2).
The duplicates it removes rarely cross the counting line, and for YOLOv8s they had been masking part of the
undercount (mean SCE -3.0 before, -3.8 now).

Test MAE is lower than dev MAE mainly because dev contains MVI_63553 (152 of the 251 GT crossings, dense
traffic), which gives most of the dev error. It is not explained by night sequences: dev has 1 of 5, test 1 of 7.

How far these numbers can be trusted:

- With 5 dev and 7 test sequences, the per-sequence differences between configurations are mostly 0 or +-1
  vehicle, and their bootstrap intervals include 0. The four configurations cannot be ranked by counting
  error; do not conclude "detector matters more than tracker" from MAE alone. Tracking metrics (HOTA, IDF1,
  IDSW) separate them much more clearly.
- The error is measured on per-sequence totals, so a double count and a miss in one sequence cancel. On test
  (2026-11-02 run) the sum of |SCE| over 7 sequences is about 12, while matching crossing events in time and
  position gives 14-20 false and 9-10 missed events per configuration. Report event precision/recall next to MAE.
- A crossing is counted per track ID and only if the track has points on both sides of the line, so an ID
  switch next to the line loses the count, and GT has no such gaps. On MVI_63553 most missed crossings (15 of
  19 for YOLOv8s + ByteTrack) have a tracker box on the vehicle but no counted event.
- Per-class rows have 1-5 GT vehicles on dev (truck: 1); their WAPE (hundreds of percent) is noise.

## 9. Reproducibility

Re-run on another machine (RTX 2080 Super, Ultralytics 8.4.176, torch 2.14.1+cu132) from `main`:

- split, ignored regions and all box counts: identical;
- detector mAP/P/R: identical on dev;
- dev tracks (141 858 rows): same frames, rows, classes and track IDs, boxes within 0.27 px (files are not
  byte-identical: GPU numerics; tracking sets seeds but only `--deterministic` makes the GPU path bit-repeatable);
- timings: **not** reproducible across machines (2-3x lower FPS on the laptop GPU, same ordering of the four
  configurations). Compare timings only on one machine.

## Tests and checks

```bash
pytest                                        # or: PYTHONPATH=. python -m unittest discover -s tests
ruff check . && ruff format --check .         # needs ".[dev]"
```

Tests cover line crossing, MOT parsing and validation, ignored-region rule, split quotas, count and detector
metrics, TrackEval combination, and the tracker loop with a stub model (no GPU or dataset needed).

## Confidence ablation

The same four detector + tracker pairs at several confidences on the dev split; confidence is the only factor
that changes. GT counts are made once, then each confidence gets its own tracking runs and evaluations.

Note what "confidence" is here: the detector always runs at `track_low_thresh` (0.1); the swept value is the
tracker's `track_high_thresh` and `new_track_thresh` together. At 0.1 it equals `track_low_thresh`, so the
low-score association step is empty; at 0.4 and 0.6 that step covers a wider score band. The sweep therefore
changes the first-step threshold, the track-birth threshold and the width of the second step at once.

```bash
python scripts/run_ablation.py --device 0 --dry-run    # check the inputs, print the plan
python scripts/run_ablation.py --device 0              # default 0.1 0.25 0.4 0.6, into outputs/ablation/confidence_dev
python scripts/run_ablation.py --device 0 --resume     # continue after an interruption
```

`manifest.json` in the output folder holds the input hashes and the hashes of every finished stage. A resumed
sweep refuses changed inputs, code or outputs. When it is complete, `notebooks/ablation.ipynb` builds the tables
and charts (`ABLATION_DIR` selects another sweep).

## Repository layout

```
pyproject.toml     package (src), dependencies, ruff and pytest settings
configs/           tracker configs (shared thresholds)
src/          library: config, utils, data/, tracking/, counting/, eval/
scripts/           command-line entry points (thin wrappers around src/):
                   prepare_dataset, evaluate_detectors, run_tracking, evaluate_tracking,
                   place_lines, annotate_line, count_gt, count_tracks, evaluate_counting, run_ablation
tests/             unit tests (pytest, or PYTHONPATH=. python -m unittest discover -s tests)
notebooks/         results.ipynb: comparison tables and plots; ablation.ipynb: confidence sweep
reports/           committed results: split, ignored regions, counting lines, metrics
data/ models/      dataset and weights (git-ignored)
outputs/           tracks, videos, notebook output (git-ignored)
```

## Known pitfalls

- Install the package first (`pip install -e .`) so that `scripts/*.py` can import `src`.
- `data/` and `*.pt` are not in git; a fresh clone needs sections 1-3 first.
- `--iou 0.7` (detector NMS) is fixed for all configurations and recorded in `timing.json`.
- Detector mAP uses the COCO protocol with at most 100 detections per frame (measured effect on mAP@0.5/0.75
  versus a limit of 300: none).
