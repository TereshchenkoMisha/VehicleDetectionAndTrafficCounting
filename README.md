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
| Detection + tracking, 4 configurations, timing | done (`run_tracking`) |
| Ground-truth counts, counting line, line counter | todo |
| Tracking metrics (TrackEval: IDF1, IDSW), counting metrics tables | todo |
| Confidence-threshold ablation | todo |
| Failure analysis | todo |

## 1. Setup

Python 3.12+ (developed on 3.13 and 3.14). Run every command **from the repository root**
(`python -m src....` does not work from other directories).

```bash
python -m venv venv
pip install -r requirements.txt
```

`pip install` may pull a CPU-only PyTorch build. With an NVIDIA GPU, reinstall the CUDA build and check it:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Reference environment of the reported runs: Ultralytics 8.4.175, torch 2.14.1+cu126, OpenCV 5.0.0,
NVIDIA GTX 1070 Ti. Pretrained weights (`yolov8n.pt`, `yolov8s.pt`) are downloaded automatically on first
use into `models/` (git-ignored), so the repository root stays clean.

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
Copies of `splits.json` and `ignored_regions.json` are kept in `reports/`; if they are missing from
`data/processed/detrac_yolo/`, copy them there:

```bash
cp reports/splits.json reports/ignored_regions.json data/processed/detrac_yolo/
```

And for Windows:
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

Run `--split test` only after the configurations, `conf` and counting line are frozen on dev.

## Repository layout

```
configs/            tracker configs (shared thresholds)
models/             pretrained weights, downloaded automatically (git-ignored)
src/data/           UA-DETRAC parser, YOLO conversion, split
src/eval/           detector evaluation, counting metrics (MAE, WAPE, SCE)
src/tracking/       detector + tracker loop
src/scripts/        prepare_dataset, evaluate_detectors, run_tracking
reports/            detector metrics, split and ignored-region copies
scripts/            helpers (peek_labels)
```

## Known pitfalls

- Run `python -m src.scripts.*` from the repository root.
- `data/` and `*.pt` are not in git; a fresh clone needs sections 1-3 first.
- Detector mAP uses the COCO protocol with at most 100 detections per frame (measured effect on mAP@0.5/0.75
  versus a limit of 300: none).
