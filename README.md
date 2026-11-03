# Vehicle Detection and Traffic Counting

Controlled 2×2 comparison of COCO-pretrained YOLOv8n/YOLOv8s and ByteTrack/BoT-SORT on UA-DETRAC. The task is to detect car/bus/truck and count line crossings. No training or fine-tuning is performed. The question is which component changes counting accuracy more, and at what computational cost.

Team: Mikhail Tereshchenko, Kamil Gainutdinov, Artem Salakhutdinov, Danis Abdulov.

## Verified run: 3 November 2026

The local pipeline results are complete and checked under run_20261103_final for **both dev and test**. The expensive detection/tracking stages had already finished. To complete the result set, the compatible dev outputs were collected under this tag and the GT/counting, charts and verification stages were rerun. Test counting values did not change. No model inference was repeated during this completion.

| Stage | Verified state |
|---|---|
| Fixed split and YOLO conversion | 12 videos, 5 dev / 7 test; 1184 / 1718 prepared images with matching labels |
| Detector evaluation | Both detectors, dev and test |
| Tracking | Four configurations on each split; 5924 dev / 8590 test frames per configuration |
| Tracking evaluation | Class-agnostic TrackEval, HOTA/IDF1/IDSW, both splits |
| GT and predicted line-crossing counts | Both splits, identical counting protocol |
| Counting metrics | MAE/WAPE/SCE for each class and all vehicles, both splits |
| Main results notebooks | Executed dev/test copies and exported tables/charts |
| Verification | Both splits pass --protocol current; 0 failed checks |
| Confidence ablation | Complete 16-condition dev sweep; hashes and saved sources verified |
| Failure analysis | Three reviewed examples in the separate [failure_analysis folder](../failure_analysis/REPORT.md) |

Additional checks: 54 project tests and 17 failure-analysis tests passed; all three source notebooks executed successfully. The failure analysis is intentionally outside this Git repository. Package it separately when submitting the whole project.

Generated datasets, weights, run outputs, execution notebooks and personal environment settings are not stored in Git. This README preserves the compact numerical result; the detailed artifacts listed below are local and reproducible from the code.

## Current protocol

- COCO-pretrained YOLOv8n/s, imgsz=960, NMS IoU=0.7. One tracking NMS implementation: **class-aware**.
- --conf=0.25 sets both tracker high-score and new-track thresholds. The detector keeps scores down to min(track_low_thresh, track_high_thresh), normally 0.1, so low-score association remains available.
- Shared tracker settings: low=0.1, buffer=30, match=0.8, fuse_score=true. BoT-SORT uses sparseOptFlow camera-motion compensation; ReID is disabled.
- Seed=42, deterministic algorithms enabled, cuDNN benchmark and TF32 disabled. Repeatability is checked in a fixed hardware/software environment; wall-clock timing is not deterministic.
- Classes: car→car, van→car, bus→bus, others→truck. others is heterogeneous, which limits interpretation of truck metrics. Motorcycles have no separate quantitative evaluation.
- Ignored-region filtering uses observation centres for GT and predictions. Counting uses the first crossing per ID, maximum frame gap=1 and the most frequent class over the complete saved track. This class rule is offline.
- Split and counting lines are fixed before final test evaluation. Detector evaluation uses every fifth frame; tracking uses all original frames. Splits are by video, never by individual frames.

The original plan described a detector-confidence sweep with fixed tracker thresholds. The implemented ablation instead changes tracker high/new thresholds together while preserving low-score detector input. The notebook labels the recorded protocol explicitly; these two experiments must not be described as the same experiment.

## Current results

Detection metrics are reported once per detector. Counting/tracking metrics are reported for all four configurations. Percentages below are multiplied by 100.

### Detection

| Detector | Dev mAP@0.5, % | Dev mAP@0.5:0.95, % | Test mAP@0.5, % | Test mAP@0.5:0.95, % |
|---|---:|---:|---:|---:|
| yolov8n | 61.23 | 45.16 | 71.77 | 49.54 |
| yolov8s | 64.45 | 49.56 | 76.70 | 55.56 |

### Counting and tracking: dev

| Configuration | MAE | WAPE, % | Mean SCE | IDF1, % | HOTA, % | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n + ByteTrack | 6.40 | 12.75 | -6.40 | 74.80 | 61.10 | 104 |
| YOLOv8n + BoT-SORT | 6.80 | 13.55 | -6.80 | 78.43 | 65.02 | 68 |
| YOLOv8s + ByteTrack | 4.20 | 8.37 | -3.00 | 78.79 | 64.38 | 71 |
| YOLOv8s + BoT-SORT | 4.20 | 8.37 | -3.00 | 81.39 | 68.21 | 56 |

### Counting and tracking: test

| Configuration | MAE | WAPE, % | Mean SCE | IDF1, % | HOTA, % | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n + ByteTrack | 1.71 | 4.80 | 0.86 | 78.66 | 64.51 | 165 |
| YOLOv8n + BoT-SORT | 2.43 | 6.80 | 1.57 | 79.48 | 66.18 | 122 |
| YOLOv8s + ByteTrack | 2.00 | 5.60 | 0.57 | 80.91 | 67.02 | 104 |
| YOLOv8s + BoT-SORT | 1.86 | 5.20 | 1.00 | 82.26 | 69.19 | 76 |

MAE is the mean absolute count error per video. WAPE is sum(|Pred−GT|)/sum(GT); an absent class denominator yields null rather than an invented percentage. Mean SCE is signed Pred−GT: positive indicates overcounting. IDF1/HOTA use TrackEval's dataset-level aggregation; IDSW is summed over videos.

On this test subset, the baseline YOLOv8n + ByteTrack has the lowest counting MAE/WAPE. YOLOv8s + BoT-SORT has the best IDF1/HOTA. The larger detector improves detection mAP but does not uniformly improve counting. A better tracking score also does not guarantee a better count. These results do not establish the original hypothesis of a consistently larger detector benefit across test videos.

### Component timing: test

| Configuration | Detector ms/frame | Tracker ms/frame | Pipeline FPS |
|---|---:|---:|---:|
| YOLOv8n + ByteTrack | 7.25 | 0.87 | 123.1 |
| YOLOv8n + BoT-SORT | 8.92 | 5.36 | 70.0 |
| YOLOv8s + ByteTrack | 10.18 | 0.79 | 91.1 |
| YOLOv8s + BoT-SORT | 10.19 | 5.40 | 64.2 |

Measured on RTX 4060 with Python 3.12.5, torch 2.14.1+cu126, torchvision 0.29.1+cu126, Ultralytics 8.4.198, NumPy 2.5.4 and OpenCV 5.0.0. timing.json records the environment of each actual run. Component times exclude image reading and warm-up. Detector time includes preprocessing, inference, NMS and GPU→CPU; tracker time includes ignored-region filtering and update. These are single-run measurements, not an isolated repeated speed benchmark.

## Setup and data

Run the examples from this repository's root in PowerShell. Python 3.12+ is required. The tested environment is described above; an NVIDIA CUDA build of torch is needed for --device 0.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,notebook]'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
~~~

For CPU execution use --device cpu. Installing the editable package is required so scripts can import src even when invoked by absolute path from another folder. Default data/output paths are rooted at the repository; manually supplied relative CLI paths are relative to the caller's current directory.

UA-DETRAC is downloaded separately, for example from the [dataset mirror used by this project](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig). The expected layout accepts the mirror's nested folders:

~~~text
data/raw/UA-DETRAC/
  DETRAC-Images/DETRAC-Images/MVI_*/img*.jpg
  DETRAC-Train-Annotations-XML/[DETRAC-Train-Annotations-XML/]MVI_*.xml
  DETRAC-Test-Annotations-XML/[DETRAC-Test-Annotations-XML/]MVI_*.xml
~~~

The bracketed annotation folder is optional nesting, not a literal folder name. Keep the committed reports/splits.json, reports/ignored_regions.json and reports/counting_lines.json. The reference selection uses a fixed seed and a day/night brightness proxy; it is not the originally proposed four-weather stratification.

Dev: MVI_20033, MVI_40141, MVI_40171, MVI_40773, MVI_63553.

Test: MVI_20064, MVI_39031, MVI_39051, MVI_39311, MVI_39821, MVI_40792, MVI_40902.

~~~powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --frame-step 5 --copy --split-file reports/splits.json
~~~

Prepared data appears in data/processed/detrac_yolo/. Do not redraw the split after experiments. detrac.yaml is written as UTF-8 so paths containing Cyrillic characters remain readable. Bare weight names are resolved inside models/ and downloaded there when missing.

## Reproduce the complete pipeline

Use a **new tag** for a new code/protocol version:

~~~powershell
$runTag = 'run_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe scripts/run_all.py --tag $runTag --splits dev test --device 0 --to ablation
~~~

The runner executes tests, data preparation, detector evaluation, tracking, TrackEval, GT counting, predicted counting, counting evaluation, both main notebooks and verification. Stage 11 adds the independent dev confidence sweep and its notebook. Prepared data is reused when present. Without --to ablation, the default range ends at verification.

To continue finished tracking without running models again:

~~~powershell
.\.venv\Scripts\python.exe scripts/run_all.py --tag $runTag --from trackeval --to verify --splits dev test --device 0
~~~

If TrackEval is also complete, continue from gtcounts. A resumed sweep requires the same inputs, weights, code, environment and recorded protocol. Changing them requires a new output directory. Avoid --clean for an ordinary new run: it deletes the selected tag's generated products and can also delete shared prepared data if prepare is selected.

### Check the current saved results

~~~powershell
.\.venv\Scripts\python.exe scripts/verify_run.py --tag run_20261103_final --splits dev test --protocol current
.\.venv\Scripts\python.exe scripts/verify_run.py --ablation-dir outputs/ablation_run_20261103_final
~~~

Both commands must exit with code 0. Missing or incomplete inputs are reported; the verifier does not silently substitute another run. Existing finished experiments can be inspected even after non-inference code changes, but cannot be resumed under changed input/code fingerprints.

## Where the current artifacts are

| Path | Contents |
|---|---|
| reports/run_20261103_final/ | Detector, tracking and counting JSON/Markdown for dev and test |
| outputs/tracks_run_20261103_final/<split>/<config>/ | Complete MOT and timing.json for all four configurations |
| outputs/reports_run_20261103_final/<split>/ | Exported tables and charts |
| outputs/reports_run_20261103_final/notebooks/ | Executed results_dev.ipynb and results_test.ipynb |
| outputs/ablation_run_20261103_final/ | Complete dev experiment: manifest, frozen inputs, GT, tracks and reports |
| outputs/ablation_run_20261103_final/analysis/ | Confidence tables, charts and executed ablation notebook |
| outputs/pipeline_review_20261103/ | Completion log, source provenance, before/after checks and executed-source notebook backups |
| ../failure_analysis/outputs/current_dev_paths_checked/ | Three reviewed cases, PNG/MP4, probes, event tables and verification |

The earlier standard dev reports/tracks remain in reports/ and outputs/tracks/dev/. They are identical inputs copied into the complete tagged result; do not mix unrelated tags when assembling comparisons.

## Notebooks and failure analysis

Source notebooks are notebooks/results.ipynb and notebooks/ablation.ipynb. They are kept in Git without executed outputs or personal kernel settings. Run through run_all.py to save executed copies under outputs/. When executing a source notebook interactively, clear outputs before committing.

Both notebooks find the project from its root, notebooks/, or the parent refactor workspace. Relative environment paths are resolved against the project root; absolute paths and ~ work. For an explicit main-run selection, set these before the setup cell:

~~~python
import os
os.environ.update(SPLIT="test", REPORTS_DIR="reports/run_20261103_final",
                  TRACKS_ROOT="outputs/tracks_run_20261103_final")
~~~

Optional variables: DETECTOR_REPORTS_DIR, RESULT_OUTPUT_ROOT and RUN_CONF. For the sweep, use ABLATION_DIR="outputs/ablation_run_20261103_final". Automatic selection requires a unique complete matching set and prints the selected paths and saved NMS. An explicit invalid path is never replaced by an archive.

The separate [failure analysis](../failure_analysis/README.md) compares all four configurations on the same dev frames. Its three reviewed mechanisms are a tracker-dependent gap at the line, van-class ambiguity combined with track-level voting, and concurrent duplicate tracks producing a double count. Automatic missing/extra/class-error candidates are diagnostic outcomes, not population frequencies of causes. Its notebook resolves FAILURE_ANALYSIS_DIR relative to the failure_analysis folder.

## Git policy

Version code, source notebooks, configs/*.yaml, dependency specifications and the three fixed reference JSON files. Ignore virtual environments, personal .env/IDE/Jupyter/agent settings, caches, weights, datasets, generated GT MOT/reports, execution notebooks, logs, videos and root-level backup archives. Portable examples and dependency files remain versioned; .gitignore is not a substitute for checking notebook outputs before a commit.

Generated GT MOT files remain on disk but are removed from Git tracking. Old experiment manifests and numerical reports are not rewritten to hide changes. No commit or push is performed by the verification workflow.
