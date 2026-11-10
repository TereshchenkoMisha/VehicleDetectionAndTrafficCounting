# Vehicle Detection and Traffic Counting

## About the project

The project detects vehicles in road videos, tracks them across frames, and counts crossings
of a predefined line separately for cars, buses, and trucks.
The detector locates and classifies each object, the tracker links observations of the same vehicle
into a trajectory with a shared ID, and the counter records the first line crossing for each ID.

The dataset is UA-DETRAC, which provides frame annotations and vehicle IDs.
YOLOv8n and YOLOv8s use COCO-pretrained weights; no training or fine-tuning is performed.
The original `van` class is merged into `car`, and `others` is mapped to `truck`.
Since `others` contains several vehicle types, this mapping has limitations.
Motorcycles are outside the scope of the project.

## Comparisons and checks

The main question is how changing the detector or tracker affects vehicle detection,
identity preservation, and traffic counting, and how the computational cost changes.
The project implements four combinations:

| Detector | ByteTrack | BoT-SORT |
|---|---|---|
| YOLOv8n | Baseline | Tracker change |
| YOLOv8s | Detector change | Both components changed |

Pairwise comparisons isolate one component while keeping the input image size,
counting lines, processing rules, and shared tracker association settings fixed.
BoT-SORT additionally uses camera motion compensation; appearance-based re-identification (ReID) is disabled.

The project also includes:

- A threshold sensitivity experiment that changes the primary association and new-track thresholds together.
  The detector continues to supply low-confidence observations for additional association.
- Comparison of counts from predicted trajectories with counts from the original annotations.
  Both use the same lines and rules for excluding ignored regions.
- Candidate detection for missed crossings, extra crossings, and class errors, with visual comparisons across configurations.
  Explaining the causes requires inspecting the frames and completing a manual review.
- Automated checks for line crossings, annotation conversion, ignored-region handling,
  saved-file integrity, and repeated runs after interruption.

The split is fixed by whole video: dev is used for debugging and choosing settings,
while test is used with frozen settings. Frames from the same video never appear in both splits.

## Project structure

Main source folders and folders created during execution:

```text
VehicleDetectionAndTrafficCounting/
├── pyproject.toml          # Python dependencies and project settings
├── requirements.txt        # compatibility entry for pip; refers to pyproject.toml
├── configs/                # ByteTrack and BoT-SORT settings
├── src/                    # reusable implementation
│   ├── config.py           # paths, classes, and shared parameters
│   ├── data/               # XML/MOT reading, frame discovery, splits, annotation conversion
│   ├── tracking/           # detector and tracker execution, trajectory visualization
│   ├── counting/           # trajectory construction and line-crossing counting
│   ├── eval/               # detection, tracking, and file-integrity checks
│   ├── failure/            # failure discovery, matching, and visualization
│   └── utils.py            # JSON, hashes, environment information, and reproducibility
├── scripts/                # commands for individual stages
├── notebooks/              # tables and figures built from saved files
├── tests/                  # automated tests
├── docs/                   # documentation and project requirements review
├── data/
│   ├── raw/UA-DETRAC/      # downloaded frames and original annotations
│   └── processed/          # prepared images and YOLO labels
├── models/                 # model weights; created when weights are downloaded
├── reports/                # fixed inputs and files for individual runs
└── outputs/                # trajectories, tables, images, videos, and notebook copies
```

The three fixed inputs in `reports/` are `splits.json` for video lists,
`ignored_regions.json` for excluded regions, and `counting_lines.json` for counting lines.
The `failure_review.json` file is added after manually inspecting failure cases.
The `reports/gt_mot/` folder contains ground-truth trajectories converted to MOT format.

Automated stages use a run tag, `TAG`, to keep files from different runs separate:

| Path | Contents |
|---|---|
| `data/processed/detrac_yolo/` | Shared prepared dataset: images, labels, split, and dataset description |
| `reports/TAG/` | Evaluation files, ground-truth and predicted counts, and failure-review material |
| `outputs/tracks_TAG/<split>/<configuration>/` | Per-video MOT trajectories and `timing.json` with parameters, environment details, and runtime |
| `outputs/reports_TAG/` | Exported tables, figures, and executed copies of the main notebook |
| `outputs/ablation_TAG/` | Independent threshold experiment: input snapshots, trajectories, and stage records |
| `outputs/failures_TAG/` | Frames and video clips for visual failure review |
| `outputs/videos_TAG/<split>/`, `outputs/videos_TAG/gt/` | Annotated tracking videos per configuration and sequence, and GT check videos with the counting line; `run_all.py --no-video` skips them |

Source notebooks stay in `notebooks/`; automated execution writes executed copies to `outputs/`.
The dataset, weights, and newly generated files normally remain local under `.gitignore`;
selected saved artifact sets have explicit exceptions in the repository.

The main commands for individual stages are in `scripts/`:

| Script | Purpose |
|---|---|
| `run_all.py` | Main entry point; orchestrates the pipeline stages |
| `prepare_dataset.py` | Reads original annotations and prepares the YOLO dataset |
| `run_tracking.py` | Detects and tracks vehicles for the selected configurations |
| `count_gt.py`, `count_tracks.py` | Counts line crossings in ground-truth and predicted trajectories |
| `evaluate_detectors.py`, `evaluate_tracking.py`, `evaluate_counting.py` | Compares predictions with annotations at the corresponding stages |
| `run_ablation.py` | Runs and resumes the threshold experiment |
| `annotate_line.py`, `place_lines.py` | Defines counting lines |
| `analyze_failures.py`, `render_failure_cases.py`, `report_failures.py` | Finds candidates, displays frames, and produces the manual review report |
| `probe_detector.py` | Replays the detector on frames from a selected case |
| `verify_run.py` | Checks the completeness and consistency of run files |

## Running the project

### 1. Set up with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the commands from the repository root.
Create the project environment and install the dependencies, including tests and Jupyter:

```bash
uv sync --all-extras --python 3.13
```

uv manages the `.venv` directory automatically; manual creation and activation are not required.
Dependencies are read from `pyproject.toml`. The `--all-extras` option includes the `dev` and `notebook` extras.
The generated `uv.lock` records the resolved versions and should be kept in Git.
Use `uv run --all-extras` for the commands below so both extras remain enabled.

### 2. Place the dataset

Download and extract [UA-DETRAC](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig) into `data/raw/UA-DETRAC/`:

```text
data/raw/UA-DETRAC/
  DETRAC-Images/
    DETRAC-Images/MVI_*/img*.jpg
  DETRAC-Train-Annotations-XML/
    MVI_*.xml
  DETRAC-Test-Annotations-XML/
    MVI_*.xml
```

An additional nested folder with the same name is supported for both images and annotations.
Keep the repository's `reports/splits.json`, `reports/ignored_regions.json`, and `reports/counting_lines.json` files.
Model weights are downloaded automatically on first use, which requires internet access.

### 3. Run the pipeline

Check that the input files are available:

```bash
uv run --all-extras python scripts/run_all.py --check
```

Run the complete automated pipeline, including the threshold experiment:

```bash
uv run --all-extras python scripts/run_all.py
```

The device is selected automatically. To select it explicitly:

```bash
uv run --all-extras python scripts/run_all.py --device cpu
uv run --all-extras python scripts/run_all.py --device 0
```

Using `--device 0` requires an NVIDIA GPU and a CUDA-enabled PyTorch installation.
Each ordinary invocation receives a new tag, printed on the `Tag:` line.
Prepared data is validated and reused; incomplete preparation is rebuilt automatically.

### 4. Continue from a selected stage

Replace `YOUR_TAG` with the tag of your run:

```bash
uv run --all-extras python scripts/run_all.py --tag YOUR_TAG --from tracking --to verify
uv run --all-extras python scripts/run_all.py --tag YOUR_TAG --from trackeval --to verify
uv run --all-extras python scripts/run_all.py --tag YOUR_TAG --from ablation
```

The first command reruns tracking; the second continues after tracking.
When tracking is repeated, the previous folders are archived in `outputs/tracks_YOUR_TAG/.previous/` after successful recomputation.
An interrupted threshold experiment resumes when its code, inputs, and environment still match;
if they have changed, use `uv run --all-extras python scripts/run_all.py` with a new tag.
Ordinary reruns do not require `--clean`.

GT video export resumes per file: completed videos are reused, while missing or incomplete videos
are rebuilt from their first frame. A video is written to a temporary `.partial.mp4` and published
only after successful completion. To resume from GT counting with the same tag:

```bash
uv run --all-extras python scripts/run_all.py --tag YOUR_TAG --from gtcounts --to verify
```

### 5. Run failure analysis

After the main pipeline:

```bash
uv run --all-extras python scripts/run_all.py --tag YOUR_TAG --from failures
```

If `reports/failure_review.json` does not exist yet, the command prepares candidates and prints the next steps.
To inspect a candidate, replace the case ID with one from `suggested_cases`:

```bash
uv run --all-extras python -m scripts.render_failure_cases --analysis reports/YOUR_TAG/failure_analysis_dev.json --tracks-dir outputs/tracks_YOUR_TAG --out-dir outputs/failures_YOUR_TAG --case-id CASE_ID --video --replace
```

After inspecting at least three different vehicles, complete `reports/failure_review.json`
using the format in `src/failure/review.py`, then repeat the `failures` stage command.

### 6. Check the code

```bash
uv run --all-extras python -m pytest -q
uv run --all-extras python scripts/run_all.py --help
```
