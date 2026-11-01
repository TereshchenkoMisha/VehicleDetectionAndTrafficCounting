"""Run the pipeline, or a part of it, and check the results.

    .venv\\Scripts\\python.exe scripts\\run_all.py                                   # all stages, new tag
    .venv\\Scripts\\python.exe scripts\\run_all.py --from tracking --splits dev --device 0
    .venv\\Scripts\\python.exe scripts\\run_all.py --from gtcounts --tag run_20261101_202443
    .venv\\Scripts\\python.exe scripts\\run_all.py --from 5 --to 8 --tag run_20261101_202443 --clean

Stages, selected with --from / --to by number or name:
     1 tests          unit tests
     2 prepare        prepare_dataset (skipped if data/processed/detrac_yolo/images exists)
     3 detectors      evaluate_detectors
     4 tracking       run_tracking
     5 trackeval      evaluate_tracking
     6 gtcounts       count_gt
     7 trackcounts    count_tracks
     8 countingeval   evaluate_counting
     9 notebook       tables and charts (needs jupyter)
    10 verify         verify_run

Needs, before any stage runs (nothing runs if something is missing):
    external inputs of the selected stages: raw dataset, reports/splits.json, reports/ignored_regions.json,
    reports/counting_lines.json, tracker configs, notebooks/results.ipynb; their content is checked too
    (every sequence covered, valid lines)
    results of earlier stages when you start in the middle: pass the --tag of that run
Produces, per tag:
    reports/<tag>/                         detector reports, tracking and counting evaluation, GT and track counts
    outputs/tracks_<tag>/                  MOT tracks and timing.json per configuration
    outputs/reports_<tag>/                 notebook tables and charts
--clean first deletes what the selected stages produce for this tag (and the prepared dataset when stage 2
is selected), after a confirmation. Without it, existing results are never overwritten.
Child processes write straight to the console, so progress bars look as usual.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_ROOT, REPORTS_DIR  # noqa: E402
from src.counting.line_crossing import validate_line  # noqa: E402
from src.data.sequences import find_sequence_dir  # noqa: E402
from src.utils import read_json  # noqa: E402

STAGES: list[str] = ["tests", "prepare", "detectors", "tracking", "trackeval", "gtcounts", "trackcounts",
                     "countingeval", "notebook", "verify"]
DETECTORS: list[str] = ["yolov8n", "yolov8s"]
RAW = "data/raw/UA-DETRAC"
PROCESSED = "data/processed/detrac_yolo"
SPLITS_JSON = "reports/splits.json"
IGNORED_JSON = "reports/ignored_regions.json"
LINES_JSON = "reports/counting_lines.json"

GREEN, RED, YELLOW, CYAN, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"


class Requirement(NamedTuple):
    """One input of one stage: what it is, and which stage (if any) creates it."""
    stage: int      # stage that needs the path
    path: str       # relative to the repository root
    kind: str       # "dir" or "file"
    by: int         # stage that produces it, 0 when no stage does
    why: str


def color(text: str, code: str) -> str:
    """Wrap text in an ANSI color code, but only when writing to a terminal, so redirected logs stay clean."""
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def stage_number(value: str) -> int:
    """Stage number or name -> number, as an argparse type."""
    if value.isdigit() and 1 <= int(value) <= len(STAGES):
        return int(value)
    if value.lower() in STAGES:
        return STAGES.index(value.lower()) + 1
    raise argparse.ArgumentTypeError(f"unknown stage '{value}': use 1-{len(STAGES)} or one of {', '.join(STAGES)}")


class Run:
    """One invocation of the runner.

    Holds the selected stage range and the tag-specific output folders. It checks the inputs of the
    selected stages, optionally cleans their old outputs, and then runs each stage as a child process.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tag: str = args.tag
        self.reports = f"reports/{args.tag}"
        self.tracks = f"outputs/tracks_{args.tag}"
        self.charts = f"outputs/reports_{args.tag}"
        self.device: list[str] = ["--device", args.device] if args.device else []
        self.timings: list[tuple[str, float, int]] = []

    def will_run(self, stage: int) -> bool:
        """True if the stage is inside the --from/--to range and not switched off by a --skip flag."""
        if stage == 1 and self.args.skip_tests or stage == 9 and self.args.skip_notebook:
            return False
        return self.args.first <= stage <= self.args.last

    # ------------------------------------------------------------ input checks

    def requirements(self) -> list[Requirement]:
        """Every input of every stage, as a flat list.

        The list does not depend on which stages are selected. `check_files` later keeps the entries
        whose stage will run and whose producing stage does not run in this invocation.
        """
        r: list[Requirement] = []

        def need(stage: int, path: str, kind: str, by: int, why: str) -> None:
            r.append(Requirement(stage, path, kind, by, why))

        # External inputs (by = 0)
        need(2, f"{RAW}/DETRAC-Images", "dir", 0, "raw frames (README section 2)")
        need(2, f"{RAW}/DETRAC-Train-Annotations-XML", "dir", 0, "raw train annotations")
        need(2, f"{RAW}/DETRAC-Test-Annotations-XML", "dir", 0, "raw test annotations")
        need(2, SPLITS_JSON, "file", 0, "reference split")
        need(3, IGNORED_JSON, "file", 0, f"ignored regions (copied into {PROCESSED} when missing)")
        for stage in (4, 5, 6, 7):
            need(stage, f"{RAW}/DETRAC-Images", "dir", 0, "raw frames")
            need(stage, SPLITS_JSON, "file", 0, "split")
        for stage in (4, 5, 7):
            need(stage, IGNORED_JSON, "file", 0, "ignored regions")
        need(4, "configs/bytetrack.yaml", "file", 0, "tracker config")
        need(4, "configs/botsort.yaml", "file", 0, "tracker config")
        need(6, f"{RAW}/DETRAC-Train-Annotations-XML", "dir", 0, "XML annotations")
        need(6, LINES_JSON, "file", 0, "counting lines")
        need(7, LINES_JSON, "file", 0, "counting lines")
        need(8, SPLITS_JSON, "file", 0, "split")
        need(9, "notebooks/results.ipynb", "file", 0, "notebook")
        need(10, SPLITS_JSON, "file", 0, "split")

        # Results of earlier stages
        need(3, f"{PROCESSED}/images", "dir", 2, "prepared images")
        need(3, f"{PROCESSED}/labels", "dir", 2, "prepared labels")
        for split in self.args.splits:
            for stage in (5, 7, 9, 10):
                need(stage, f"{self.tracks}/{split}", "dir", 4, f"tracks of split {split}")
            for model in DETECTORS:
                for stage in (9, 10):
                    need(stage, f"{self.reports}/detection_{model}_{split}.json", "file", 3, "detector report")
            for stage in (9, 10):
                need(stage, f"{self.reports}/tracking_eval_{split}.json", "file", 5, "tracking evaluation")
            need(10, f"{self.reports}/counting_eval_{split}.json", "file", 8, "counting evaluation")
        need(8, f"{self.reports}/gt_counts.json", "file", 6, "GT counts")
        need(8, f"{self.reports}/track_counts.json", "file", 7, "track counts")
        need(10, f"{self.reports}/gt_counts.json", "file", 6, "GT counts")
        return r

    def check_files(self) -> int:
        """Print whether each needed input exists, and return how many are missing.

        Only inputs of stages that will run are listed. An input that a stage of this same invocation
        produces is skipped, because it does not exist yet and does not need to. That is why starting
        in the middle of the pipeline demands the results of the earlier stages (by --tag), while a full
        run demands only the external files.
        """
        a, missing = self.args, 0
        seen: set[tuple[int, str]] = set()
        print(color(f"\nChecking inputs of stages {a.first}-{a.last} "
                    f"({STAGES[a.first - 1]} .. {STAGES[a.last - 1]}), tag {self.tag}", CYAN))

        for req in self.requirements():
            if not self.will_run(req.stage) or req.by >= a.first or (req.stage, req.path) in seen:
                continue
            seen.add((req.stage, req.path))
            target = ROOT / req.path
            ok = target.is_dir() if req.kind == "dir" else target.is_file()
            origin = ("external" if req.by == 0
                      else f"from stage {req.by} '{STAGES[req.by - 1]}' of an earlier run - pass its --tag")
            print(("  " + color("[OK]     ", GREEN) if ok else "  " + color("[MISSING]", RED))
                  + f" stage {req.stage:<2} {req.path}  - {req.why} ({origin})")
            missing += not ok
        return missing

    def check_content(self) -> int:
        """Check the content of the external inputs for the chosen splits; returns the number of problems.

        An existing file can still lack a sequence or hold an invalid line, which would otherwise fail
        hours into a run. Checked per sequence: counting line, ignored regions, raw frames, XML annotation.
        """
        a, problems = self.args, 0

        def check(ok: bool, name: str) -> None:
            nonlocal problems
            if not ok:
                problems += 1
                print("  " + color("[MISSING]", RED) + f" {name}")

        print(color("\nChecking input content", CYAN))
        try:
            splits = read_json(REPORTS_DIR / "splits.json")
        except (OSError, ValueError) as exc:
            print("  " + color("[MISSING]", RED) + f" splits.json is not readable JSON: {exc}")
            return 1
        check(all(splits.get(s) for s in a.splits), f"splits.json lists sequences for {', '.join(a.splits)}")
        sequences = [seq for s in a.splits for seq in splits.get(s, [])]

        # Counting lines: present and valid for each sequence
        if self.will_run(6) or self.will_run(7):
            try:
                lines = read_json(REPORTS_DIR / "counting_lines.json")
            except (OSError, ValueError):
                lines = {}
            for seq in sequences:
                try:
                    validate_line(lines[seq]["p1"], lines[seq]["p2"])
                except (KeyError, ValueError, TypeError):
                    check(False, f"valid counting line for {seq} in counting_lines.json")

        # Ignored regions
        if any(self.will_run(s) for s in (4, 5, 7)):
            try:
                ignored = read_json(REPORTS_DIR / "ignored_regions.json")
            except (OSError, ValueError):
                ignored = {}
            for seq in sequences:
                check(seq in ignored, f"ignored regions for {seq} in ignored_regions.json")

        # Raw dataset
        if any(self.will_run(s) for s in (2, 4, 5, 6, 7)):
            xml_names = {p.stem for p in Path(RAW_ROOT).rglob("MVI_*.xml")}
            for seq in sequences:
                check(find_sequence_dir(RAW_ROOT, seq) is not None, f"raw frames of {seq}")
                check(seq in xml_names, f"XML annotation of {seq}")

        print("  " + (color("all content checks passed", GREEN) if not problems
                      else color(f"{problems} problem(s)", RED)))
        return problems

    # ------------------------------------------------------------ clean

    def clean_targets(self) -> list[str]:
        """Existing paths that the selected stages would produce for this tag.

        Stage 2 owns the prepared dataset, stage 4 the tracks folder, stage 9 the charts folder; the
        other stages own report files named with the tag. Only paths that exist are returned.
        """
        targets: list[str] = []
        if self.will_run(2):
            targets.append(PROCESSED)
        patterns = {3: [f"{self.reports}/detection_*.json"],
                    5: [f"{self.reports}/tracking_eval_*"],
                    6: [f"{self.reports}/gt_counts.*"],
                    7: [f"{self.reports}/track_counts.json"],
                    8: [f"{self.reports}/counting_eval_*"]}
        for stage, globs in patterns.items():
            if self.will_run(stage):
                for pattern in globs:
                    targets += [str(p.relative_to(ROOT)) for p in ROOT.glob(pattern)]
        if self.will_run(4):
            targets.append(self.tracks)
        if self.will_run(9):
            targets.append(self.charts)
        return [p for p in dict.fromkeys(targets) if (ROOT / p).exists()]

    def clean(self) -> None:
        """Delete the old outputs of the selected stages: list them, ask for confirmation (unless --yes), delete.

        Answering anything but "y" aborts the whole run.
        """
        targets = self.clean_targets()
        if not targets:
            print(color("\n--clean: nothing to delete.", YELLOW))
            return
        print(color("\n--clean will delete:", YELLOW))
        for p in targets:
            print(f"  {p}")
        if not self.args.yes and input("Delete these? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            sys.exit(1)
        for p in targets:
            path = ROOT / p
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        print("Deleted.")

    # ------------------------------------------------------------ execution

    def step(self, name: str, argv: list[str]) -> None:
        """Run `python <argv>` as a child process, record its time, and stop the whole run if it fails.

        The child inherits the console, so progress bars display normally.
        """
        print(color("\n" + "=" * 100 + f"\n>>> {name}", CYAN))
        print("    " + " ".join(["python", *map(str, argv)]))
        print(color("=" * 100, CYAN), flush=True)

        start = time.perf_counter()
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        code = subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, env=env).returncode
        self.timings.append((name, round(time.perf_counter() - start, 1), code))

        if code:
            print(color(f"\nSTEP FAILED (exit code {code}): {name}", RED))
            self.summary()
            sys.exit(code)

    def summary(self) -> None:
        """Print a table of every step that ran: name, seconds, exit code."""
        print("\n{:<95} {:>9} {:>5}".format("Step", "Seconds", "Exit"))
        for name, seconds, code in self.timings:
            print(f"{name:<95} {seconds:>9} {code:>5}")

    def run(self) -> None:
        """Run the selected stages in order, each as one or more child processes.

        Stages that work per split (detectors, tracking) run once per split; the others take all
        splits in one call. All outputs go to tag-specific folders, so nothing existing is overwritten.
        """
        a, s = self.args, self.args.splits

        # Print the interpreter and GPU first: the first thing to look at when a run is slow or fails
        self.step("Environment check", ["-c",
                  "import torch, ultralytics, cv2, platform; print('python', platform.python_version()); "
                  "print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device:', "
                  "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); "
                  "print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)"])

        # Stage 1: unit tests need no data or GPU, so they run first and fail fast
        if self.will_run(1):
            self.step("1. Unit tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])

        # Stage 2: build the YOLO copy of the dataset once; an existing copy is kept unless --clean removed it
        if self.will_run(2):
            if (ROOT / PROCESSED / "images").exists():
                print(color(f"\n2. Prepared dataset already exists in {PROCESSED} - skipping prepare_dataset "
                            "(use --clean to rebuild it).", YELLOW))
            else:
                self.step("2. Prepare dataset (reference split, every 5th frame)",
                          ["scripts/prepare_dataset.py", "--frame-step", "5", "--copy", "--split-file", SPLITS_JSON])

        # Stage 3: detector mAP and precision/recall, one report per model and split
        if self.will_run(3):
            # evaluate_detectors reads the ignored regions from the processed folder
            if not (ROOT / PROCESSED / "ignored_regions.json").exists():
                shutil.copy(ROOT / IGNORED_JSON, ROOT / PROCESSED / "ignored_regions.json")
            for split in s:
                self.step(f"3. Detector evaluation ({split})",
                          ["scripts/evaluate_detectors.py", "--models", "yolov8n.pt", "yolov8s.pt", "--split", split,
                           "--out-dir", self.reports, *self.device])

        # Stage 4: the long one. Detect and track every frame of every configuration, per split.
        if self.will_run(4):
            for split in s:
                self.step(f"4. Detection + tracking, 4 configurations ({split})",
                          ["scripts/run_tracking.py", "--split", split, "--out", self.tracks, *self.device])

        # Stage 5: tracking metrics against GT, computed from the stage 4 tracks
        if self.will_run(5):
            self.step("5. Tracking metrics (HOTA, MOTA, IDF1, IDSW)",
                      ["scripts/evaluate_tracking.py", "--splits", *s, "--tracks-dir", self.tracks,
                       "--out-prefix", f"{self.reports}/tracking_eval"])

        # Stages 6-8: counting. GT counts and track counts use the same line and rules; stage 8 compares them.
        if self.will_run(6):
            self.step("6. Ground-truth line-crossing counts",
                      ["scripts/count_gt.py", "--splits", *s, "--out", f"{self.reports}/gt_counts.json",
                       "--md", f"{self.reports}/gt_counts.md"])

        if self.will_run(7):
            self.step("7. Line-crossing counts from tracks",
                      ["scripts/count_tracks.py", "--splits", *s, "--tracks-dir", self.tracks,
                       "--out", f"{self.reports}/track_counts.json"])

        if self.will_run(8):
            self.step("8. Counting metrics (MAE, WAPE, SCE)",
                      ["scripts/evaluate_counting.py", "--splits", *s, "--gt", f"{self.reports}/gt_counts.json",
                       "--tracks", f"{self.reports}/track_counts.json",
                       "--out-prefix", f"{self.reports}/counting_eval"])

        # Stage 9: tables and charts from the reports, by executing the notebook once per split
        if self.will_run(9):
            if importlib.util.find_spec("nbconvert") is None:
                print(color("\n9. Skipping notebook: jupyter/nbconvert is not installed in this environment "
                            "(uv pip install jupyter --python .venv).", YELLOW))
            else:
                # The notebook reads its input and output folders from these variables
                os.environ.update(REPORTS_DIR=str(ROOT / self.reports), TRACKS_ROOT=str(ROOT / self.tracks),
                                  RESULT_OUTPUT_ROOT=str(ROOT / self.charts))
                for split in s:
                    os.environ["SPLIT"] = split
                    self.step(f"9. Tables and charts ({split})",
                              ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                               "notebooks/results.ipynb", "--output-dir", f"{self.charts}/notebooks",
                               "--output", f"results_{split}.ipynb"])

        # Stage 10: consistency checks and comparison with the reference results
        if self.will_run(10):
            self.step("10. Verify results (consistency checks + comparison with the reference results)",
                      ["scripts/verify_run.py", "--tag", self.tag, "--splits", *s])

        print(color("\n" + "=" * 100 + f"\nFINISHED stages {a.first}-{a.last}. Tag: {self.tag}", GREEN))
        self.summary()
        print(f"Results: {self.reports}   Tracks: {self.tracks}   Charts: {self.charts}")


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for the stages and what they need."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=time.strftime("run_%Y%m%d_%H%M%S"), help="name of this run (default: timestamp)")
    ap.add_argument("--from", dest="first", type=stage_number, default=1, help="first stage (number or name)")
    ap.add_argument("--to", dest="last", type=stage_number, default=len(STAGES), help="last stage (number or name)")
    ap.add_argument("--device", default="", help="e.g. 0 or cpu (default: automatic)")
    ap.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    ap.add_argument("--skip-tests", action="store_true", help="skip stage 1 even if it is in range")
    ap.add_argument("--skip-notebook", action="store_true", help="skip stage 9 even if it is in range")
    ap.add_argument("--clean", action="store_true",
                    help="delete what the selected stages produce for this tag before running (asks first)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation with --clean")
    args = ap.parse_args()
    if args.first > args.last:
        ap.error("--from is after --to")
    return args


def main() -> None:
    """Check the inputs of the selected stages, clean if asked, then run the stages."""
    args = parse_args()
    run = Run(args)
    print(f"Tag: {args.tag}\nRepository: {ROOT}\nPython: {sys.executable}")
    if ".venv" not in Path(sys.executable).parts and ".venv" not in str(Path(sys.prefix)):
        print(color("Warning: not running inside the repository's .venv.", YELLOW))

    # Check everything first: a missing input must not cost a long run, and --clean must not delete for nothing
    missing = run.check_files()
    problems = run.check_content() if not missing else 0
    if missing or problems:
        print(color(f"\n{missing or problems} required input(s) missing or invalid - nothing was run.", RED))
        sys.exit(1)

    if args.clean:
        run.clean()
    run.run()


if __name__ == "__main__":
    main()
