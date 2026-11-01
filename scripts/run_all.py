"""Run the pipeline (or a part of it) and check the results. Replaces run_all.ps1 and check_inputs.py.

    .venv\\Scripts\\python.exe scripts\\run_all.py                                   # everything, new tag
    .venv\\Scripts\\python.exe scripts\\run_all.py --from tracking --splits dev --device 0
    .venv\\Scripts\\python.exe scripts\\run_all.py --from gtcounts --tag run_20261101_202443
    .venv\\Scripts\\python.exe scripts\\run_all.py --from 5 --to 8 --tag run_20261101_202443 --clean

Stages (select a range with --from / --to, by number or name):
     1 tests          unit tests
     2 prepare        prepare_dataset (skipped if data/processed/detrac_yolo/images exists)
     3 detectors      evaluate_detectors
     4 tracking       run_tracking (detector x tracker)
     5 trackeval      evaluate_tracking (HOTA, IDF1, IDSW)
     6 gtcounts       count_gt
     7 trackcounts    count_tracks
     8 countingeval   evaluate_counting
     9 notebook       tables and charts (needs jupyter in the environment)
    10 verify         verify_run.py

Before anything runs, the script checks what the selected stages need that is not produced by a stage of this run:
external files (splits, counting lines, ignored regions, raw dataset, tracker configs), their content (all
sequences covered, valid lines), and the results of earlier stages when you start in the middle (pass their --tag).
Nothing runs if something is missing.

Results go to fresh locations: reports/<tag>/, outputs/tracks_<tag>/, outputs/reports_<tag>/. Existing results are
never overwritten, unless you pass --clean: it first deletes what the selected stages produce for this tag (and the
prepared dataset if stage 2 is selected), after asking for confirmation (--yes skips the question).
Child processes write straight to the console, so progress bars are not altered.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_ROOT, REPORTS_DIR  # noqa: E402
from src.counting.line_crossing import validate_line  # noqa: E402
from src.data.sequences import find_sequence_dir  # noqa: E402
from src.utils import read_json  # noqa: E402

STAGES = ["tests", "prepare", "detectors", "tracking", "trackeval", "gtcounts", "trackcounts", "countingeval",
          "notebook", "verify"]
DETECTORS = ["yolov8n", "yolov8s"]
RAW = "data/raw/UA-DETRAC"
PROCESSED = "data/processed/detrac_yolo"
SPLITS_JSON, IGNORED_JSON, LINES_JSON = "reports/splits.json", "reports/ignored_regions.json", "reports/counting_lines.json"

GREEN, RED, YELLOW, CYAN, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"


def color(text, code):
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def stage_number(value):
    if value.isdigit() and 1 <= int(value) <= len(STAGES):
        return int(value)
    if value.lower() in STAGES:
        return STAGES.index(value.lower()) + 1
    raise argparse.ArgumentTypeError(f"unknown stage '{value}': use 1-{len(STAGES)} or one of {', '.join(STAGES)}")


class Run:
    def __init__(self, args):
        self.args = args
        self.tag = args.tag
        self.reports = f"reports/{args.tag}"
        self.tracks = f"outputs/tracks_{args.tag}"
        self.charts = f"outputs/reports_{args.tag}"
        self.device = ["--device", args.device] if args.device else []
        self.timings = []

    def will_run(self, stage):
        if stage == 1 and self.args.skip_tests or stage == 9 and self.args.skip_notebook:
            return False
        return self.args.first <= stage <= self.args.last

    # ------------------------------------------------------------ requirements
    def requirements(self):
        """(stage that needs it, path, 'dir' | 'file', stage that produces it or 0, why)."""
        r, a = [], self.args
        need = lambda *item: r.append(item)   # noqa: E731
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
        need(3, f"{PROCESSED}/images", "dir", 2, "prepared images")
        need(3, f"{PROCESSED}/labels", "dir", 2, "prepared labels")
        for split in a.splits:
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

    def check_files(self):
        """Existence of everything the selected stages need and no selected stage produces."""
        a, missing, seen = self.args, 0, set()
        print(color(f"\nChecking inputs of stages {a.first}-{a.last} "
                    f"({STAGES[a.first - 1]} .. {STAGES[a.last - 1]}), tag {self.tag}", CYAN))
        for stage, path, kind, by, why in self.requirements():
            if not self.will_run(stage) or by >= a.first or (stage, path) in seen:
                continue
            seen.add((stage, path))
            target = ROOT / path
            ok = target.is_dir() if kind == "dir" else target.is_file()
            origin = "external" if by == 0 else f"from stage {by} '{STAGES[by - 1]}' of an earlier run - pass its --tag"
            print(("  " + color("[OK]     ", GREEN) if ok else "  " + color("[MISSING]", RED))
                  + f" stage {stage:<2} {path}  - {why} ({origin})")
            missing += not ok
        return missing

    def check_content(self):
        """Content of the external JSON files: sequences covered, valid lines, raw frames and XML present."""
        a, problems = self.args, 0

        def check(ok, name):
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
        if any(self.will_run(s) for s in (4, 5, 7)):
            try:
                ignored = read_json(REPORTS_DIR / "ignored_regions.json")
            except (OSError, ValueError):
                ignored = {}
            for seq in sequences:
                check(seq in ignored, f"ignored regions for {seq} in ignored_regions.json")
        if any(self.will_run(s) for s in (2, 4, 5, 6, 7)):
            xml_names = {p.stem for p in Path(RAW_ROOT).rglob("MVI_*.xml")}
            for seq in sequences:
                check(find_sequence_dir(RAW_ROOT, seq) is not None, f"raw frames of {seq}")
                check(seq in xml_names, f"XML annotation of {seq}")
        print("  " + (color("all content checks passed", GREEN) if not problems else color(f"{problems} problem(s)", RED)))
        return problems

    # ------------------------------------------------------------ clean
    def clean_targets(self):
        """What the selected stages produce for this tag."""
        t = []
        if self.will_run(2):
            t.append(PROCESSED)
        patterns = {3: [f"{self.reports}/detection_*.json"],
                    5: [f"{self.reports}/tracking_eval_*"],
                    6: [f"{self.reports}/gt_counts.*"],
                    7: [f"{self.reports}/track_counts.json"],
                    8: [f"{self.reports}/counting_eval_*"]}
        for stage, globs in patterns.items():
            if self.will_run(stage):
                for pattern in globs:
                    t += [str(p.relative_to(ROOT)) for p in ROOT.glob(pattern)]
        if self.will_run(4):
            t.append(self.tracks)
        if self.will_run(9):
            t.append(self.charts)
        return [p for p in dict.fromkeys(t) if (ROOT / p).exists()]

    def clean(self):
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
    def step(self, name, argv):
        print(color("\n" + "=" * 100 + f"\n>>> {name}", CYAN))
        print("    " + " ".join(["python"] + [str(x) for x in argv]))
        print(color("=" * 100, CYAN), flush=True)
        start = time.perf_counter()
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        code = subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, env=env).returncode
        self.timings.append((name, round(time.perf_counter() - start, 1), code))
        if code:
            print(color(f"\nSTEP FAILED (exit code {code}): {name}", RED))
            self.summary()
            sys.exit(code)

    def summary(self):
        print("\n{:<95} {:>9} {:>5}".format("Step", "Seconds", "Exit"))
        for name, seconds, code in self.timings:
            print(f"{name:<95} {seconds:>9} {code:>5}")

    def run(self):
        a, s = self.args, self.args.splits
        self.step("Environment check", ["-c",
            "import torch, ultralytics, cv2, platform; print('python', platform.python_version()); "
            "print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device:', "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); "
            "print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)"])

        if self.will_run(1):
            self.step("1. Unit tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
        if self.will_run(2):
            if (ROOT / PROCESSED / "images").exists():
                print(color(f"\n2. Prepared dataset already exists in {PROCESSED} - skipping prepare_dataset "
                            "(use --clean to rebuild it).", YELLOW))
            else:
                self.step("2. Prepare dataset (reference split, every 5th frame)",
                          ["scripts/prepare_dataset.py", "--frame-step", "5", "--copy", "--split-file", SPLITS_JSON])
        if self.will_run(3):
            if not (ROOT / PROCESSED / "ignored_regions.json").exists():
                shutil.copy(ROOT / IGNORED_JSON, ROOT / PROCESSED / "ignored_regions.json")
            for split in s:
                self.step(f"3. Detector evaluation ({split})",
                          ["scripts/evaluate_detectors.py", "--models", "yolov8n.pt", "yolov8s.pt", "--split", split,
                           "--out-dir", self.reports, *self.device])
        if self.will_run(4):
            for split in s:
                self.step(f"4. Detection + tracking, 4 configurations ({split})",
                          ["scripts/run_tracking.py", "--split", split, "--out", self.tracks, *self.device])
        if self.will_run(5):
            self.step("5. Tracking metrics (HOTA, MOTA, IDF1, IDSW)",
                      ["scripts/evaluate_tracking.py", "--splits", *s, "--tracks-dir", self.tracks,
                       "--out-prefix", f"{self.reports}/tracking_eval"])
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
                       "--tracks", f"{self.reports}/track_counts.json", "--out-prefix", f"{self.reports}/counting_eval"])
        if self.will_run(9):
            if importlib.util.find_spec("nbconvert") is None:
                print(color("\n9. Skipping notebook: jupyter/nbconvert is not installed in this environment "
                            "(uv pip install jupyter --python .venv).", YELLOW))
            else:
                os.environ.update(REPORTS_DIR=str(ROOT / self.reports), TRACKS_ROOT=str(ROOT / self.tracks),
                                  RESULT_OUTPUT_ROOT=str(ROOT / self.charts))
                for split in s:
                    os.environ["SPLIT"] = split
                    self.step(f"9. Tables and charts ({split})",
                              ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "notebooks/results.ipynb",
                               "--output-dir", f"{self.charts}/notebooks", "--output", f"results_{split}.ipynb"])
        if self.will_run(10):
            self.step("10. Verify results (consistency checks + comparison with the reference results)",
                      ["scripts/verify_run.py", "--tag", self.tag, "--splits", *s])

        print(color("\n" + "=" * 100 + f"\nFINISHED stages {a.first}-{a.last}. Tag: {self.tag}", GREEN))
        self.summary()
        print(f"Results: {self.reports}   Tracks: {self.tracks}   Charts: {self.charts}")


def main():
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

    run = Run(args)
    print(f"Tag: {args.tag}\nRepository: {ROOT}\nPython: {sys.executable}")
    if ".venv" not in Path(sys.executable).parts and ".venv" not in str(Path(sys.prefix)):
        print(color("Warning: not running inside the repository's .venv.", YELLOW))

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
