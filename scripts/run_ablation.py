"""Confidence ablation on the dev split: the same detector + tracker pairs at several confidences.

    python scripts/run_ablation.py --confidences 0.1 0.25 0.4 0.6 --device 0

Confidence is the only factor that changes. For each value the script runs the existing scripts:
tracking, track counting, counting evaluation and tracking evaluation. GT counts are made once.
Hashes of the inputs and of every finished stage go into manifest.json, so an interrupted sweep can
be continued with --resume and a changed input or output is detected instead of silently mixed in.

Needs:
    data/raw/UA-DETRAC/, models/*.pt
    reports/splits.json, reports/ignored_regions.json, reports/counting_lines.json
Produces, in <out> (default outputs/ablation/confidence_dev):
    manifest.json                              the plan, input hashes and the hashes of each stage's outputs
    shared/gt_counts.json|md, shared/gt_mot/   GT for all confidences
    conf<c>/tracks/dev/<configuration>/        tracking runs (run_tracking)
    conf<c>/reports/track_counts.json          predicted counts
    conf<c>/reports/counting_eval_dev.*        counting evaluation
    conf<c>/reports/tracking_eval_dev.*        tracking evaluation
The notebook notebooks/ablation.ipynb turns a finished sweep into tables and charts.
"""
import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import CONFIG_DIR, IMGSZ, NMS_IOU, RAW_ROOT, REPORTS_DIR, ROOT, SEED
from src.eval.validation import validate_tracking_run
from src.utils import file_info, fraction, positive_int, read_json, seed_value, write_json

MODELS: tuple[str, ...] = ("yolov8n.pt", "yolov8s.pt")
TRACKERS: tuple[str, ...] = ("bytetrack", "botsort")
METADATA: tuple[str, ...] = ("splits.json", "ignored_regions.json", "counting_lines.json")
LIBRARIES: tuple[str, ...] = ("ultralytics", "torch", "numpy", "opencv-python", "trackeval")


def fingerprints() -> dict[str, str]:
    """SHA-256 of everything a result depends on, keyed by path relative to the repository.

    That is the three metadata files, the tracker configs and all code. Changing any of them while
    a sweep is running would mix results of two setups.
    """
    paths = [REPORTS_DIR / name for name in METADATA] + [CONFIG_DIR / f"{t}.yaml" for t in TRACKERS]
    paths += sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("scripts/*.py"))
    return {p.relative_to(ROOT).as_posix(): file_info(p)["sha256"] for p in paths}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Check the inputs and describe every condition. Nothing is written yet."""
    confidences = sorted(set(args.confidences))
    if len(confidences) < 2:
        raise SystemExit("Give at least two different confidence values")

    # Every dev sequence needs ignored regions and a counting line, or a stage would fail much later
    sequences: list[str] = read_json(REPORTS_DIR / "splits.json")["dev"]
    for name in METADATA[1:]:
        missing = sorted(set(sequences) - set(read_json(REPORTS_DIR / name)))
        if missing:
            raise SystemExit(f"{name} has no entry for {missing}")

    # The name matches the folder run_tracking creates for the same model, tracker and confidence
    jobs = [{"confidence": c, "model": m, "tracker": t, "config": f"{Path(m).stem}_{t}_conf{c:g}"}
            for c in confidences for m in MODELS for t in TRACKERS]
    return {"sequences": sequences, "confidences": confidences, "jobs": jobs, "raw": str(Path(args.raw).resolve()),
            "imgsz": args.imgsz, "iou": args.iou, "seed": args.seed,
            "inputs": fingerprints(), "software": {name: metadata.version(name) for name in LIBRARIES}}


def run_script(name: str, args: list[Any]) -> None:
    """Run scripts/<name> in the current interpreter. Its output goes straight to the console."""
    print(f"[run] {name} {' '.join(map(str, args))}", flush=True)
    command = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / name), *map(str, args)]
    try:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Stage failed: {name} (exit code {exc.returncode})") from exc


class Sweep:
    """One experiment folder: runs the stages in order and records each finished stage."""

    def __init__(self, out: Path, manifest: dict[str, Any]) -> None:
        self.out = out
        self.manifest = manifest
        self.plan: dict[str, Any] = manifest["plan"]

    def hashes(self, paths: list[Path]) -> dict[str, str]:
        """SHA-256 of output files, keyed by path inside the experiment (so the folder can be moved)."""
        return {p.relative_to(self.out).as_posix(): file_info(p)["sha256"] for p in paths}

    def stage(self, key: str, script: str, args: list[Any], outputs: list[Path], reuse: bool = False,
              check: Callable[[], None] | None = None) -> None:
        """Run `script` unless the stage is already recorded, then record the hashes of `outputs`.

        A recorded stage is skipped only if its outputs are unchanged. With `reuse`, outputs found
        on disk are kept instead of recomputed (a finished tracking run that was not recorded yet,
        because the sweep stopped in between); `check` must then accept them.
        """
        if fingerprints() != self.plan["inputs"]:
            raise SystemExit("An input or a script changed during the experiment. Use a new --out.")
        done = self.manifest["stages"]
        if key in done:
            if done[key] != self.hashes(outputs):
                raise SystemExit(f"Saved outputs of stage '{key}' changed")
            return
        if not (reuse and all(p.exists() for p in outputs)):
            run_script(script, args)
        if check:
            check()
        done[key] = self.hashes(outputs)
        write_json(self.out / "manifest.json", self.manifest)

    def validate(self, job: dict[str, Any]) -> None:
        """Check that a tracking run is complete and was made with exactly this plan."""
        directory = self.out / f"conf{job['confidence']:g}/tracks/dev" / job["config"]
        try:
            timing = validate_tracking_run(directory, self.plan["sequences"], self.plan["raw"], "dev",
                                           ignored=read_json(REPORTS_DIR / "ignored_regions.json"))
        except (ValueError, OSError) as exc:
            raise SystemExit(f"{directory} is incomplete or inconsistent: {exc}. Move it aside and resume.") from exc
        settings = {"detector": job["model"], "tracker": job["tracker"], "conf": job["confidence"],
                    "imgsz": self.plan["imgsz"], "iou": self.plan["iou"], "seed": self.plan["seed"]}
        if any(timing[key] != value for key, value in settings.items()):
            raise SystemExit(f"Settings of {directory} differ from the plan. Move it aside and resume.")

        # The run must have used the metadata and tracker config of this plan
        used = {"splits": "reports/splits.json", "ignored": "reports/ignored_regions.json",
                "tracker_config": f"configs/{job['tracker']}.yaml"}
        if any(timing["provenance"][key]["sha256"] != self.plan["inputs"][path] for key, path in used.items()):
            raise SystemExit(f"{directory} was made from other inputs. Move it aside and resume.")

        # Latency is compared between conditions, so all of them need the same machine and libraries
        env = {key: value for key, value in timing["env"].items() if key != "date"}
        if self.manifest.setdefault("tracking_environment", env) != env:
            raise SystemExit("Tracking conditions were run in different environments")

    def run(self, device: str | None) -> None:
        """All stages: GT counts once, then tracking and evaluation for each confidence."""
        plan, shared = self.plan, self.out / "shared"
        gt_counts = shared / "gt_counts.json"
        self.stage("gt_counts", "count_gt.py", ["--splits", "dev", "--raw", plan["raw"], "--out", gt_counts,
                                                "--md", shared / "gt_counts.md"],
                   [gt_counts, shared / "gt_counts.md"])

        for confidence in plan["confidences"]:
            folder = self.out / f"conf{confidence:g}"
            tracks, reports = folder / "tracks", folder / "reports"

            for job in (j for j in plan["jobs"] if j["confidence"] == confidence):
                directory = tracks / "dev" / job["config"]
                args = ["--split", "dev", "--models", job["model"], "--trackers", job["tracker"],
                        "--conf", confidence, "--raw", plan["raw"], "--imgsz", plan["imgsz"], "--iou", plan["iou"],
                        "--seed", plan["seed"], "--out", tracks]
                self.stage(f"tracks/{job['config']}", "run_tracking.py", args + (["--device", device] if device else []),
                           [directory / "timing.json", *(directory / f"{s}.txt" for s in plan["sequences"])],
                           reuse=True, check=lambda job=job: self.validate(job))

            # The evaluation scripts read every configuration in the folder: one report per confidence
            counts, counting, tracking = reports / "track_counts.json", reports / "counting_eval", reports / "tracking_eval"
            self.stage(f"counts/{confidence:g}", "count_tracks.py",
                       ["--splits", "dev", "--tracks-dir", tracks, "--raw", plan["raw"], "--out", counts], [counts])
            self.stage(f"counting/{confidence:g}", "evaluate_counting.py",
                       ["--splits", "dev", "--gt", gt_counts, "--tracks", counts, "--out-prefix", counting],
                       [Path(f"{counting}_dev.json"), Path(f"{counting}_dev.md")])
            self.stage(f"tracking/{confidence:g}", "evaluate_tracking.py",
                       ["--splits", "dev", "--tracks-dir", tracks, "--gt-mot-dir", shared / "gt_mot",
                        "--raw", plan["raw"], "--out-prefix", tracking],
                       [Path(f"{tracking}_dev.json"), Path(f"{tracking}_dev.md")])

        self.manifest["complete"] = True
        write_json(self.out / "manifest.json", self.manifest)


def parse_args() -> argparse.Namespace:
    """Command-line options; see the module docstring for what the script reads and writes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confidences", nargs="+", type=fraction, default=[0.1, 0.25, 0.4, 0.6])
    ap.add_argument("--out", default=str(ROOT / "outputs/ablation/confidence_dev"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--iou", type=fraction, default=NMS_IOU, help="detector NMS IoU")
    ap.add_argument("--seed", type=seed_value, default=SEED)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--resume", action="store_true", help="continue a saved experiment with the same inputs")
    ap.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan, write nothing")
    return ap.parse_args()


def main() -> None:
    """Create a new experiment, or continue a saved one with --resume."""
    args = parse_args()
    plan = build_plan(args)
    out = Path(args.out).resolve()
    print(f"{len(plan['sequences'])} dev sequences, confidences {plan['confidences']}, {len(plan['jobs'])} "
          f"tracking runs -> {out}")
    if args.dry_run:
        return

    # A new experiment needs a new folder. A resumed one must have been made from the same plan.
    manifest_path = out / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f"Nothing to resume: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest["plan"] != plan:
            raise SystemExit("The saved experiment has other inputs or settings. Use a new --out.")
    elif out.exists():
        raise SystemExit(f"{out} exists. Use a new --out, or --resume to continue it.")
    else:
        manifest = {"plan": plan, "stages": {}}
        write_json(manifest_path, manifest)
    Sweep(out, manifest).run(args.device)


if __name__ == "__main__":
    main()
