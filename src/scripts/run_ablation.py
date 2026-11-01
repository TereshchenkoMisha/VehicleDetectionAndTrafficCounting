"""Run a confidence sweep on the fixed development split

The existing scripts create tracks and calculate metrics
The analysis notebook reads the saved reports after all conditions finish
"""
import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from ..data.config import RAW_ROOT, SEED
from ..data.detrac_parser import _parse_ann_root
from ..data.frames import list_frames
from ..data.provenance import file_info, write_json
from ..eval.line_crossing import DEFAULT_MAX_FRAME_GAP, PROTOCOL_VERSION
from ..eval.run_validation import validate_tracking_run

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("yolov8n.pt", "yolov8s.pt")
TRACKERS = ("bytetrack", "botsort")


def read_json(path):
    """Read a saved report or a configuration file as text"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_plan(args):
    """Check the inputs and describe every condition before creating output files

    The same development videos and counting rules are used for every confidence
    Input fingerprints help us detect changes when an interrupted run is resumed
    """
    confidences = sorted(args.confidences)
    if len(confidences) < 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in confidences):
        raise ValueError("Provide at least two finite confidence values between 0 and 1")
    if len({f"{value:g}" for value in confidences}) != len(confidences):
        raise ValueError("Confidence values must have unique configuration names")
    if args.imgsz < 1 or not 0 <= args.iou <= 1 or not 0 <= args.seed < 2**31:
        raise ValueError("Invalid image size, NMS IoU or seed")
    raw = (ROOT / args.raw).resolve()
    split_path = ROOT / "reports/splits.json"
    ignored_path = ROOT / "reports/ignored_regions.json"
    lines_path = ROOT / "reports/counting_lines.json"
    splits, ignored, lines = map(read_json, (split_path, ignored_path, lines_path))
    sequences = splits["dev"]
    if not sequences or len(set(sequences)) != len(sequences) or set(sequences) & set(splits["test"]):
        raise ValueError("The development split must be nonempty and disjoint from test")
    if set(sequences) - set(ignored) or set(sequences) - set(lines):
        raise ValueError("Every development video needs ignored regions and a counting line")

    # GT is prepared once because confidence does not change the annotated trajectories
    # Check the XML regions because the GT counter reads them directly from XML
    # Only the selected XML files belong to this experiment
    annotations = {}
    for xml_path in sorted(raw.rglob("MVI_*.xml")):
        if xml_path.stem in sequences:
            element = ET.parse(xml_path).getroot()
            if element.find("frame") is not None:
                ann = _parse_ann_root(element)
                if ann.name in annotations:
                    raise ValueError(f"Duplicate XML annotations for {ann.name}")
                annotations[ann.name] = xml_path
                xml_regions = [list(region.as_xyxy()) for region in ann.ignored_regions]
                if xml_regions != ignored[ann.name]:
                    raise ValueError(f"XML and saved ignored regions differ for {ann.name}")
    if set(annotations) != set(sequences):
        raise ValueError("Some development XML annotations are missing")

    # Record the source frame names and file sizes without loading images into memory
    # Tracking later checks each image when it is read
    frames = {}
    frame_inventory = hashlib.sha256()
    for sequence in sequences:
        source_frames = list_frames(sequence, raw)
        frames[sequence] = len(source_frames)
        for frame_id, image in source_frames:
            info = image.stat()
            frame_inventory.update(f"{sequence}/{frame_id}/{info.st_size}/{info.st_mtime_ns}\n".encode())

    # Keep the source code and tracker settings fixed across all conditions
    # A changed fingerprint requires a new experiment directory
    inputs = [split_path, ignored_path, lines_path, *annotations.values()]
    inputs.extend(ROOT / "configs" / f"{tracker}.yaml" for tracker in TRACKERS)
    inputs.extend(sorted((ROOT / "src").rglob("*.py")))
    fingerprints = {str(p.resolve()): file_info(p)["sha256"] for p in inputs}
    versions = {name: metadata.version(name) for name in
                ("ultralytics", "torch", "numpy", "opencv-python", "trackeval")}
    # Use the same four model and tracker pairs at every confidence
    # The folder names also identify the inputs that the notebook will read
    jobs = []
    for confidence in confidences:
        folder = f"conf{confidence:g}"
        for model in MODELS:
            for tracker in TRACKERS:
                jobs.append({"confidence": confidence, "model": model, "tracker": tracker,
                             "config": f"{Path(model).stem}_{tracker}_conf{confidence:g}",
                             "tracks": f"{folder}/tracks", "reports": f"{folder}/reports"})
    return {"split": "dev", "sequences": sequences, "frames": frames,
            "raw": str(raw), "confidences": confidences, "jobs": jobs,
            "imgsz": args.imgsz, "iou": args.iou, "seed": args.seed, "device": args.device,
            "max_frame_gap": DEFAULT_MAX_FRAME_GAP, "counting_protocol": PROTOCOL_VERSION,
            "input_fingerprints": fingerprints, "frame_inventory": frame_inventory.hexdigest(),
            "software": versions, "python": platform.python_version()}


def verify_inputs(plan):
    """Stop if a configuration or a source file changes during the experiment"""
    for name, expected in plan["input_fingerprints"].items():
        if file_info(name)["sha256"] != expected:
            raise ValueError(f"An experiment input changed: {name}")


def output_records(paths, run_dir):
    """Store relative output paths so the notebook can read a copied experiment"""
    return [{"path": p.relative_to(run_dir).as_posix(), "sha256": file_info(p)["sha256"]}
            for p in paths]


def run_command(module, arguments):
    """Run one existing pipeline command with the active Python interpreter

    A failed child command stops the sweep and keeps its output for inspection
    """
    command = [sys.executable, "-X", "utf8", "-B", "-m", module, *map(str, arguments)]
    print("Running", module, *map(str, arguments), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_stage(key, module, arguments, outputs, run_dir, manifest):
    """Reuse a completed stage only when all of its saved output files still match

    A stage that stopped before completion can be run again
    The manifest is saved before and after each command
    """
    manifest_path = run_dir / "manifest.json"
    verify_inputs(manifest["plan"])
    previous = manifest["stages"].get(key, {})
    if previous.get("status") == "complete":
        if previous["outputs"] != output_records(outputs, run_dir):
            raise ValueError(f"Saved outputs changed for stage {key}")
        print("Already complete", key, flush=True)
        return
    # Record completion only after the child command and all file checks succeed
    # This allows evaluation stages to be repeated after an interruption
    manifest["stages"][key] = {"status": "running"}
    write_json(manifest_path, manifest)
    run_command(module, arguments)
    verify_inputs(manifest["plan"])
    manifest["stages"][key] = {"status": "complete", "outputs": output_records(outputs, run_dir)}
    write_json(manifest_path, manifest)


def validate_condition(directory, job, plan):
    """Check the saved tracks against the full source sequence and fixed settings

    These checks also apply when a completed tracking folder is found after a restart
    """
    timing = validate_tracking_run(directory, plan["sequences"], plan["raw"], "dev")
    expected = {"detector": job["model"], "tracker": job["tracker"], "conf": job["confidence"],
                "imgsz": plan["imgsz"], "iou": plan["iou"], "seed": plan["seed"],
                "deterministic": True}
    if any(timing.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Tracking settings differ from the plan in {directory}")
    provenance = timing.get("provenance", {})
    for key, source in (("tracker_config", ROOT / "configs" / f"{job['tracker']}.yaml"),
                        ("splits", ROOT / "reports/splits.json"),
                        ("ignored", ROOT / "reports/ignored_regions.json")):
        if provenance.get(key, {}).get("sha256") != file_info(source)["sha256"]:
            raise ValueError(f"Tracking input differs for {directory}: {key}")
    weights = provenance.get("weights", {})
    if not weights.get("path") or file_info(weights["path"])["sha256"] != weights.get("sha256"):
        raise ValueError(f"Model weights differ from the saved run in {directory}")
    return timing


def execute_plan(plan, run_dir, manifest):
    """Prepare shared GT once and then evaluate all confidence conditions"""
    # Shared snapshots give every condition the same split and counting rules
    # Existing pipeline commands still read the fixed split from the repository
    shared = run_dir / "shared"
    data_dir = shared / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("splits.json", "ignored_regions.json", "counting_lines.json"):
        shutil.copyfile(ROOT / "reports" / name, data_dir / name)
    gt_counts = shared / "gt_counts.json"
    gt_mot = shared / "gt_mot"
    run_stage("gt_counts", "src.scripts.count_gt_crossings",
              ["--splits", "dev", "--raw", plan["raw"], "--lines", data_dir / "counting_lines.json",
               "--out", gt_counts, "--md", shared / "gt_counts.md",
               "--max-frame-gap", plan["max_frame_gap"]],
              [gt_counts, shared / "gt_counts.md"], run_dir, manifest)
    run_stage("gt_mot", "src.scripts.prepare_gt_mot",
              ["--splits", "dev", "--raw", plan["raw"], "--ignored", data_dir / "ignored_regions.json",
               "--out", gt_mot], [gt_mot / f"{seq}.txt" for seq in plan["sequences"]], run_dir, manifest)

    # Each tracking call creates one detector and tracker condition
    # This lets a resumed sweep skip conditions that have already finished
    environment = manifest.get("tracking_environment")
    for job in plan["jobs"]:
        tracks_root = run_dir / job["tracks"]
        directory = tracks_root / "dev" / job["config"]
        outputs = [directory / "timing.json", *[directory / f"{seq}.txt" for seq in plan["sequences"]]]
        key = "tracks/" + job["config"]
        # A folder can remain after a crash before the manifest records completion
        # Accept it only if the saved tracks pass the full condition checks
        if directory.exists():
            try:
                validate_condition(directory, job, plan)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"Incomplete or incompatible tracks at {directory}\n"
                                 "Move that condition folder aside before resuming") from exc
            if manifest["stages"].get(key, {}).get("status") != "complete":
                manifest["stages"][key] = {"status": "complete", "outputs": output_records(outputs, run_dir)}
        arguments = ["--split", "dev", "--models", job["model"], "--trackers", job["tracker"],
                     "--conf", job["confidence"], "--raw", plan["raw"], "--data", data_dir,
                     "--imgsz", plan["imgsz"], "--iou", plan["iou"], "--seed", plan["seed"],
                     "--out", tracks_root]
        if plan["device"] is not None:
            arguments.extend(["--device", plan["device"]])
        run_stage(key, "src.scripts.run_tracking", arguments, outputs, run_dir, manifest)
        timing = validate_condition(directory, job, plan)
        current = {key: value for key, value in timing["env"].items() if key != "date"}
        if environment is not None and current != environment:
            raise ValueError("Tracking conditions were run in different environments")
        environment = current
        manifest["tracking_environment"] = environment
        write_json(run_dir / "manifest.json", manifest)

    # Recheck every weights file after tracking so replacing weights during the sweep is detected
    for job in plan["jobs"]:
        validate_condition(run_dir / job["tracks"] / "dev" / job["config"], job, plan)

    # Evaluation scripts read every condition in the selected confidence folder
    # Each report therefore contains four configurations with the same confidence
    for confidence in plan["confidences"]:
        folder = run_dir / f"conf{confidence:g}"
        reports = folder / "reports"
        predicted = reports / "track_counts.json"
        run_stage(f"counts/{confidence:g}", "src.scripts.count_tracks_crossings",
                  ["--splits", "dev", "--tracks-dir", folder / "tracks", "--raw", plan["raw"],
                   "--ignored", data_dir / "ignored_regions.json", "--lines", data_dir / "counting_lines.json",
                   "--max-frame-gap", plan["max_frame_gap"], "--out", predicted],
                  [predicted], run_dir, manifest)
        run_stage(f"counting/{confidence:g}", "src.scripts.eval_counting",
                  ["--splits", "dev", "--gt", gt_counts, "--tracks", predicted,
                   "--out-prefix", reports / "counting_eval"],
                  [reports / "counting_eval_dev.json", reports / "counting_eval_dev.md"], run_dir, manifest)
        run_stage(f"tracking/{confidence:g}", "src.scripts.eval_tracking",
                  ["--splits", "dev", "--tracks-dir", folder / "tracks", "--gt-mot-dir", gt_mot,
                   "--raw", plan["raw"], "--ignored", data_dir / "ignored_regions.json",
                   "--out-prefix", reports / "tracking_eval"],
                  [reports / "tracking_eval_dev.json", reports / "tracking_eval_dev.md"], run_dir, manifest)


def main():
    """Create a new experiment or continue a compatible saved experiment"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidences", nargs="+", type=float, default=[0.1, 0.25, 0.4, 0.6])
    parser.add_argument("--out", default="outputs/ablation/confidence_dev")
    parser.add_argument("--raw", default=str(RAW_ROOT))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continue a saved experiment with matching inputs and settings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the inputs and print the plan without creating experiment files")
    args = parser.parse_args()
    try:
        plan = build_plan(args)
        run_dir = (ROOT / args.out).resolve()
        print(f"Development videos: {len(plan['sequences'])}")
        print(f"Confidence values: {plan['confidences']}")
        print(f"Models: {list(MODELS)}")
        print(f"Trackers: {list(TRACKERS)}")
        print(f"Conditions: {len(plan['jobs'])}")
        print(f"Frame visits: {sum(plan['frames'].values()) * len(plan['jobs'])}")
        print(f"Experiment directory: {run_dir}")
        if args.dry_run:
            return
        existing_run = run_dir.exists()
        if existing_run and not args.resume:
            raise ValueError("Output directory exists so use a new directory or request resume")
        if not existing_run:
            run_dir.mkdir(parents=True, exist_ok=False)

        # A small lock prevents two processes from writing the same experiment
        # A lock left after a system crash can be removed after that process has stopped
        lock_path = run_dir / ".running"
        with lock_path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        manifest = None
        try:
            # Read the saved state after acquiring the lock so it is current
            # Reject a different plan before changing an existing manifest
            if existing_run:
                manifest = read_json(run_dir / "manifest.json")
                if manifest.get("version") != 1 or manifest.get("plan") != plan:
                    manifest = None
                    raise ValueError("The saved experiment has different inputs or settings")
            else:
                manifest = {"version": 1, "plan": plan, "status": "pending", "stages": {}}
            manifest["status"] = "running"
            manifest.pop("error", None)
            write_json(run_dir / "manifest.json", manifest)
            execute_plan(plan, run_dir, manifest)
            manifest["status"] = "complete"
            write_json(run_dir / "manifest.json", manifest)
        except (Exception, KeyboardInterrupt) as exc:
            if manifest is not None:
                manifest["status"] = "failed"
                manifest["error"] = str(exc)
                write_json(run_dir / "manifest.json", manifest)
            raise
        finally:
            lock_path.unlink()
    except (OSError, ValueError, KeyError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Ablation stopped: {exc}\n")


if __name__ == "__main__":
    main()
