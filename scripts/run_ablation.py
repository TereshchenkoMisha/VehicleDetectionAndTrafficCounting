"""Confidence sweep on dev, with input snapshots, integrity checks and resumable stages."""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import CONFIG_DIR, IMGSZ, NMS_IOU, RAW_ROOT, REPORTS_DIR, ROOT, SEED, TRACK_AGNOSTIC_NMS
from src.counting.line_crossing import DEFAULT_MAX_FRAME_GAP, PROTOCOL_VERSION, validate_line
from src.data.detrac import _parse_ann_root
from src.data.export import convert_all
from src.data.sequences import list_frames
from src.eval.validation import validate_sweep, validate_tracking_run
from src.tracking.pipeline import detector_conf, tracker_settings
from src.utils import file_info, fraction, positive_int, read_json, seed_value, write_json

MODELS = ("yolov8n.pt", "yolov8s.pt")
TRACKERS = ("bytetrack", "botsort")
METADATA = ("splits.json", "ignored_regions.json", "counting_lines.json")
LIBRARIES = ("ultralytics", "torch", "numpy", "opencv-python", "trackeval")


def fingerprints() -> dict[str, str]:
    """Hash reference metadata, tracker settings and all pipeline code."""
    paths = [REPORTS_DIR / name for name in METADATA] + [CONFIG_DIR / f"{t}.yaml" for t in TRACKERS]
    paths += sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("scripts/*.py"))
    return {p.relative_to(ROOT).as_posix(): file_info(p)["sha256"] for p in paths}


def frame_inventory(raw: str, sequences: list[str]) -> dict[str, Any]:
    """Record source frame IDs, sizes and modification times without decoding images."""
    digest, counts = hashlib.sha256(), {}
    for seq in sequences:
        frames = list_frames(seq, raw)
        counts[seq] = len(frames)
        for fid, path in frames:
            stat = path.stat()
            digest.update(f"{seq}/{fid}/{stat.st_size}/{stat.st_mtime_ns}\n".encode())
    return {"frames": counts, "sha256": digest.hexdigest()}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the dataset and freeze the conditions before creating any outputs."""
    confidences = sorted(args.confidences)
    if len(confidences) < 2 or len({f"{c:g}" for c in confidences}) != len(confidences):
        raise SystemExit("Give at least two confidence values with unique configuration names")
    splits, ignored, lines = [read_json(REPORTS_DIR / name) for name in METADATA]
    sequences = splits["dev"]
    if not sequences or len(set(sequences)) != len(sequences) or set(sequences) & set(splits["test"]):
        raise SystemExit("Dev sequences must be nonempty, unique and disjoint from test")
    for seq in sequences:
        if seq not in ignored or seq not in lines:
            raise SystemExit(f"Missing counting line or ignored regions for {seq}")
        validate_line(lines[seq]["p1"], lines[seq]["p2"])
    raw, annotations = str(Path(args.raw).resolve()), {}
    for xml in sorted(Path(raw).rglob("MVI_*.xml")):
        if xml.stem not in sequences:
            continue
        element = ET.parse(xml).getroot()
        if element.find("frame") is None:
            continue
        ann = _parse_ann_root(element)
        if ann.name != xml.stem or ann.name in annotations:
            raise SystemExit(f"Duplicate or inconsistent XML annotation: {xml}")
        if [list(r.as_xyxy()) for r in ann.ignored_regions] != ignored[ann.name]:
            raise SystemExit(f"XML and saved ignored regions differ for {ann.name}")
        annotations[ann.name] = file_info(xml)
    if set(annotations) != set(sequences):
        raise SystemExit("Some development XML annotations are missing")
    jobs = [{"confidence": c, "model": m, "tracker": t, "config": f"{Path(m).stem}_{t}_conf{c:g}"}
            for c in confidences for m in MODELS for t in TRACKERS]
    return {
        "sequences": sequences,
        "confidences": confidences,
        "jobs": jobs,
        "raw": raw,
        "imgsz": args.imgsz,
        "iou": args.iou,
        "seed": args.seed,
        "device": getattr(args, "device", None),
        "inputs": fingerprints(),
        "annotations": annotations,
        "frame_inventory": frame_inventory(raw, sequences),
        "max_frame_gap": DEFAULT_MAX_FRAME_GAP,
        "counting_protocol": PROTOCOL_VERSION,
        "software": {name: metadata.version(name) for name in LIBRARIES},
        "python": platform.python_version(),
        "inference_protocol": {
            "detector_threshold": "min(low, high)",
            "tracker_thresholds": "high=new=conf",
            "agnostic_nms": TRACK_AGNOSTIC_NMS,
            "deterministic": True,
        },
    }


def run_script(name: str, args: list[Any]) -> None:
    """Run a pipeline command with this interpreter and preserve console progress."""
    print(f"[run] {name} {' '.join(map(str, args))}", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-X", "utf8", str(ROOT / "scripts" / name), *map(str, args)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Stage failed: {name} (exit code {exc.returncode})") from exc


class Sweep:
    """One experiment folder; completed stages are reused only after integrity checks."""

    def __init__(self, out: Path, manifest: dict[str, Any]) -> None:
        self.out, self.manifest, self.plan = out, manifest, manifest["plan"]

    def hashes(self, paths: list[Path]) -> dict[str, str]:
        return {p.relative_to(self.out).as_posix(): file_info(p)["sha256"] for p in paths}

    def check_inputs(self) -> None:
        if fingerprints() != self.plan["inputs"]:
            raise SystemExit("An input or a script changed during the experiment. Use a new --out.")
        for record in self.plan.get("annotations", {}).values():
            if file_info(record["path"])["sha256"] != record["sha256"]:
                raise SystemExit("An XML annotation changed during the experiment")
        if (
            "frame_inventory" in self.plan
            and frame_inventory(self.plan["raw"], self.plan["sequences"]) != self.plan["frame_inventory"]
        ):
            raise SystemExit("Source frames changed during the experiment")
        for record in self.manifest.get("weights", {}).values():
            if file_info(record["path"])["sha256"] != record["sha256"]:
                raise SystemExit("Model weights changed during the experiment")

    def stage(self, key: str, script: str, args: list[Any], outputs: list[Path], reuse: bool = False,
              check: Callable[[], None] | None = None) -> None:
        self.check_inputs()
        done = self.manifest["stages"]
        if key in done:
            if done[key] != self.hashes(outputs):
                raise SystemExit(f"Saved outputs of stage '{key}' changed")
        else:
            if not (reuse and all(p.exists() for p in outputs)):
                run_script(script, args)
            self.check_inputs()
        if check:
            check()
        done[key] = self.hashes(outputs)
        write_json(self.out / "manifest.json", self.manifest)

    def validate(self, job: dict[str, Any]) -> None:
        directory = self.out / f"conf{job['confidence']:g}/tracks/dev" / job["config"]
        try:
            timing = validate_tracking_run(directory, self.plan["sequences"], self.plan["raw"], "dev",
                                           ignored=read_json(REPORTS_DIR / "ignored_regions.json"))
        except (ValueError, OSError) as exc:
            raise SystemExit(f"{directory} is incomplete or inconsistent: {exc}. Move it aside and resume.") from exc
        settings = {
            "detector": job["model"],
            "tracker": job["tracker"],
            "conf": job["confidence"],
            "imgsz": self.plan["imgsz"],
            "iou": self.plan["iou"],
            "seed": self.plan["seed"],
            "deterministic": True,
            "agnostic_nms": TRACK_AGNOSTIC_NMS,
            "detector_conf": detector_conf(tracker_settings(job["tracker"], job["confidence"])),
        }
        if any(timing[key] != value for key, value in settings.items()):
            raise SystemExit(f"Settings of {directory} differ from the plan")
        expected_tracker = tracker_settings(job["tracker"], job["confidence"])
        if timing.get("tracker_settings") != expected_tracker:
            raise SystemExit(f"Tracker thresholds of {directory} differ from the plan")
        used = {"splits": "reports/splits.json", "ignored": "reports/ignored_regions.json",
                "tracker_config": f"configs/{job['tracker']}.yaml"}
        for key, path in used.items():
            recorded = timing["provenance"][key]
            if recorded["sha256"] == self.plan["inputs"][path]:
                continue
            equivalent = False
            if key in ("splits", "ignored"):
                try:
                    source = Path(recorded["path"])
                    equivalent = file_info(source)["sha256"] == recorded["sha256"] and read_json(source) == read_json(
                        ROOT / path
                    )
                except (OSError, ValueError, KeyError):
                    pass
            if not equivalent:
                raise SystemExit(f"{directory} was made from other inputs ({key})")
        weights = timing["provenance"]["weights"]
        if file_info(weights["path"])["sha256"] != weights["sha256"]:
            raise SystemExit("Model weights changed since tracking")
        recorded = self.manifest.setdefault("weights", {}).setdefault(job["model"], weights)
        if recorded["sha256"] != weights["sha256"]:
            raise SystemExit("Conditions used different model weights")
        env = {key: value for key, value in timing["env"].items() if key != "date"}
        if self.manifest.setdefault("tracking_environment", env) != env:
            raise SystemExit("Tracking conditions were run in different environments")

    def run(self, device: str | None) -> None:
        plan, shared = self.plan, self.out / "shared"
        data = shared / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name in METADATA:
            target = data / name
            if target.exists() and target.read_bytes() != (REPORTS_DIR / name).read_bytes():
                raise SystemExit(f"Input snapshot changed: {target}")
            if not target.exists():
                shutil.copyfile(REPORTS_DIR / name, target)
        self.stage("inputs", "", [], [data / name for name in METADATA], reuse=True)
        gt_counts = shared / "gt_counts.json"
        self.stage(
            "gt_counts",
            "count_gt.py",
            [
                "--splits",
                "dev",
                "--raw",
                plan["raw"],
                "--lines",
                data / "counting_lines.json",
                "--out",
                gt_counts,
                "--md",
                shared / "gt_counts.md",
            ],
            [gt_counts, shared / "gt_counts.md"],
        )
        gt_outputs = [shared / "gt_mot" / f"{s}.txt" for s in plan["sequences"]]
        self.check_inputs()
        if "gt_mot" not in self.manifest["stages"]:
            convert_all(plan["sequences"], shared / "gt_mot", plan["raw"], data / "ignored_regions.json")
            self.manifest["stages"]["gt_mot"] = self.hashes(gt_outputs)
        self.stage("gt_mot", "", [], gt_outputs)
        for confidence in plan["confidences"]:
            folder = self.out / f"conf{confidence:g}"
            tracks, reports = folder / "tracks", folder / "reports"
            for job in (j for j in plan["jobs"] if j["confidence"] == confidence):
                directory = tracks / "dev" / job["config"]
                args = [
                    "--split",
                    "dev",
                    "--models",
                    job["model"],
                    "--trackers",
                    job["tracker"],
                    "--conf",
                    confidence,
                    "--raw",
                    plan["raw"],
                    "--data",
                    data,
                    "--imgsz",
                    plan["imgsz"],
                    "--iou",
                    plan["iou"],
                    "--seed",
                    plan["seed"],
                    "--out",
                    tracks,
                ]
                self.stage(f"tracks/{job['config']}", "run_tracking.py", args + (["--device", device] if device else []),
                           [directory / "timing.json", *(directory / f"{s}.txt" for s in plan["sequences"])],
                           reuse=True, check=lambda job=job: self.validate(job))
            counts = reports / "track_counts.json"
            counting, tracking = reports / "counting_eval", reports / "tracking_eval"
            self.stage(
                f"counts/{confidence:g}",
                "count_tracks.py",
                [
                    "--splits",
                    "dev",
                    "--tracks-dir",
                    tracks,
                    "--raw",
                    plan["raw"],
                    "--ignored",
                    data / "ignored_regions.json",
                    "--lines",
                    data / "counting_lines.json",
                    "--out",
                    counts,
                ],
                [counts],
            )
            self.stage(f"counting/{confidence:g}", "evaluate_counting.py",
                       ["--splits", "dev", "--gt", gt_counts, "--tracks", counts, "--out-prefix", counting],
                       [Path(f"{counting}_dev.json"), Path(f"{counting}_dev.md")])
            self.stage(
                f"tracking/{confidence:g}",
                "evaluate_tracking.py",
                [
                    "--splits",
                    "dev",
                    "--tracks-dir",
                    tracks,
                    "--gt-mot-dir",
                    shared / "gt_mot",
                    "--raw",
                    plan["raw"],
                    "--ignored",
                    data / "ignored_regions.json",
                    "--out-prefix",
                    tracking,
                ],
                [Path(f"{tracking}_dev.json"), Path(f"{tracking}_dev.md")],
            )
        self.check_inputs()
        for job in plan["jobs"]:
            self.validate(job)
        self.manifest.update(complete=True, status="complete")
        write_json(self.out / "manifest.json", self.manifest)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confidences", nargs="+", type=fraction, default=[0.1, 0.25, 0.4, 0.6])
    ap.add_argument("--out", default=str(ROOT / "outputs/ablation/confidence_dev"))
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--imgsz", type=positive_int, default=IMGSZ)
    ap.add_argument("--iou", type=fraction, default=NMS_IOU)
    ap.add_argument("--seed", type=seed_value, default=SEED)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    out = Path(args.out).resolve()
    print(
        f"{len(plan['sequences'])} dev sequences, confidences {plan['confidences']}, {len(plan['jobs'])} runs -> {out}"
    )
    if args.dry_run:
        return
    if out.exists() and not args.resume:
        raise SystemExit("Output directory exists. Use a new --out, or --resume.")
    if args.resume and not (out / "manifest.json").is_file():
        raise SystemExit("Nothing to resume")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".running"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise SystemExit(
            f"Experiment is locked: {lock}. Check the owning process before removing a stale lock."
        ) from exc
    with handle:
        handle.write(str(os.getpid()))
    manifest = None
    try:
        if args.resume:
            saved = read_json(out / "manifest.json")
            if saved.get("plan") != plan:
                raise SystemExit("The saved experiment has other inputs or settings. Use a new --out.")
            manifest = saved
        else:
            manifest = {"version": 2, "plan": plan, "stages": {}}
        manifest.update(status="running")
        manifest.pop("complete", None)
        manifest.pop("error", None)
        write_json(out / "manifest.json", manifest)
        Sweep(out, manifest).run(args.device)
        validate_sweep(out)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        if manifest is not None:
            manifest.update(status="failed", error=str(exc))
            manifest.pop("complete", None)
            write_json(out / "manifest.json", manifest)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
