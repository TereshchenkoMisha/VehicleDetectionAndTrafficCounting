"""Tests for the ablation controller and notebook on a one-sequence project with simulated tracking.

Tests (all in AblationTests; tracking is simulated, every other script runs for real):
    test_equivalent_metadata_and_changed_copies   a re-formatted copy of splits.json passes; changed bytes or
                                                  a different split are refused
    test_dataset_and_weights_changes_are_detected a changed XML, frame, weights or ignored region is refused
    test_lock_and_legacy_manifest                 a locked sweep cannot resume; an old manifest format still validates
    test_sweep_resume_and_changed_settings        a finished sweep resumes without work; other settings are refused
    test_dry_run_and_invalid_confidences          `--dry-run` writes nothing; duplicate confidences are refused
    test_edited_outputs_and_inputs_are_refused    edited tracks or a changed counting line stop a resume
    test_resume_after_a_failed_stage_keeps_finished_tracking   a failed stage is resumed without new tracking
    test_notebook_builds_tables_and_rejects_changed_outputs    the notebook makes tables and charts, and checks hashes
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import matplotlib
from PIL import Image

from scripts import run_ablation
from src.config import ROOT, TRACK_AGNOSTIC_NMS
from src.eval.validation import validate_sweep
from src.tracking.pipeline import detector_conf, tracker_settings
from src.utils import file_info, read_json, write_json

CAR_POSITIONS: tuple[int, ...] = (15, 35, 55)       # left edge of the car in frames 1-3; it crosses x=40 in frame 2
EXPECTED_JOBS: int = 8          # 2 confidences x 2 detectors x 2 trackers


class AblationTests(unittest.TestCase):
    """The ablation controller in a temporary copy of the project."""

    def setUp(self) -> None:
        """Copy the code and build one dev sequence MVI_1 in which a car crosses the counting line at frame 2."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        for folder in ("src", "scripts", "configs", "notebooks"):
            shutil.copytree(ROOT / folder, self.project / folder, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "pyproject.toml", self.project / "pyproject.toml")

        # Fixed inputs of the experiment
        reports = self.project / "reports"
        write_json(reports / "splits.json", {"dev": ["MVI_1"], "test": []})
        write_json(reports / "ignored_regions.json", {"MVI_1": []})
        write_json(reports / "counting_lines.json", {"MVI_1": {"p1": [40, 0], "p2": [40, 39]}})

        # Raw dataset: three frames and the XML annotation of the moving car
        self.raw = self.project / "raw"
        (self.raw / "frames/MVI_1").mkdir(parents=True)
        (self.raw / "xml").mkdir()
        targets = []
        for frame, left in enumerate(CAR_POSITIONS, 1):
            Image.new("RGB", (80, 40)).save(self.raw / f"frames/MVI_1/img{frame:05d}.jpg")
            targets.append(f'<frame num="{frame}"><target_list><target id="1"><box left="{left}" top="15" '
                           'width="10" height="10"/><attribute vehicle_type="car"/></target></target_list></frame>')
        (self.raw / "xml/MVI_1.xml").write_text(
            '<sequence name="MVI_1"><ignored_region/>' + "".join(targets) + "</sequence>", encoding="utf-8")

        # Weights are only hashed, never loaded, so any bytes will do
        (self.project / "models").mkdir()
        for name in run_ablation.MODELS:
            (self.project / "models" / name).write_bytes(b"fixture weights")

        self.out = self.project / "outputs/ablation"
        self.calls: list[str] = []
        self.fail_once_on: str | None = None        # script name that fails on its next call
        self.real_run_script = run_ablation.run_script

    def project_paths(self) -> Any:
        """Patch the module paths of `run_ablation` so that it works inside the temporary project."""
        return patch.multiple(run_ablation, ROOT=self.project, REPORTS_DIR=self.project / "reports",
                              CONFIG_DIR=self.project / "configs")

    def plan_args(self) -> SimpleNamespace:
        """Arguments of `build_plan` for the confidences 0.1 and 0.6."""
        return SimpleNamespace(confidences=[0.1, 0.6], raw=self.raw, imgsz=960, iou=0.7, seed=42)

    def tracking_args(self, model: str, tracker: str, conf: float, out: Path) -> list[str]:
        """Command-line arguments with which the sweep calls `run_tracking.py`."""
        return ["--models", model, "--trackers", tracker, "--conf", str(conf), "--out", str(out),
                "--imgsz", "960", "--iou", "0.7", "--seed", "42"]

    def fake_run_script(self, name: str, args: list[str]) -> Any:
        """Run the real script, except tracking, which is simulated: confidence 0.6 loses the last detection."""
        self.calls.append(name)
        if name == self.fail_once_on:
            self.fail_once_on = None
            raise SystemExit("Stage failed")
        if name != "run_tracking.py":
            return self.real_run_script(name, args)

        # Read the arguments the sweep passed, as the real script would
        values = dict(zip(map(str, args[::2]), map(str, args[1::2]), strict=True))
        model, tracker, conf = values["--models"], values["--trackers"], float(values["--conf"])
        directory = Path(values["--out"]) / "dev" / f"{Path(model).stem}_{tracker}_conf{conf:g}"
        directory.mkdir(parents=True)

        # The simulated track follows the car, and stops one frame early at confidence 0.6
        positions = CAR_POSITIONS if conf < 0.6 else CAR_POSITIONS[:2]
        mot = directory / "MVI_1.txt"
        mot.write_text("".join(f"{i},1,{x},15,10,10,0.9,0\n" for i, x in enumerate(positions, 1)), encoding="utf-8")

        # The timing report with the provenance that the sweep validates
        reports = Path(values.get("--data", self.project / "reports"))
        write_json(
            directory / "timing.json",
            {
                "detector": model,
                "tracker": tracker,
                "conf": conf,
                "split": "dev",
                "frames": 3,
                "deterministic": True,
                "agnostic_nms": TRACK_AGNOSTIC_NMS,
                "detector_conf": detector_conf(tracker_settings(tracker, conf)),
                "tracker_settings": tracker_settings(tracker, conf),
                "imgsz": int(values["--imgsz"]),
                "iou": float(values["--iou"]),
                "seed": int(values["--seed"]),
                "per_seq": {"MVI_1": {"frames": 3, "rows": len(positions), "mot_sha256": file_info(mot)["sha256"]}},
                "provenance": {
                    "weights": file_info(self.project / "models" / model),
                    "tracker_config": file_info(self.project / f"configs/{tracker}.yaml"),
                    "splits": file_info(reports / "splits.json"),
                    "ignored": file_info(reports / "ignored_regions.json"),
                    "ignored_regions": {"MVI_1": []},
                },
                "env": {"device": "test", "date": "today"},
                "det_ms_per_frame": 5,
                "trk_ms_per_frame": 1,
                "total_ms_per_frame": 6,
                "fps": 1000 / 6,
            },
        )

    def main(self, *extra: str) -> None:
        """Call the script's entry point as a user would, inside the temporary project."""
        argv = ["run_ablation.py", "--confidences", "0.1", "0.6", "--raw", str(self.raw), "--out", str(self.out),
                *extra]
        with (self.project_paths(), patch("sys.argv", argv),
              patch.object(run_ablation, "run_script", side_effect=self.fake_run_script),
              contextlib.redirect_stdout(io.StringIO())):
            run_ablation.main()

    def test_equivalent_metadata_and_changed_copies(self) -> None:
        """Accept a re-formatted copy of the inputs; refuse changed bytes and a different split."""
        with self.project_paths():
            plan = run_ablation.build_plan(self.plan_args())
            job = plan["jobs"][0]
            tracks = self.out / "conf0.1/tracks"
            self.fake_run_script("run_tracking.py", self.tracking_args(job["model"], job["tracker"], 0.1, tracks))
            timing_path = tracks / "dev" / job["config"] / "timing.json"
            timing = read_json(timing_path)

            # Processed copies with the same JSON content but other bytes, as a tool may write them
            for key, name in (("splits", "splits.json"), ("ignored", "ignored_regions.json")):
                reference = self.project / "reports" / name
                copied = self.project / "processed" / name
                copied.parent.mkdir(exist_ok=True)
                copied.write_text(json.dumps(read_json(reference), separators=(",", ":")), encoding="utf-8")
                timing["provenance"][key] = file_info(copied)
                self.assertNotEqual(file_info(reference)["sha256"], file_info(copied)["sha256"])
            write_json(timing_path, timing)
            sweep = run_ablation.Sweep(self.out, {"plan": plan, "stages": {}})

            sweep.validate(job)

            # Bytes changed since tracking, although the JSON is still equivalent
            copied = self.project / "processed/splits.json"
            original = copied.read_bytes()
            copied.write_bytes(original + b" ")
            with self.assertRaisesRegex(SystemExit, "other inputs"):
                sweep.validate(job)

            # A genuinely different split must still be refused
            copied.write_bytes(original)
            changed = read_json(copied)
            changed["test"] = ["MVI_2"]
            write_json(copied, changed)
            timing["provenance"]["splits"] = file_info(copied)
            write_json(timing_path, timing)
            with self.assertRaisesRegex(SystemExit, "other inputs"):
                sweep.validate(job)

    def test_dataset_and_weights_changes_are_detected(self) -> None:
        """Refuse a sweep when the XML, a frame or the weights changed, or when an ignored region appeared."""
        xml = self.raw / "xml/MVI_1.xml"
        frame = self.raw / "frames/MVI_1/img00001.jpg"
        weights = self.project / "models/yolov8n.pt"

        with self.project_paths():
            for target, message in ((xml, "XML"), (frame, "frames"), (weights, "weights")):
                plan = run_ablation.build_plan(self.plan_args())
                sweep = run_ablation.Sweep(
                    self.out, {"plan": plan, "stages": {}, "weights": {"yolov8n.pt": file_info(weights)}}
                )

                # One extra byte in the input is enough to change its hash
                original = target.read_bytes()
                target.write_bytes(original + b" ")
                with self.subTest(target=target), self.assertRaisesRegex(SystemExit, message):
                    sweep.check_inputs()
                target.write_bytes(original)

            # A new ignored region changes the plan itself
            xml.write_text(
                xml.read_text(encoding="utf-8").replace(
                    "<ignored_region/>", '<ignored_region><box left="0" top="0" width="5" height="5"/></ignored_region>'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "ignored regions"):
                run_ablation.build_plan(self.plan_args())

    def test_lock_and_legacy_manifest(self) -> None:
        """Refuse to resume a locked sweep, and still validate a manifest written in the old format."""
        self.main()

        # A leftover lock file means another run may be active, so the manifest must stay untouched
        lock = self.out / ".running"
        lock.write_text("123", encoding="utf-8")
        original = (self.out / "manifest.json").read_bytes()
        with self.assertRaisesRegex(SystemExit, "locked"):
            self.main("--resume")
        self.assertEqual((self.out / "manifest.json").read_bytes(), original)
        lock.unlink()

        # Rewrite the manifest in the version-1 layout
        manifest = read_json(self.out / "manifest.json")
        manifest["version"], manifest["status"] = 1, "complete"
        manifest.pop("complete")
        manifest["stages"].pop("inputs")
        manifest["stages"] = {
            k: {"status": "complete", "outputs": [{"path": p, "sha256": h} for p, h in stage.items()]}
            for k, stage in manifest["stages"].items()
        }
        write_json(self.out / "manifest.json", manifest)
        self.assertTrue(validate_sweep(self.out)["complete"])

        # An output path that leaves the sweep folder is refused
        manifest["stages"]["gt_counts"]["outputs"][0]["path"] = "../outside.json"
        write_json(self.out / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_sweep(self.out)

    def test_sweep_resume_and_changed_settings(self) -> None:
        """Compute the whole sweep once, resume it without work, and refuse changed settings."""
        self.main()
        manifest = read_json(self.out / "manifest.json")

        self.assertTrue(manifest["complete"])
        self.assertEqual(self.calls.count("run_tracking.py"), EXPECTED_JOBS)
        self.assertEqual(self.calls.count("count_gt.py"), 1)
        for confidence, mae in ((0.1, 0), (0.6, 1)):
            report = read_json(self.out / f"conf{confidence:g}/reports/counting_eval_dev.json")
            self.assertEqual(report[f"yolov8n_bytetrack_conf{confidence:g}"]["all"]["mae"], mae)

        # A resumed finished sweep runs nothing; other settings are refused
        done = len(self.calls)
        self.main("--resume")
        self.assertEqual(len(self.calls), done)
        with self.assertRaises(SystemExit):
            self.main("--resume", "--imgsz", "640")
        with self.assertRaises(SystemExit):
            self.main()      # the folder exists and --resume is missing

    def test_dry_run_and_invalid_confidences(self) -> None:
        """Write and run nothing for `--dry-run`, and refuse duplicate confidences and a resume with no sweep."""
        self.main("--dry-run")

        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])
        with self.assertRaises(SystemExit):
            self.main("--confidences", "0.1", "0.1", "--dry-run")
        with self.assertRaises(SystemExit):
            self.main("--resume")      # nothing saved yet

    def test_edited_outputs_and_inputs_are_refused(self) -> None:
        """Refuse a resume after an output file was edited, and after the counting line changed."""
        self.main()
        mot = next(self.out.glob("conf0.1/tracks/dev/*/MVI_1.txt"))
        original = mot.read_bytes()

        mot.write_bytes(original + original)
        with self.assertRaisesRegex(SystemExit, "changed"):
            self.main("--resume")

        # Restoring the file makes the resume work again
        mot.write_bytes(original)
        self.main("--resume")

        write_json(self.project / "reports/counting_lines.json", {"MVI_1": {"p1": [30, 0], "p2": [30, 39]}})
        with self.assertRaises(SystemExit):
            self.main("--resume")

    def test_resume_after_a_failed_stage_keeps_finished_tracking(self) -> None:
        """Record a failed stage, release the lock, and finish the sweep without tracking again."""
        self.fail_once_on = "evaluate_counting.py"

        with self.assertRaises(SystemExit):
            self.main()

        manifest = read_json(self.out / "manifest.json")
        self.assertNotIn("complete", manifest)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((self.out / ".running").exists())

        self.main("--resume")

        self.assertEqual(self.calls.count("run_tracking.py"), EXPECTED_JOBS)
        self.assertTrue(read_json(self.out / "manifest.json")["complete"])

    def test_notebook_builds_tables_and_rejects_changed_outputs(self) -> None:
        """Run the ablation notebook code on the sweep: it makes tables and charts, and refuses edited outputs."""
        matplotlib.use("Agg", force=True)       # plot tests must not depend on a local Tk installation
        self.main()
        cells = [c for c in json.loads((ROOT / "notebooks/ablation.ipynb").read_text(encoding="utf-8"))["cells"]
                 if c["cell_type"] == "code"]
        scope: dict[str, Any] = {}

        with (contextlib.chdir(self.project),
              patch.dict(os.environ, {"ABLATION_DIR": str(self.out), "MPLBACKEND": "Agg"}),
              contextlib.redirect_stdout(io.StringIO())):
            for cell in cells:
                exec("".join(cell["source"]), scope)

            # 8 configurations; 8 x 4 class rows and 8 x 4 sequence rows (three classes and "all")
            self.assertEqual((len(scope["summary"]), len(scope["per_class"]), len(scope["per_sequence"])), (8, 32, 32))
            self.assertTrue((self.out / "analysis/tables/ablation_summary_dev.csv").is_file())
            self.assertEqual(len(list((self.out / "analysis/charts").glob("*.png"))), 5)

            # An edited result file must be detected when the notebook loads it again
            (self.out / "conf0.1/reports/counting_eval_dev.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Output changed"):
                exec("".join(cells[1]["source"]), scope)


if __name__ == "__main__":
    unittest.main()
