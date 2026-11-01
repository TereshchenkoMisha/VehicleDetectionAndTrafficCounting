"""Check the sweep controller with small local data and simulated detector output"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.data.provenance import file_info, write_json
from src.scripts import run_ablation


class AblationTests(unittest.TestCase):
    def setUp(self):
        """Create one tiny development video with a known line crossing"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.project.mkdir()
        shutil.copytree(run_ablation.ROOT / "src", self.project / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        reports = self.project / "reports"
        reports.mkdir()
        write_json(reports / "splits.json", {"dev": ["MVI_1"], "test": []})
        write_json(reports / "ignored_regions.json", {"MVI_1": []})
        write_json(reports / "counting_lines.json", {"MVI_1": {"p1": [40, 0], "p2": [40, 39]}})
        self.raw = self.project / "data/raw/UA-DETRAC"
        frames = self.raw / "DETRAC-Images/MVI_1"
        frames.mkdir(parents=True)
        xml_frames = []
        for frame, left in enumerate((15, 35, 55), 1):
            Image.new("RGB", (80, 40)).save(frames / f"img{frame:05d}.jpg")
            xml_frames.append(
                f'<frame num="{frame}"><target_list><target id="1">'
                f'<box left="{left}" top="15" width="10" height="10"/>'
                '<attribute vehicle_type="car"/></target></target_list></frame>')
        annotations = self.raw / "DETRAC-Train-Annotations-XML"
        annotations.mkdir()
        (annotations / "MVI_1.xml").write_text(
            '<sequence name="MVI_1"><ignored_region/>' + "".join(xml_frames) + '</sequence>',
            encoding="utf-8")
        (self.project / "configs").mkdir()
        (self.project / "models").mkdir()
        for tracker in run_ablation.TRACKERS:
            (self.project / "configs" / f"{tracker}.yaml").write_text(
                f"tracker_type: {tracker}\n", encoding="utf-8")
        for model in run_ablation.MODELS:
            (self.project / "models" / model).write_bytes(b"test weights")
        self.output = self.project / "outputs/ablation/test"
        self.commands = []

    def command(self, module, arguments):
        """Simulate inference and let the real scripts prepare GT and evaluate results"""
        self.commands.append(module)
        if module != "src.scripts.run_tracking":
            command = [sys.executable, "-X", "utf8", "-B", "-m", module, *map(str, arguments)]
            run_ablation.subprocess.run(command, cwd=self.project, check=True,
                                        stdout=run_ablation.subprocess.PIPE,
                                        stderr=run_ablation.subprocess.PIPE)
            return
        values = dict(zip(map(str, arguments[::2]), map(str, arguments[1::2])))
        model, tracker = values["--models"], values["--trackers"]
        confidence = float(values["--conf"])
        config = f"{Path(model).stem}_{tracker}_conf{confidence:g}"
        directory = Path(values["--out"]) / "dev" / config
        directory.mkdir(parents=True)
        selected = (15, 35, 55) if confidence < 0.6 else (15, 35)
        mot = directory / "MVI_1.txt"
        mot.write_text("".join(f"{frame},1,{left},15,10,10,0.9,0,-1,-1\n"
                               for frame, left in enumerate(selected, 1)), encoding="utf-8")
        data = Path(values["--data"])
        provenance = {"tracker_config": file_info(self.project / "configs" / f"{tracker}.yaml"),
                      "weights": file_info(self.project / "models" / model),
                      "splits": file_info(data / "splits.json"),
                      "ignored": file_info(data / "ignored_regions.json"),
                      "ignored_regions": {"MVI_1": []}}
        timing = {"detector": model, "tracker": tracker, "conf": confidence, "split": "dev",
                  "imgsz": int(values["--imgsz"]), "iou": float(values["--iou"]),
                  "seed": int(values["--seed"]), "deterministic": True, "frames": 3,
                  "per_seq": {"MVI_1": {"frames": 3, "rows": len(selected),
                                         "mot_sha256": file_info(mot)["sha256"]}},
                  "provenance": provenance, "env": {"device": "test", "date": "test"},
                  "det_ms_per_frame": 5, "trk_ms_per_frame": 1,
                  "total_ms_per_frame": 6, "fps": 1000 / 6}
        write_json(directory / "timing.json", timing)

    def run_controller(self, extra=()):
        """Use the public entry point with the same arguments as a user command"""
        arguments = ["run_ablation", "--confidences", "0.1", "0.6", "--raw", str(self.raw),
                     "--out", str(self.output), *extra]
        with patch.object(run_ablation, "ROOT", self.project), patch("sys.argv", arguments):
            with patch.object(run_ablation, "run_command", side_effect=self.command):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    run_ablation.main()

    def test_complete_sweep_and_resume(self):
        """Check the full pipeline and confirm that resume reuses validated stages"""
        self.run_controller()
        manifest = run_ablation.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["plan"]["jobs"]), 8)
        self.assertEqual(self.commands.count("src.scripts.count_gt_crossings"), 1)
        self.assertEqual(self.commands.count("src.scripts.prepare_gt_mot"), 1)
        self.assertEqual(self.commands.count("src.scripts.run_tracking"), 8)
        for confidence, expected in ((0.1, 0), (0.6, 1)):
            result = run_ablation.read_json(self.output / f"conf{confidence:g}/reports/counting_eval_dev.json")
            for name, body in result.items():
                if not name.startswith("_"):
                    self.assertEqual(body["all"]["mae"], expected)
        previous_count = len(self.commands)
        self.run_controller(["--resume"])
        self.assertEqual(len(self.commands), previous_count)
        with self.assertRaises(SystemExit):
            self.run_controller(["--resume", "--imgsz", "640"])
        self.assertEqual(run_ablation.read_json(self.output / "manifest.json")["status"], "complete")

    def test_dry_run_and_invalid_confidences(self):
        """Check that inspection and invalid options do not create an experiment"""
        self.run_controller(["--dry-run"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.commands, [])
        for values in ([0.1, 0.1], [0.1, float("nan")], [0.1], [-0.1, 0.6]):
            args = argparse.Namespace(confidences=values, imgsz=960, iou=0.7, seed=42,
                                      device=None, raw=str(self.raw))
            with patch.object(run_ablation, "ROOT", self.project), self.assertRaises(ValueError):
                run_ablation.build_plan(args)

    def test_resume_rejects_changed_tracks_and_input_files(self):
        """Detect edited output files and settings before they are reused"""
        self.run_controller()
        manifest = run_ablation.read_json(self.output / "manifest.json")
        job = manifest["plan"]["jobs"][0]
        mot = self.output / job["tracks"] / "dev" / job["config"] / "MVI_1.txt"
        original = mot.read_bytes()
        mot.write_bytes(original + original)
        with self.assertRaises(SystemExit):
            self.run_controller(["--resume"])
        mot.write_bytes(original)
        self.run_controller(["--resume"])
        lines = self.project / "reports/counting_lines.json"
        write_json(lines, {"MVI_1": {"p1": [30, 0], "p2": [30, 39]}})
        with self.assertRaises(SystemExit):
            self.run_controller(["--resume"])

    def test_resume_after_evaluation_failure(self):
        """Continue a failed evaluation without repeating the finished tracking runs"""
        normal_command = self.command
        failed = False

        def fail_once(module, arguments):
            nonlocal failed
            if module == "src.scripts.eval_counting" and not failed:
                failed = True
                raise run_ablation.subprocess.CalledProcessError(1, module)
            normal_command(module, arguments)

        with patch.object(self, "command", side_effect=fail_once):
            with self.assertRaises(SystemExit):
                self.run_controller()
        self.assertEqual(run_ablation.read_json(self.output / "manifest.json")["status"], "failed")
        self.assertFalse((self.output / ".running").exists())
        self.run_controller(["--resume"])
        self.assertEqual(self.commands.count("src.scripts.run_tracking"), 8)
        self.assertEqual(run_ablation.read_json(self.output / "manifest.json")["status"], "complete")

    def test_notebook_tables_plots_and_corrupt_report(self):
        """Run all notebook cells on the saved test experiment and reject stale metrics"""
        self.run_controller()
        notebook_path = run_ablation.ROOT / "scripts/ablation.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        scripts = self.project / "scripts"
        scripts.mkdir()
        shutil.copyfile(notebook_path, scripts / "ablation.ipynb")
        scope = {"__name__": "__main__"}
        environment = {"ABLATION_DIR": str(self.output), "MPLBACKEND": "Agg"}
        with contextlib.chdir(self.project), patch.dict(os.environ, environment):
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), "ablation.ipynb", "exec"), scope)
        self.assertEqual(len(scope["summary"]), 8)
        self.assertEqual(len(scope["per_class"]), 32)
        self.assertEqual(len(scope["per_sequence"]), 32)
        self.assertTrue((self.output / "analysis/tables/ablation_summary_dev.csv").is_file())
        self.assertEqual(len(list((self.output / "analysis/charts").glob("*.png"))), 5)
        damaged = self.output / "conf0.1/reports/counting_eval_dev.json"
        damaged.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output changed"):
            exec("".join(notebook["cells"][3]["source"]), scope)


if __name__ == "__main__":
    unittest.main()
