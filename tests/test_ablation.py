"""Tests for the ablation controller and notebook on a one-sequence project with simulated tracking."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import run_ablation
from src.config import ROOT
from src.utils import file_info, read_json, write_json


class AblationTests(unittest.TestCase):
    def setUp(self):
        """A copy of the code with one dev sequence: a car crosses the counting line at frame 2."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        for folder in ("src", "scripts", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder, ignore=shutil.ignore_patterns("__pycache__"))
        reports = self.project / "reports"
        write_json(reports / "splits.json", {"dev": ["MVI_1"], "test": []})
        write_json(reports / "ignored_regions.json", {"MVI_1": []})
        write_json(reports / "counting_lines.json", {"MVI_1": {"p1": [40, 0], "p2": [40, 39]}})

        self.raw = self.project / "raw"
        (self.raw / "frames/MVI_1").mkdir(parents=True)
        (self.raw / "xml").mkdir()
        targets = []
        for frame, left in enumerate((15, 35, 55), 1):
            Image.new("RGB", (80, 40)).save(self.raw / f"frames/MVI_1/img{frame:05d}.jpg")
            targets.append(f'<frame num="{frame}"><target_list><target id="1"><box left="{left}" top="15" '
                           'width="10" height="10"/><attribute vehicle_type="car"/></target></target_list></frame>')
        (self.raw / "xml/MVI_1.xml").write_text(
            '<sequence name="MVI_1"><ignored_region/>' + "".join(targets) + "</sequence>", encoding="utf-8")

        self.out = self.project / "outputs/ablation"
        self.calls: list[str] = []
        self.fail_once_on: str | None = None        # script name that fails on its next call
        self.real_run_script = run_ablation.run_script

    def fake_run_script(self, name, args):
        """Real scripts, but tracking is simulated: confidence 0.6 loses the last detection."""
        self.calls.append(name)
        if name == self.fail_once_on:
            self.fail_once_on = None
            raise SystemExit("Stage failed")
        if name != "run_tracking.py":
            return self.real_run_script(name, args)
        values = dict(zip(map(str, args[::2]), map(str, args[1::2]), strict=True))
        model, tracker, conf = values["--models"], values["--trackers"], float(values["--conf"])
        directory = Path(values["--out"]) / "dev" / f"{Path(model).stem}_{tracker}_conf{conf:g}"
        directory.mkdir(parents=True)
        positions = (15, 35, 55) if conf < 0.6 else (15, 35)
        mot = directory / "MVI_1.txt"
        mot.write_text("".join(f"{i},1,{x},15,10,10,0.9,0\n" for i, x in enumerate(positions, 1)), encoding="utf-8")
        reports = self.project / "reports"
        write_json(directory / "timing.json", {
            "detector": model, "tracker": tracker, "conf": conf, "split": "dev", "frames": 3,
            "imgsz": int(values["--imgsz"]), "iou": float(values["--iou"]), "seed": int(values["--seed"]),
            "per_seq": {"MVI_1": {"frames": 3, "rows": len(positions), "mot_sha256": file_info(mot)["sha256"]}},
            "provenance": {"tracker_config": file_info(self.project / f"configs/{tracker}.yaml"),
                           "splits": file_info(reports / "splits.json"),
                           "ignored": file_info(reports / "ignored_regions.json"),
                           "ignored_regions": {"MVI_1": []}},
            "env": {"device": "test", "date": "today"},
            "det_ms_per_frame": 5, "trk_ms_per_frame": 1, "total_ms_per_frame": 6, "fps": 1000 / 6})

    def main(self, *extra):
        """Call the script's entry point as a user would, inside the temporary project."""
        argv = ["run_ablation.py", "--confidences", "0.1", "0.6", "--raw", str(self.raw), "--out", str(self.out),
                *extra]
        paths = {"ROOT": self.project, "REPORTS_DIR": self.project / "reports", "CONFIG_DIR": self.project / "configs"}
        with (patch.multiple(run_ablation, **paths), patch("sys.argv", argv),
              patch.object(run_ablation, "run_script", side_effect=self.fake_run_script),
              contextlib.redirect_stdout(io.StringIO())):
            run_ablation.main()

    def test_sweep_resume_and_changed_settings(self):
        self.main()
        manifest = read_json(self.out / "manifest.json")
        self.assertTrue(manifest["complete"])
        self.assertEqual(self.calls.count("run_tracking.py"), 8)
        self.assertEqual(self.calls.count("count_gt.py"), 1)
        for confidence, mae in ((0.1, 0), (0.6, 1)):
            report = read_json(self.out / f"conf{confidence:g}/reports/counting_eval_dev.json")
            self.assertEqual(report["yolov8n_bytetrack_conf" + f"{confidence:g}"]["all"]["mae"], mae)

        # A resumed finished sweep runs nothing; other settings are refused
        done = len(self.calls)
        self.main("--resume")
        self.assertEqual(len(self.calls), done)
        with self.assertRaises(SystemExit):
            self.main("--resume", "--imgsz", "640")
        with self.assertRaises(SystemExit):
            self.main()                         # the folder exists and --resume is missing

    def test_dry_run_and_invalid_confidences(self):
        self.main("--dry-run")
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])
        with self.assertRaises(SystemExit):
            self.main("--confidences", "0.1", "0.1", "--dry-run")
        with self.assertRaises(SystemExit):
            self.main("--resume")               # nothing saved yet

    def test_edited_outputs_and_inputs_are_refused(self):
        self.main()
        mot = next(self.out.glob("conf0.1/tracks/dev/*/MVI_1.txt"))
        original = mot.read_bytes()
        mot.write_bytes(original + original)
        with self.assertRaisesRegex(SystemExit, "changed"):
            self.main("--resume")
        mot.write_bytes(original)
        self.main("--resume")

        write_json(self.project / "reports/counting_lines.json", {"MVI_1": {"p1": [30, 0], "p2": [30, 39]}})
        with self.assertRaises(SystemExit):
            self.main("--resume")

    def test_resume_after_a_failed_stage_keeps_finished_tracking(self):
        self.fail_once_on = "evaluate_counting.py"
        with self.assertRaises(SystemExit):
            self.main()
        self.assertNotIn("complete", read_json(self.out / "manifest.json"))
        self.main("--resume")
        self.assertEqual(self.calls.count("run_tracking.py"), 8)
        self.assertTrue(read_json(self.out / "manifest.json")["complete"])

    def test_notebook_builds_tables_and_rejects_changed_outputs(self):
        self.main()
        cells = [c for c in json.loads((ROOT / "notebooks/ablation.ipynb").read_text(encoding="utf-8"))["cells"]
                 if c["cell_type"] == "code"]
        scope: dict = {}
        with (contextlib.chdir(self.project), patch.dict(os.environ, {"ABLATION_DIR": str(self.out),
                                                                       "MPLBACKEND": "Agg"}),
              contextlib.redirect_stdout(io.StringIO())):
            for cell in cells:
                exec("".join(cell["source"]), scope)
            self.assertEqual((len(scope["summary"]), len(scope["per_class"]), len(scope["per_sequence"])), (8, 32, 32))
            self.assertTrue((self.out / "analysis/tables/ablation_summary_dev.csv").is_file())
            self.assertEqual(len(list((self.out / "analysis/charts").glob("*.png"))), 5)

            (self.out / "conf0.1/reports/counting_eval_dev.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Output changed"):
                exec("".join(cells[1]["source"]), scope)


if __name__ == "__main__":
    unittest.main()
