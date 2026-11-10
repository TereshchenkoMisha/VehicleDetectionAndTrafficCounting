"""Regression checks for repeated runs, interrupted writes and portable saved artifacts."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from scripts import evaluate_detectors, prepare_dataset, render_failure_cases, run_all, run_tracking
from src.config import ROOT
from src.data.detrac import Box, FrameAnn, SequenceAnn, SequenceMeta, Target
from src.data.export import convert
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, matches_file_hash, publish_directory, read_json, write_json


class RerunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        sequence = self.raw / "MVI_1"
        sequence.mkdir(parents=True)
        for fid in range(1, 4):
            Image.new("RGB", (80, 40), (fid, 20, 30)).save(sequence / f"img{fid:05d}.jpg")
        self.anns = {"MVI_1": SequenceAnn("MVI_1", {
            fid: FrameAnn(fid, [Target(1, "car", Box(10 * fid, 10, 10, 10), 0, 0)]) for fid in range(1, 4)
        })}
        self.split = {"dev": ["MVI_1"], "test": [], "weather": {"MVI_1": "day"}}
        self.data = self.root / "processed"

    def convert(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return convert(self.anns, self.split, out_dir=self.data, frames_root=self.raw, copy=True, **kwargs)

    def test_identical_export_is_reused_without_copying(self):
        first = self.convert()
        with patch("src.data.export._place_image", side_effect=AssertionError("should reuse")):
            self.assertEqual(self.convert(), first)
        self.assertFalse((self.root / ".previous").exists())

    def test_incomplete_export_is_rebuilt_and_old_files_are_retained(self):
        (self.data / "images/dev").mkdir(parents=True)
        (self.data / "images/dev/interrupted.jpg").write_bytes(b"unfinished")
        self.convert()
        self.assertEqual(len(list((self.data / "images/dev").glob("*.jpg"))), 3)
        backups = list((self.root / ".previous").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "images/dev/interrupted.jpg").read_bytes(), b"unfinished")

    def test_missing_label_and_changed_annotations_are_repaired(self):
        self.convert()
        label = self.data / "labels/dev/MVI_1_f000001.txt"
        expected = label.read_bytes()
        label.unlink()
        self.convert()
        self.assertEqual(label.read_bytes(), expected)
        self.anns["MVI_1"].frames[1].targets[0].cls = "bus"
        self.convert()
        self.assertTrue(label.read_text().startswith("1 "))

    def test_changed_sampling_does_not_mix_old_frames(self):
        self.convert()
        self.convert(frame_step=2)
        self.assertEqual([p.name for p in (self.data / "images/dev").glob("*.jpg")], ["MVI_1_f000002.jpg"])
        self.assertEqual([p.name for p in (self.data / "labels/dev").glob("*.txt")], ["MVI_1_f000002.txt"])

    def test_old_split_does_not_block_requested_fixed_split(self):
        write_json(self.data / "splits.json", {"dev": ["MVI_2"], "test": [], "weather": {"MVI_2": "day"}})
        fixed = self.root / "fixed.json"
        write_json(fixed, self.split)
        argv = ["prepare_dataset.py", "--raw", str(self.raw), "--out", str(self.data), "--split-file", str(fixed)]
        with (patch("sys.argv", argv), patch.object(prepare_dataset, "load_all", return_value=(
                self.anns, {"MVI_1": SequenceMeta("MVI_1", "day")})), contextlib.redirect_stdout(io.StringIO())):
            prepare_dataset.main()
        self.assertEqual(read_json(self.data / "splits.json"), self.split)

    def test_extra_jpeg_is_not_reused_as_part_of_the_current_dataset(self):
        self.convert()
        Image.new("RGB", (80, 40)).save(self.data / "images/dev/unlabelled.JPEG")
        self.convert()
        self.assertFalse((self.data / "images/dev/unlabelled.JPEG").exists())

    def test_failed_rebuild_keeps_active_export(self):
        self.convert()
        original = (self.data / "stats.json").read_bytes()
        with (patch("src.data.export._place_image", side_effect=OSError("disk full")),
              self.assertRaisesRegex(OSError, "disk full")):
            self.convert(frame_step=2)
        self.assertEqual((self.data / "stats.json").read_bytes(), original)
        self.assertEqual(len(list((self.data / "images/dev").glob("*.jpg"))), 3)

    def tracking(self, fail=False):
        metadata, configs = self.root / "metadata", self.root / "configs"
        write_json(metadata / "splits.json", self.split)
        write_json(metadata / "ignored_regions.json", {"MVI_1": []})
        configs.mkdir(exist_ok=True)
        (configs / "bytetrack.yaml").write_text("tracker_type: bytetrack\n")
        weights = self.root / "yolov8n.pt"
        weights.write_bytes(b"test weights")
        argv = ["run_tracking.py", "--models", "yolov8n.pt", "--trackers", "bytetrack", "--data", str(metadata),
                "--raw", str(self.raw), "--out", str(self.root / "tracks"), "--device", "cpu"]

        def fake_track(_weights, _tracker, _frames, _ignored, out, **_kwargs):
            mot = out / "MVI_1.txt"
            mot.write_text("1,1,10,10,10,10,0.9,0,-1,-1\n")
            if fail:
                raise RuntimeError("inference interrupted")
            return {"frames": 3, "per_seq": {"MVI_1": {
                "frames": 3, "rows": 1, "mot_sha256": file_info(mot)["sha256"]}},
                "det_ms_per_frame": 5, "trk_ms_per_frame": 1, "fps": 100}

        with (patch("sys.argv", argv), patch.object(run_tracking, "CONFIG_DIR", configs),
              patch.object(run_tracking, "weights_path", return_value=str(weights)),
              patch.object(run_tracking, "track_config", side_effect=fake_track),
              contextlib.redirect_stdout(io.StringIO())):
            run_tracking.main()

    def test_tracking_twice_archives_old_run_outside_configuration_list(self):
        self.tracking()
        self.tracking()
        split_dir = self.root / "tracks/dev"
        self.assertEqual(len(run_dirs(split_dir)), 1)
        validate_tracking_run(run_dirs(split_dir)[0], ["MVI_1"], self.raw, "dev")
        self.assertEqual(len(list((self.root / "tracks/.previous/dev").iterdir())), 1)

    def test_interrupted_tracking_keeps_previous_complete_run(self):
        self.tracking()
        cfg = run_dirs(self.root / "tracks/dev")[0]
        before = (cfg / "timing.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.tracking(fail=True)
        self.assertEqual((cfg / "timing.json").read_bytes(), before)
        self.assertEqual(run_dirs(self.root / "tracks/dev"), [cfg])
        self.tracking()

    def test_detector_twice_and_failed_inference_leave_valid_report(self):
        argv = ["evaluate_detectors.py", "--models", "yolov8n.pt", "--out-dir", str(self.root)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            for frames in (3, 4):
                with patch.object(evaluate_detectors, "evaluate_detector", return_value={"frames": frames}):
                    evaluate_detectors.main()
            with (patch.object(evaluate_detectors, "evaluate_detector", side_effect=RuntimeError("failed")),
                  self.assertRaises(RuntimeError)):
                evaluate_detectors.main()
        self.assertEqual(read_json(self.root / "detection_yolov8n_dev.json"), {"frames": 4})

    def test_publish_failure_restores_previous_directory(self):
        active, staged = self.root / "active", self.root / "staged"
        active.mkdir()
        (active / "old.txt").write_text("old")
        staged.mkdir()
        original_rename = Path.rename

        def fail_publish(path, target):
            if path == staged:
                raise OSError("publication failed")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_publish), self.assertRaisesRegex(OSError, "publication failed"):
            publish_directory(staged, active)
        self.assertEqual((active / "old.txt").read_text(), "old")

    def test_cross_platform_hashes_allow_only_line_endings(self):
        text = self.root / "tracks.txt"
        text.write_bytes(b"1,2,3\r\n4,5,6\r\n")
        digest = file_info(text)["sha256"]
        text.write_bytes(b"1,2,3\n4,5,6\n")
        self.assertTrue(matches_file_hash(text, digest))
        text.write_bytes(b"1,2,3 \n4,5,6\n")
        self.assertFalse(matches_file_hash(text, digest))
        binary = self.root / "weights.pt"
        binary.write_bytes(b"1,2,3\n4,5,6\n")
        self.assertFalse(matches_file_hash(binary, digest))

    def test_results_notebook_accepts_line_endings_but_rejects_changed_timing(self):
        timing = self.root / "timing.json"
        timing.write_bytes(b'{"frames": 3}\r\n')
        digest = file_info(timing)["sha256"]
        timing.write_bytes(b'{"frames": 3}\n')
        metric_names = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1")
        counters = {name: 0 for name in ("IDSW", "MT", "ML")}
        write_json(self.root / "tracking_eval_dev.json", {"baseline": {
            "sources": {"timing": {"sha256": digest}},
            "aggregate": {"combined": {name: 50 for name in metric_names}, "sum": counters}}})
        notebook = read_json(ROOT / "notebooks/results.ipynb")
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "tracking_json =" in "".join(cell["source"]))
        scope = {"reports_dir": self.root, "SPLIT": "dev", "json": json, "pd": pd,
                 "matches_file_hash": matches_file_hash, "APPROACHES_CONFIG": [{
                     "config": "baseline", "name": "baseline", "timing_source": file_info(timing)}]}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "results.ipynb", "exec"), scope)
            timing.write_bytes(b'{"frames": 2}\n')
            with self.assertRaisesRegex(ValueError, "different run"):
                exec(compile(source, "results.ipynb", "exec"), scope)

    def test_preparation_stage_does_not_skip_a_partial_images_folder(self):
        with patch("sys.argv", ["scripts/run_all.py", "--from", "prepare", "--to", "prepare"]):
            run = run_all.Run(run_all.parse_args())
        (self.root / run_all.PROCESSED / "images").mkdir(parents=True)
        with (patch.object(run_all, "ROOT", self.root), patch.object(run, "step") as step,
              contextlib.redirect_stdout(io.StringIO())):
            run.run()
        self.assertTrue(any(call.args[1][0] == "scripts/prepare_dataset.py" for call in step.call_args_list))

    def test_default_includes_ablation_and_tag_cannot_escape_project(self):
        with patch("sys.argv", ["scripts/run_all.py"]):
            self.assertEqual(run_all.parse_args().last, 11)
        with (patch("sys.argv", ["scripts/run_all.py", "--tag", "../../outside"]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            run_all.parse_args()

    def test_input_check_never_runs_or_cleans(self):
        with (patch("sys.argv", ["scripts/run_all.py", "--check", "--clean", "--yes"]),
              patch.object(run_all.Run, "check_files", return_value=0),
              patch.object(run_all.Run, "check_content", return_value=0),
              patch.object(run_all.Run, "clean") as clean, patch.object(run_all.Run, "run") as run,
              contextlib.redirect_stdout(io.StringIO())):
            run_all.main()
        clean.assert_not_called()
        run.assert_not_called()

    def test_failure_candidate_can_be_rendered_without_manual_review(self):
        analysis = self.root / "analysis.json"
        ignored, lines = self.root / "ignored.json", self.root / "lines.json"
        write_json(analysis, {"split": "dev", "rows": [{
            "case_id": "case", "sequence": "MVI_1", "gt_event": {"frame": 2}}]})
        write_json(ignored, {"MVI_1": []})
        write_json(lines, {"MVI_1": {"p1": [40, 0], "p2": [40, 39]}})
        argv = ["render_failure_cases.py", "--analysis", str(analysis), "--review", str(self.root / "absent.json"),
                "--case-id", "case", "--lines", str(lines), "--ignored", str(ignored),
                "--out-dir", str(self.root / "pictures")]
        with (patch("sys.argv", argv), patch.object(render_failure_cases, "build_view"),
              patch.object(render_failure_cases, "write_case") as write, contextlib.redirect_stdout(io.StringIO())):
            render_failure_cases.main()
        write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
