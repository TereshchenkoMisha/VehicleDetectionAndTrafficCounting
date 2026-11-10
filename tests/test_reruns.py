"""Regression checks for repeated runs, interrupted writes and portable saved artifacts.

Tests (all in RerunTests, on a temporary project with one three-frame sequence):
    test_identical_export_is_reused_without_copying     an unchanged dataset export is not copied again
    test_incomplete_export_is_rebuilt_and_old_files_are_retained   a half-written export is rebuilt; old files kept
    test_missing_label_and_changed_annotations_are_repaired        a deleted label or changed annotation is regenerated
    test_changed_sampling_does_not_mix_old_frames       a new frame step removes the frames of the old sampling
    test_old_split_does_not_block_requested_fixed_split a fixed split file replaces an older saved split
    test_extra_jpeg_is_not_reused_as_part_of_the_current_dataset   a stray image makes the export rebuild
    test_failed_rebuild_keeps_active_export             a failed rebuild leaves the previous export in place
    test_tracking_twice_archives_old_run_outside_configuration_list   a repeated tracking run archives the old one
    test_interrupted_tracking_keeps_previous_complete_run   a failed tracking run leaves the previous run valid
    test_sequences_without_a_line_are_reported_next_to_the_videos   sequences without a line are listed in a report
    test_detector_twice_and_failed_inference_leave_valid_report     a failed detector run keeps the last good report
    test_publish_failure_restores_previous_directory    a failed publish puts the old directory back
    test_cross_platform_hashes_allow_only_line_endings  a hash matches across LF and CRLF only for text files
    test_results_notebook_accepts_line_endings_but_rejects_changed_timing   the notebook checks timing by hash
    test_preparation_stage_does_not_skip_a_partial_images_folder   run_all prepares data when images are incomplete
    test_default_includes_ablation_and_tag_cannot_escape_project   default stages and a safe run tag
    test_input_check_never_runs_or_cleans               `--check` neither cleans nor runs any stage
    test_failure_candidate_can_be_rendered_without_manual_review   a failure case renders without a review file
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from PIL import Image

from scripts import count_gt, evaluate_detectors, prepare_dataset, render_failure_cases, run_all, run_tracking
from src.config import ROOT
from src.data.detrac import Box, FrameAnn, SequenceAnn, SequenceMeta, Target
from src.data.export import convert
from src.data.sequences import read_frame
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, matches_file_hash, publish_directory, read_json, write_json

FRAME_COUNT: int = 3        # frames of the test sequence
LINE: dict[str, list[int]] = {"p1": [40, 0], "p2": [40, 39]}        # counting line in the render test


class RerunTests(unittest.TestCase):
    """Reruns of the pipeline stages on a small temporary project."""

    def setUp(self) -> None:
        """Create sequence MVI_1 with three frames and one moving car, and the paths of the processed data."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        sequence = self.raw / "MVI_1"
        sequence.mkdir(parents=True)
        for fid in range(1, FRAME_COUNT + 1):
            Image.new("RGB", (80, 40), (fid, 20, 30)).save(sequence / f"img{fid:05d}.jpg")
        self.anns = {"MVI_1": SequenceAnn("MVI_1", {
            fid: FrameAnn(fid, [Target(1, "car", Box(10 * fid, 10, 10, 10), 0, 0)])
            for fid in range(1, FRAME_COUNT + 1)
        })}
        self.split = {"dev": ["MVI_1"], "test": [], "weather": {"MVI_1": "day"}}
        self.data = self.root / "processed"

    def convert(self, **kwargs: Any) -> Any:
        """Export the test dataset to `self.data` without printing."""
        with contextlib.redirect_stdout(io.StringIO()):
            return convert(self.anns, self.split, out_dir=self.data, frames_root=self.raw, copy=True, **kwargs)

    def test_identical_export_is_reused_without_copying(self) -> None:
        """Reuse an unchanged export: no image is copied and nothing is archived."""
        first = self.convert()

        with patch("src.data.export._place_image", side_effect=AssertionError("should reuse")):
            self.assertEqual(self.convert(), first)

        self.assertFalse((self.root / ".previous").exists())

    def test_incomplete_export_is_rebuilt_and_old_files_are_retained(self) -> None:
        """Rebuild an export that was interrupted, and archive its files instead of deleting them."""
        (self.data / "images/dev").mkdir(parents=True)
        (self.data / "images/dev/interrupted.jpg").write_bytes(b"unfinished")

        self.convert()

        self.assertEqual(len(list((self.data / "images/dev").glob("*.jpg"))), FRAME_COUNT)
        backups = list((self.root / ".previous").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "images/dev/interrupted.jpg").read_bytes(), b"unfinished")

    def test_missing_label_and_changed_annotations_are_repaired(self) -> None:
        """Regenerate a deleted label file, and a label whose annotation changed."""
        self.convert()
        label = self.data / "labels/dev/MVI_1_f000001.txt"
        expected = label.read_bytes()

        # A deleted label comes back with the same content
        label.unlink()
        self.convert()
        self.assertEqual(label.read_bytes(), expected)

        # A car that became a bus is exported with class id 1
        self.anns["MVI_1"].frames[1].targets[0].cls = "bus"
        self.convert()
        self.assertTrue(label.read_text(encoding="utf-8").startswith("1 "))

    def test_changed_sampling_does_not_mix_old_frames(self) -> None:
        """Keep only the frames of the new sampling when the frame step changes."""
        self.convert()

        self.convert(frame_step=2)

        self.assertEqual([p.name for p in (self.data / "images/dev").glob("*.jpg")], ["MVI_1_f000002.jpg"])
        self.assertEqual([p.name for p in (self.data / "labels/dev").glob("*.txt")], ["MVI_1_f000002.txt"])

    def test_old_split_does_not_block_requested_fixed_split(self) -> None:
        """Replace an older saved split with the requested split file."""
        write_json(self.data / "splits.json", {"dev": ["MVI_2"], "test": [], "weather": {"MVI_2": "day"}})
        fixed = self.root / "fixed.json"
        write_json(fixed, self.split)
        argv = ["prepare_dataset.py", "--raw", str(self.raw), "--out", str(self.data), "--split-file", str(fixed)]

        with (patch("sys.argv", argv),
              patch.object(prepare_dataset, "load_all",
                           return_value=(self.anns, {"MVI_1": SequenceMeta("MVI_1", "day")})),
              contextlib.redirect_stdout(io.StringIO())):
            prepare_dataset.main()

        self.assertEqual(read_json(self.data / "splits.json"), self.split)

    def test_extra_jpeg_is_not_reused_as_part_of_the_current_dataset(self) -> None:
        """Rebuild the export when an unlabelled image appears in it."""
        self.convert()
        Image.new("RGB", (80, 40)).save(self.data / "images/dev/unlabelled.JPEG")

        self.convert()

        self.assertFalse((self.data / "images/dev/unlabelled.JPEG").exists())

    def test_failed_rebuild_keeps_active_export(self) -> None:
        """Leave the previous export untouched when a rebuild fails halfway."""
        self.convert()
        original = (self.data / "stats.json").read_bytes()

        with (patch("src.data.export._place_image", side_effect=OSError("disk full")),
              self.assertRaisesRegex(OSError, "disk full")):
            self.convert(frame_step=2)

        self.assertEqual((self.data / "stats.json").read_bytes(), original)
        self.assertEqual(len(list((self.data / "images/dev").glob("*.jpg"))), FRAME_COUNT)

    def tracking(self, fail: bool = False) -> None:
        """Run `run_tracking.main` for one configuration, with a fake `track_config` that may fail."""
        metadata, configs = self.root / "metadata", self.root / "configs"
        write_json(metadata / "splits.json", self.split)
        write_json(metadata / "ignored_regions.json", {"MVI_1": []})
        configs.mkdir(exist_ok=True)
        (configs / "bytetrack.yaml").write_text("tracker_type: bytetrack\n", encoding="utf-8")
        weights = self.root / "yolov8n.pt"
        weights.write_bytes(b"test weights")
        argv = ["run_tracking.py", "--models", "yolov8n.pt", "--trackers", "bytetrack", "--data", str(metadata),
                "--raw", str(self.raw), "--out", str(self.root / "tracks"), "--device", "cpu"]

        def fake_track(_weights: str, _tracker: str, _frames: Any, _ignored: Any, out: Path,
                       **_kwargs: Any) -> dict[str, Any]:
            """Write one MOT row and return the timing report, or fail after writing."""
            mot = out / "MVI_1.txt"
            mot.write_text("1,1,10,10,10,10,0.9,0,-1,-1\n", encoding="utf-8")
            if fail:
                raise RuntimeError("inference interrupted")
            return {"frames": FRAME_COUNT, "per_seq": {"MVI_1": {
                "frames": FRAME_COUNT, "rows": 1, "mot_sha256": file_info(mot)["sha256"]}},
                "det_ms_per_frame": 5, "trk_ms_per_frame": 1, "fps": 100}

        with (patch("sys.argv", argv), patch.object(run_tracking, "CONFIG_DIR", configs),
              patch.object(run_tracking, "weights_path", return_value=str(weights)),
              patch.object(run_tracking, "track_config", side_effect=fake_track),
              contextlib.redirect_stdout(io.StringIO())):
            run_tracking.main()

    def test_tracking_twice_archives_old_run_outside_configuration_list(self) -> None:
        """Archive the first run when tracking is repeated, so only the new run is listed."""
        self.tracking()
        self.tracking()
        split_dir = self.root / "tracks/dev"

        self.assertEqual(len(run_dirs(split_dir)), 1)
        validate_tracking_run(run_dirs(split_dir)[0], ["MVI_1"], self.raw, "dev")
        self.assertEqual(len(list((self.root / "tracks/.previous/dev").iterdir())), 1)

    def test_interrupted_tracking_keeps_previous_complete_run(self) -> None:
        """Keep the previous run unchanged when a repeated run fails, and allow a later retry."""
        self.tracking()
        cfg = run_dirs(self.root / "tracks/dev")[0]
        before = (cfg / "timing.json").read_bytes()

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.tracking(fail=True)

        self.assertEqual((cfg / "timing.json").read_bytes(), before)
        self.assertEqual(run_dirs(self.root / "tracks/dev"), [cfg])
        self.tracking()

    def test_sequences_without_a_line_are_reported_next_to_the_videos(self) -> None:
        """List the videos of sequences without a counting line in a report, and warn."""
        out_dirs = [self.root / "tracks/dev/yolov8n_bytetrack_conf0.25"]

        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_tracking.report_missing_lines(["MVI_1", "MVI_2"], {"MVI_1": {}}, "lines.json",
                                              self.root / "videos", out_dirs)
            run_tracking.report_missing_lines(["MVI_1"], {"MVI_1": {}}, "lines.json", self.root / "none", out_dirs)

        report = read_json(self.root / "videos/videos_without_line.json")
        self.assertEqual(report["sequences"], ["MVI_2"])
        self.assertEqual(report["videos"], ["yolov8n_bytetrack_conf0.25_MVI_2.mp4"])
        self.assertIn("MVI_2", output.getvalue())
        self.assertFalse((self.root / "none").exists())      # nothing is missing there, so no report is written

    def test_detector_twice_and_failed_inference_leave_valid_report(self) -> None:
        """Replace the detector report on a repeat, and keep the last good one when inference fails."""
        argv = ["evaluate_detectors.py", "--models", "yolov8n.pt", "--out-dir", str(self.root)]

        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            for frames in (3, 4):
                with patch.object(evaluate_detectors, "evaluate_detector", return_value={"frames": frames}):
                    evaluate_detectors.main()
            with (patch.object(evaluate_detectors, "evaluate_detector", side_effect=RuntimeError("failed")),
                  self.assertRaises(RuntimeError)):
                evaluate_detectors.main()

        self.assertEqual(read_json(self.root / "detection_yolov8n_dev.json"), {"frames": 4})

    def test_publish_failure_restores_previous_directory(self) -> None:
        """Put the previous directory back when publishing the new one fails."""
        active, staged = self.root / "active", self.root / "staged"
        active.mkdir()
        (active / "old.txt").write_text("old", encoding="utf-8")
        staged.mkdir()
        original_rename = Path.rename

        def fail_publish(path: Path, target: Path) -> Path:
            """Fail only the move of the staged directory; archiving the old one works."""
            if path == staged:
                raise OSError("publication failed")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_publish), self.assertRaisesRegex(OSError, "publication failed"):
            publish_directory(staged, active)

        self.assertEqual((active / "old.txt").read_text(encoding="utf-8"), "old")

    def test_cross_platform_hashes_allow_only_line_endings(self) -> None:
        """Accept a changed line ending in a text file, and nothing else, and no change at all in a binary file."""
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

    def test_results_notebook_accepts_line_endings_but_rejects_changed_timing(self) -> None:
        """Run the notebook cell that checks the timing file: a line-ending change passes, other changes fail."""
        timing = self.root / "timing.json"
        timing.write_bytes(b'{"frames": 3}\r\n')
        digest = file_info(timing)["sha256"]
        timing.write_bytes(b'{"frames": 3}\n')
        metric_names = ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1")
        counters = {name: 0 for name in ("IDSW", "MT", "ML")}
        write_json(self.root / "tracking_eval_dev.json", {"baseline": {
            "sources": {"timing": {"sha256": digest}},
            "aggregate": {"combined": {name: 50 for name in metric_names}, "sum": counters}}})

        # The cell under test is the one that builds `tracking_json`; the other variables are what it reads
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

    def test_preparation_stage_does_not_skip_a_partial_images_folder(self) -> None:
        """Run the prepare stage when the images folder exists but is not a complete dataset."""
        with patch("sys.argv", ["scripts/run_all.py", "--from", "prepare", "--to", "prepare"]):
            run = run_all.Run(run_all.parse_args())
        (self.root / run_all.PROCESSED / "images").mkdir(parents=True)

        with (patch.object(run_all, "ROOT", self.root), patch.object(run, "step") as step,
              contextlib.redirect_stdout(io.StringIO())):
            run.run()

        self.assertTrue(any(call.args[1][0] == "scripts/prepare_dataset.py" for call in step.call_args_list))

    def test_default_includes_ablation_and_tag_cannot_escape_project(self) -> None:
        """Run up to the ablation stage by default, and refuse a tag that points outside the project."""
        with patch("sys.argv", ["scripts/run_all.py"]):
            self.assertEqual(run_all.parse_args().last, 11)

        with (patch("sys.argv", ["scripts/run_all.py", "--tag", "../../outside"]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            run_all.parse_args()

    def test_input_check_never_runs_or_cleans(self) -> None:
        """Do nothing but check inputs for `--check`, even together with `--clean`."""
        with (patch("sys.argv", ["scripts/run_all.py", "--check", "--clean", "--yes"]),
              patch.object(run_all.Run, "check_files", return_value=0),
              patch.object(run_all.Run, "check_content", return_value=0),
              patch.object(run_all.Run, "clean") as clean, patch.object(run_all.Run, "run") as run,
              contextlib.redirect_stdout(io.StringIO())):
            run_all.main()

        clean.assert_not_called()
        run.assert_not_called()

    def test_gt_export_is_enabled_when_its_folder_already_exists(self) -> None:
        """Resume GT export inside an existing folder, except when videos are explicitly disabled."""
        for flags in ([], ["--no-video"]):
            with self.subTest(flags=flags), patch("sys.argv", [
                    "scripts/run_all.py", "--tag", "resume", "--from", "gtcounts", "--to", "gtcounts", *flags]):
                run = run_all.Run(run_all.parse_args())
            (self.root / run.videos / "gt").mkdir(parents=True, exist_ok=True)
            with (patch.object(run_all, "ROOT", self.root), patch.object(run, "step") as step,
                  contextlib.redirect_stdout(io.StringIO())):
                run.run()
            commands = [call.args[1] for call in step.call_args_list if call.args[1][0] == "scripts/count_gt.py"]
            self.assertEqual(len(commands), 1)
            self.assertEqual("--vis-dir" in commands[0], not flags)

    def test_gt_stage_resumes_missing_video_after_interruption(self) -> None:
        """An interrupted second sequence is rebuilt while the first sequence's completed video is retained."""
        second = self.raw / "MVI_2"
        second.mkdir()
        for fid in range(1, FRAME_COUNT + 1):
            Image.new("RGB", (80, 40)).save(second / f"img{fid:05d}.jpg")
        annotations = {**self.anns, "MVI_2": SequenceAnn("MVI_2", {
            fid: FrameAnn(fid, []) for fid in range(1, FRAME_COUNT + 1)})}
        metadata = self.root / "metadata"
        write_json(metadata / "splits.json", {"dev": ["MVI_1", "MVI_2"], "test": []})
        write_json(metadata / "counting_lines.json", {"MVI_1": LINE, "MVI_2": LINE})
        videos = self.root / "videos"
        output = self.root / "gt_counts.json"
        argv = ["count_gt.py", "--splits", "dev", "--raw", str(self.raw),
                "--lines", str(metadata / "counting_lines.json"), "--vis-dir", str(videos),
                "--out", str(output), "--md", str(self.root / "gt_counts.md")]
        render = count_gt.write_gt_video

        def interrupt_second(ann, line, seq, events, path, **kwargs):
            if seq != "MVI_2":
                return render(ann, line, seq, events, path, **kwargs)
            reads = 0

            def interrupted(frame_path):
                nonlocal reads
                reads += 1
                if reads == 3:
                    raise KeyboardInterrupt()
                return read_frame(frame_path)

            with patch("src.tracking.viz.read_frame", side_effect=interrupted):
                return render(ann, line, seq, events, path, **kwargs)

        with (patch("sys.argv", argv), patch.object(count_gt, "REPORTS_DIR", metadata),
              patch.object(count_gt, "load_all", return_value=(annotations, {})),
              contextlib.redirect_stdout(io.StringIO())):
            with (patch.object(count_gt, "write_gt_video", side_effect=interrupt_second),
                  self.assertRaises(KeyboardInterrupt)):
                count_gt.main()
            first = videos / "gt_line_MVI_1.mp4"
            before = first.read_bytes(), first.stat().st_mtime_ns
            self.assertFalse((videos / "gt_line_MVI_2.mp4").exists())
            self.assertFalse(list(videos.glob("*.partial.mp4")))
            self.assertFalse(output.exists())

            count_gt.main()
        self.assertEqual((first.read_bytes(), first.stat().st_mtime_ns), before)
        self.assertTrue((videos / "gt_line_MVI_2.mp4").is_file())
        self.assertEqual(set(read_json(output)["dev"]), {"MVI_1", "MVI_2"})

    def test_failure_candidate_can_be_rendered_without_manual_review(self) -> None:
        """Render a failure case before any review file exists."""
        analysis = self.root / "analysis.json"
        ignored, lines = self.root / "ignored.json", self.root / "lines.json"
        write_json(analysis, {"split": "dev", "rows": [{
            "case_id": "case", "sequence": "MVI_1", "gt_event": {"frame": 2}}]})
        write_json(ignored, {"MVI_1": []})
        write_json(lines, {"MVI_1": LINE})
        argv = ["render_failure_cases.py", "--analysis", str(analysis), "--review", str(self.root / "absent.json"),
                "--case-id", "case", "--lines", str(lines), "--ignored", str(ignored),
                "--out-dir", str(self.root / "pictures")]

        with (patch("sys.argv", argv), patch.object(render_failure_cases, "build_view"),
              patch.object(render_failure_cases, "write_case") as write, contextlib.redirect_stdout(io.StringIO())):
            render_failure_cases.main()

        write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
