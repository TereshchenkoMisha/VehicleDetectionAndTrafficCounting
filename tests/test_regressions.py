import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.data.detrac_parser import Box, FrameAnn, SequenceAnn, Target
from src.data.frames import frame_paths, resolve_naming
from src.data.mot import load_mot
from src.data.provenance import file_info, write_json
from src.eval.line_crossing import first_crossing, validate_line
from src.eval.run_validation import validate_tracking_run
from src.scripts.count_gt_crossings import count_sequence
from src.scripts.count_tracks_crossings import count_file


class CrossingTests(unittest.TestCase):
    def hit(self, coordinates, frames=None, gap=1):
        frames = range(1, len(coordinates) + 1) if frames is None else frames
        points = [(frame, x, y) for frame, (x, y) in zip(frames, coordinates)]
        return first_crossing(points, (100, 0), (100, 10), max_frame_gap=gap)

    def test_direct_crossing(self):
        self.assertEqual(self.hit([(90, 5), (110, 5)])[0], 2)

    def test_contact_then_crossing(self):
        self.assertEqual(self.hit([(90, 5), (100, 5), (110, 5)])[0], 3)

    def test_contact_then_return(self):
        self.assertIsNone(self.hit([(90, 5), (100, 5), (90, 5)]))

    def test_start_or_end_on_line(self):
        self.assertIsNone(self.hit([(100, 5), (110, 5)]))
        self.assertIsNone(self.hit([(90, 5), (100, 5)]))

    def test_contact_outside_segment(self):
        self.assertIsNone(self.hit([(90, 5), (100, 20), (110, 5)]))

    def test_endpoints_and_direction(self):
        forward = self.hit([(90, 0), (110, 0)])
        reverse = self.hit([(110, 0), (90, 0)])
        self.assertEqual(forward[1], -reverse[1])
        self.assertEqual(forward[2], (100, 0))

    def test_gap_policy(self):
        points = [(90, 5), (110, 5)]
        self.assertIsNone(self.hit(points, frames=[1, 20]))
        self.assertIsNotNone(self.hit(points, frames=[1, 20], gap=19))

    def test_resume_after_gap(self):
        self.assertEqual(self.hit([(90, 5), (90, 5), (110, 5)], frames=[1, 20, 21])[0], 21)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            validate_line((1, 1), (1, 1))
        with self.assertRaises(ValueError):
            self.hit([(90, 5), (110, 5)], frames=[2, 2])
        with self.assertRaises(ValueError):
            self.hit([(float("nan"), 5)])


class NumericTests(unittest.TestCase):
    def test_empty_source_frames_are_preserved(self):
        try:
            from src.eval.tracking_metrics import _build_trackeval_data
        except ImportError:
            self.skipTest("NumPy is unavailable")
        target = (1, [0, 0, 10, 10], 0, 1)
        data = _build_trackeval_data({1: [target], 3: [target]}, {}, frame_ids=[1, 2, 3, 4])
        self.assertEqual(data["num_timesteps"], 4)
        self.assertEqual(len(data["gt_ids"][1]), 0)
        with self.assertRaises(ValueError):
            _build_trackeval_data({5: [target]}, {}, frame_ids=[1, 2])

    def test_detector_matching(self):
        try:
            import numpy as np
            from src.eval.detector_eval import _ap, _frame_pr
        except ImportError:
            self.skipTest("NumPy is unavailable")
        self.assertIsNone(_ap(-1))
        self.assertIsNone(_ap(float("nan")))
        self.assertEqual(_ap(0), 0)
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        result = _frame_pr(boxes, np.array([0, 0]), np.array([0.9, 0.8]), boxes[:1], np.array([0]))
        self.assertEqual(result[0], {"tp": 1, "fp": 1, "fn": 0})
        result = _frame_pr(boxes[:1], np.array([1]), np.array([0.9]), boxes[:1], np.array([0]))
        self.assertEqual(result[0]["fn"], 1)
        self.assertEqual(result[1]["fp"], 1)

    def test_seed_repeats_random_values(self):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "cv2", "numpy")):
            self.skipTest("PyTorch OpenCV and NumPy are required")
        import random
        import numpy as np
        import torch
        from src.reproducibility import set_seed

        def sample():
            return random.random(), np.random.random(), torch.rand(4)

        set_seed(42)
        first = sample()
        set_seed(42)
        second = sample()
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_trackeval_combination(self):
        if importlib.util.find_spec("trackeval") is None:
            self.skipTest("TrackEval is unavailable")
        from src.eval.tracking_metrics import evaluate_sequence, combine_sequence_metrics
        with tempfile.TemporaryDirectory() as folder:
            gt, pred = Path(folder) / "gt.txt", Path(folder) / "pred.txt"
            gt.write_text("1,1,0,0,10,10,1,0\n", encoding="utf-8")
            pred.write_text(gt.read_text(encoding="utf-8"), encoding="utf-8")
            raw = {"first": {}, "second": {}}
            metrics = evaluate_sequence(gt, pred, frame_ids=[1, 2], raw_results=raw["first"])
            self.assertAlmostEqual(metrics["all"]["HOTA"], 100)
            self.assertEqual(metrics["all"]["frames"], 2)
            gt.write_text("".join(f"{fid},1,0,0,10,10,1,0\n" for fid in (1, 2, 3)), encoding="utf-8")
            pred.write_text("", encoding="utf-8")
            evaluate_sequence(gt, pred, frame_ids=[1, 2, 3], raw_results=raw["second"])
            combined = combine_sequence_metrics(raw)
            self.assertAlmostEqual(combined["MOTA"], 25)
            self.assertAlmostEqual(combined["IDF1"], 40)

    def test_circular_flow_direction(self):
        try:
            import numpy as np
            from scripts.place_lines import dominant_direction, clip_segment
        except ImportError:
            self.skipTest("NumPy is unavailable")
        import math
        streams = [{"ang": angle, "len": 100, "pts": np.zeros((5, 2))}
                   for angle in (0.01, math.pi - 0.01)]
        selected, angle = dominant_direction(streams)
        self.assertEqual(len(selected), 2)
        self.assertLess(min(angle, math.pi - angle), 0.02)
        with self.assertRaises(ValueError):
            dominant_direction([])
        a, b = clip_segment(np.array([-5., 0.]), np.array([15., 10.]), (11, 11))
        np.testing.assert_allclose(a, [0, 2.5])
        np.testing.assert_allclose(b, [10, 7.5])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def mot(self, text):
        path = self.root / "track.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_mot_validation(self):
        for text in ("1,1,0,0,10,10,1,0\n1,1,1,1,10,10,1,0\n",
                     "1,1,nan,0,10,10,1,0\n", "1,1,0,0,0,10,1,0\n",
                     "1,1,0,0,10,10,1,3\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_mot(self.mot(text))
        self.assertEqual(load_mot(self.mot("")), {})

    def test_frame_index_and_offset(self):
        (self.root / "img00001.JPEG").touch()
        (self.root / "IMG00002.jpg").touch()
        self.assertEqual(list(frame_paths(self.root)), [1, 2])
        self.assertEqual(resolve_naming(self.root, [0, 1]), 1)
        with self.assertRaises(ValueError):
            resolve_naming(self.root, [1])
        (self.root / "IMG1.jpg").touch()
        with self.assertRaises(ValueError):
            frame_paths(self.root)

    def test_atomic_json_rejects_nan(self):
        path = self.root / "out.json"
        write_json(path, {"valid": 1})
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            write_json(path, {"invalid": float("nan")})
        self.assertEqual(path.read_bytes(), original)

    def test_gt_and_prediction_ignore_observations_equally(self):
        ann = SequenceAnn("MVI_1", ignored_regions=[Box(99, 0, 2, 10)])
        for fid, x in enumerate((90, 100, 110), 1):
            target = Target(1, "car", Box(x - 1, 4, 2, 2), 0, 0)
            ann.frames[fid] = FrameAnn(fid, [target])
        line = {"p1": [100, 0], "p2": [100, 10]}
        path = self.mot("".join(f"{fid},1,{x - 1},4,2,2,1,0\n"
                                for fid, x in enumerate((90, 100, 110), 1)))
        for gap, expected in ((1, 0), (2, 1)):
            gt, _ = count_sequence(ann, line, max_frame_gap=gap)
            pred = count_file(path, line, [(99, 0, 101, 10)], max_frame_gap=gap)
            self.assertEqual(gt["exclude_ignored"], pred["counts"])
            self.assertEqual(pred["counts"]["all"]["total"], expected)
            self.assertEqual(gt["protocol"], pred["protocol"])

    def test_gt_mot_preflight(self):
        from unittest.mock import patch
        from src.data.convert_mot import convert_all
        sequence = self.root / "MVI_1"
        sequence.mkdir()
        (sequence / "IMG00001.jpg").touch()
        ignored = self.root / "ignored.json"
        write_json(ignored, {"MVI_1": []})
        ann = SequenceAnn("MVI_1", frames={1: FrameAnn(1, [])})
        with patch("src.data.convert_mot.load_all", return_value=({"MVI_1": ann}, {})):
            with self.assertRaises(FileNotFoundError):
                convert_all(["MVI_1", "MVI_2"], self.root / "out", self.root, ignored)
        self.assertFalse((self.root / "out").exists())

    def test_gt_video_keeps_unannotated_frames(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is unavailable")
        from unittest.mock import MagicMock, patch
        from src.scripts.count_gt_crossings import write_video
        sequence = self.root / "MVI_1"
        sequence.mkdir()
        for fid in (1, 2, 3):
            (sequence / f"IMG{fid:05d}.jpg").touch()
        ann = SequenceAnn("MVI_1", frames={1: FrameAnn(1, []), 3: FrameAnn(3, [])})
        cv2 = MagicMock()
        writer = cv2.VideoWriter.return_value
        writer.isOpened.return_value = True
        progress = MagicMock()
        progress.tqdm = lambda items, **kwargs: items
        line = {"p1": [10, 0], "p2": [10, 9]}
        with patch.dict("sys.modules", {"cv2": cv2, "tqdm": progress}):
            with patch("src.scripts.count_gt_crossings.read_frame", return_value=np.zeros((10, 20, 3))):
                write_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            self.assertEqual(writer.write.call_count, 3)
            writer.release.assert_called_once()
            writer.reset_mock()
            with patch("src.scripts.count_gt_crossings.read_frame", side_effect=[
                    np.zeros((10, 20, 3)), np.zeros((10, 20, 3)), RuntimeError("Unreadable image")]):
                with self.assertRaisesRegex(RuntimeError, "Unreadable image"):
                    write_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            writer.release.assert_called_once()

    def test_run_validation_and_fingerprint(self):
        sequence = self.root / "raw" / "MVI_1"
        sequence.mkdir(parents=True)
        for fid in (1, 2):
            (sequence / f"IMG{fid:05d}.jpg").touch()
        cfg = self.root / "run"
        cfg.mkdir()
        mot = cfg / "MVI_1.txt"
        mot.write_text("1,1,0,0,10,10,1,0\n", encoding="utf-8")
        timing = {"split": "dev", "frames": 2,
                  "per_seq": {"MVI_1": {"frames": 2, "rows": 1,
                                          "mot_sha256": file_info(mot)["sha256"]}}}
        write_json(cfg / "timing.json", timing)
        validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")
        mot.write_text("2,1,0,0,10,10,1,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")
        timing["per_seq"]["MVI_1"].pop("mot_sha256")
        timing["per_seq"]["MVI_1"]["frames"] = 1
        write_json(cfg / "timing.json", timing)
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")


if __name__ == "__main__":
    unittest.main()
