"""Tests for count errors, detector matching, TrackEval wrappers and the dev/test split."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.counting.metrics import count_errors
from src.data.detrac import SequenceMeta
from src.data.split import _quotas, select_and_split
from src.eval.detector import _ap, frame_pr
from src.eval.tracking import _build_trackeval_data, combine_sequence_metrics, evaluate_sequence


class CountErrorTests(unittest.TestCase):
    def test_values(self):
        m = count_errors([3, 5, 4], [4, 5, 2])
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["wape"], 3 / 11)
        self.assertEqual(m["sce"], [-1.0, 0.0, 2.0])
        self.assertAlmostEqual(m["over"], 1 / 3)
        self.assertAlmostEqual(m["under"], 1 / 3)

    def test_invalid_count_values(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                count_errors([value], [1])

    def test_zero_gt_and_bad_shapes(self):
        self.assertIsNone(count_errors([1], [0])["wape"])
        with self.assertRaises(ValueError):
            count_errors([1, 2], [1])
        with self.assertRaises(ValueError):
            count_errors([], [])


class DetectorMatchingTests(unittest.TestCase):
    def test_matching(self):
        self.assertIsNone(_ap(-1))
        self.assertIsNone(_ap(float("nan")))
        self.assertEqual(_ap(0), 0)
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        result = frame_pr(boxes, np.array([0, 0]), np.array([0.9, 0.8]), boxes[:1], np.array([0]))
        self.assertEqual(result[0], {"tp": 1, "fp": 1, "fn": 0})
        result = frame_pr(boxes[:1], np.array([1]), np.array([0.9]), boxes[:1], np.array([0]))
        self.assertEqual(result[0]["fn"], 1)
        self.assertEqual(result[1]["fp"], 1)

    def test_no_ground_truth_and_low_scores(self):
        boxes = np.array([[0, 0, 10, 10]])
        result = frame_pr(boxes, np.array([0]), np.array([0.9]), np.zeros((0, 4)), np.zeros(0))
        self.assertEqual(result[0], {"tp": 0, "fp": 1, "fn": 0})
        result = frame_pr(boxes, np.array([0]), np.array([0.1]), boxes, np.array([0]))
        self.assertEqual(result[0], {"tp": 0, "fp": 0, "fn": 1})


class TrackingMetricTests(unittest.TestCase):
    def test_empty_source_frames_are_preserved(self):
        target = (1, [0, 0, 10, 10], 0, 1)
        data = _build_trackeval_data({1: [target], 3: [target]}, {}, frame_ids=[1, 2, 3, 4])
        self.assertEqual(data["num_timesteps"], 4)
        self.assertEqual(len(data["gt_ids"][1]), 0)
        with self.assertRaises(ValueError):
            _build_trackeval_data({5: [target]}, {}, frame_ids=[1, 2])

    def test_per_class_tracking(self):
        with tempfile.TemporaryDirectory() as folder:
            gt, pred = Path(folder) / "gt.txt", Path(folder) / "pred.txt"
            gt.write_text("1,1,0,0,10,10,1,0\n1,2,20,0,10,10,1,1\n", encoding="utf-8")
            pred.write_text("1,5,0,0,10,10,1,0\n1,6,20,0,10,10,1,2\n", encoding="utf-8")
            result = evaluate_sequence(gt, pred, frame_ids=[1, 2], classes=["car", "bus", "truck"])
            self.assertEqual(set(result), {"all", "car", "bus", "truck"})
            self.assertAlmostEqual(result["all"]["HOTA"], 100)
            self.assertAlmostEqual(result["car"]["HOTA"], 100)
            self.assertEqual(result["bus"]["CLR_FN"], 1)
            self.assertEqual(result["truck"]["CLR_FP"], 1)

    def test_trackeval_combination(self):
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

    def test_ignored_regions_filter_both_sides_and_keep_frames(self):
        # Centre-in-region rule, border included: only the box at (9, 9) has its centre outside (0, 0, 10, 10)
        with tempfile.TemporaryDirectory() as folder:
            gt, pred = Path(folder) / "gt.txt", Path(folder) / "pred.txt"
            gt.write_text("1,1,9,9,10,10,1,0\n1,2,0,0,10,10,1,1\n2,2,5,5,10,10,1,1\n3,2,0,0,10,10,1,1\n",
                          encoding="utf-8")
            pred.write_text("1,10,9,9,10,10,1,0\n2,20,0,0,10,10,1,2\n3,20,5,5,10,10,1,2\n", encoding="utf-8")
            plain = evaluate_sequence(gt, pred, frame_ids=[1, 2, 3])["all"]
            self.assertEqual((plain["GT_dets"], plain["TRK_dets"]), (4, 3))

            metrics = evaluate_sequence(gt, pred, frame_ids=[1, 2, 3], ignored_regions=[(0, 0, 10, 10)])["all"]
            for name in ("HOTA", "MOTA", "IDF1"):
                self.assertAlmostEqual(metrics[name], 100)
            self.assertEqual((metrics["GT_dets"], metrics["TRK_dets"], metrics["frames"]), (1, 1, 3))


class SplitTests(unittest.TestCase):
    def test_quotas_add_up(self):
        for _ in range(50):
            groups = {s: list(range(random.randint(1, 20))) for s in ("day", "night", "unknown")}
            k = random.randint(1, sum(len(v) for v in groups.values()))
            quota = _quotas(groups, k)
            self.assertEqual(sum(quota.values()), k)
            self.assertTrue(all(quota[s] <= len(groups[s]) for s in groups))

    def test_split_is_deterministic_and_disjoint(self):
        metas = {f"MVI_{i}": SequenceMeta(f"MVI_{i}", weather="night" if i % 5 == 0 else "day") for i in range(40)}
        first, second = select_and_split(metas), select_and_split(metas)
        self.assertEqual(first, second)
        self.assertEqual((len(first["dev"]), len(first["test"])), (5, 7))
        self.assertFalse(set(first["dev"]) & set(first["test"]))




class VerificationInputTests(unittest.TestCase):
    def test_missing_gt_split_is_a_failed_check_instead_of_a_traceback(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import verify_run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "gt_counts.json").write_text(json.dumps({"test": {}}), encoding="utf-8")
            with patch.object(verify_run, "failures", []), contextlib.redirect_stdout(io.StringIO()) as output:
                verify_run.check_gt_counts(root, "dev")
                self.assertIn("GT counts include dev", verify_run.failures)
                self.assertIn("available splits", output.getvalue())


if __name__ == "__main__":
    unittest.main()
