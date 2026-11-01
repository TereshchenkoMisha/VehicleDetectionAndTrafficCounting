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


if __name__ == "__main__":
    unittest.main()
