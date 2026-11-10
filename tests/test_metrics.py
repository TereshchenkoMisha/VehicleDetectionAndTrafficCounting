"""Tests for count errors, detector matching, TrackEval wrappers, the dev/test split and the verification script.

Tests:
    CountErrorTests
        test_values                     MAE, WAPE, signed errors and over/under shares on a small example
        test_invalid_count_values       negative, NaN and infinite counts are rejected
        test_zero_gt_and_bad_shapes     WAPE is None for zero GT; empty or unequal inputs are rejected
    DetectorMatchingTests
        test_ap_and_frame_matching      AP rejects invalid values; boxes pair one-to-one into TP, FP and FN
        test_no_ground_truth_and_low_scores   every box is a false positive without GT; low scores do not count
    TrackingMetricTests
        test_empty_source_frames_are_preserved   frames without objects stay in the TrackEval data
        test_per_class_tracking         TrackEval metrics are computed per class and for all classes
        test_trackeval_combination      per-sequence results combine into dataset-level MOTA and IDF1
        test_ignored_regions_filter_both_sides   ignored boxes are removed from GT and tracks; empty frames stay
    SplitTests
        test_quotas_add_up              stratum quotas sum to the requested number and fit the strata
        test_split_is_deterministic_and_disjoint   the same input gives the same 5 + 7 split with no overlap
    VerificationInputTests
        test_missing_gt_split           a GT file without the requested split is a failed check, not a traceback
"""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import verify_run
from src.counting.metrics import count_errors
from src.data.detrac import SequenceMeta
from src.data.split import _quotas, select_and_split
from src.eval.detector import _ap, frame_pr
from src.eval.tracking import _build_trackeval_data, combine_sequence_metrics, evaluate_sequence

BOX = np.array([[0, 0, 10, 10]])        # one 10 x 10 box, shared by the detector matching tests


class CountErrorTests(unittest.TestCase):
    """Counting error metrics over per-sequence totals."""

    def test_values(self) -> None:
        """Compute every metric for predicted [3, 5, 4] against GT [4, 5, 2]."""
        m = count_errors([3, 5, 4], [4, 5, 2])

        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["wape"], 3 / 11)
        self.assertEqual(m["sce"], [-1.0, 0.0, 2.0])
        self.assertAlmostEqual(m["over"], 1 / 3)
        self.assertAlmostEqual(m["under"], 1 / 3)

    def test_invalid_count_values(self) -> None:
        """Reject negative, NaN and infinite counts."""
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                count_errors([value], [1])

    def test_zero_gt_and_bad_shapes(self) -> None:
        """Return no WAPE for zero GT and reject unequal or empty inputs."""
        self.assertIsNone(count_errors([1], [0])["wape"])
        with self.assertRaises(ValueError):
            count_errors([1, 2], [1])
        with self.assertRaises(ValueError):
            count_errors([], [])


class DetectorMatchingTests(unittest.TestCase):
    """Box pairing used for precision and recall."""

    def test_ap_and_frame_matching(self) -> None:
        """Reject an invalid AP value and pair boxes one-to-one."""
        self.assertIsNone(_ap(-1))
        self.assertIsNone(_ap(float("nan")))
        self.assertEqual(_ap(0), 0)

        # Two identical predictions for one GT box: one true positive, one false positive
        two_boxes = np.repeat(BOX, 2, axis=0)
        result = frame_pr(two_boxes, np.array([0, 0]), np.array([0.9, 0.8]), BOX, np.array([0]))
        self.assertEqual(result[0], {"tp": 1, "fp": 1, "fn": 0})

        # A prediction of the wrong class leaves the GT box unmatched
        result = frame_pr(BOX, np.array([1]), np.array([0.9]), BOX, np.array([0]))
        self.assertEqual(result[0]["fn"], 1)
        self.assertEqual(result[1]["fp"], 1)

    def test_no_ground_truth_and_low_scores(self) -> None:
        """Count a box without GT as a false positive and ignore boxes below the score threshold."""
        result = frame_pr(BOX, np.array([0]), np.array([0.9]), np.zeros((0, 4)), np.zeros(0))
        self.assertEqual(result[0], {"tp": 0, "fp": 1, "fn": 0})

        result = frame_pr(BOX, np.array([0]), np.array([0.1]), BOX, np.array([0]))
        self.assertEqual(result[0], {"tp": 0, "fp": 0, "fn": 1})


class TrackingMetricTests(unittest.TestCase):
    """TrackEval wrappers on tiny MOT files."""

    def test_empty_source_frames_are_preserved(self) -> None:
        """Keep frames without objects and reject objects outside the frame list."""
        target = (1, [0, 0, 10, 10], 0, 1)

        data = _build_trackeval_data({1: [target], 3: [target]}, {}, frame_ids=[1, 2, 3, 4])

        self.assertEqual(data["num_timesteps"], 4)
        self.assertEqual(len(data["gt_ids"][1]), 0)
        with self.assertRaises(ValueError):
            _build_trackeval_data({5: [target]}, {}, frame_ids=[1, 2])

    def test_per_class_tracking(self) -> None:
        """Report metrics per class: a missed bus is a false negative, a wrong-class truck a false positive."""
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

    def test_trackeval_combination(self) -> None:
        """Combine a perfect sequence and a fully missed one into dataset-level MOTA and IDF1."""
        with tempfile.TemporaryDirectory() as folder:
            gt, pred = Path(folder) / "gt.txt", Path(folder) / "pred.txt"
            raw: dict[str, dict] = {"first": {}, "second": {}}

            # A perfect sequence of 1 object in 1 of 2 frames
            gt.write_text("1,1,0,0,10,10,1,0\n", encoding="utf-8")
            pred.write_text(gt.read_text(encoding="utf-8"), encoding="utf-8")
            metrics = evaluate_sequence(gt, pred, frame_ids=[1, 2], raw_results=raw["first"])
            self.assertAlmostEqual(metrics["all"]["HOTA"], 100)
            self.assertEqual(metrics["all"]["frames"], 2)

            # A sequence where the tracker output is empty
            gt.write_text("".join(f"{fid},1,0,0,10,10,1,0\n" for fid in (1, 2, 3)), encoding="utf-8")
            pred.write_text("", encoding="utf-8")
            evaluate_sequence(gt, pred, frame_ids=[1, 2, 3], raw_results=raw["second"])

        combined = combine_sequence_metrics(raw)

        self.assertAlmostEqual(combined["MOTA"], 25)
        self.assertAlmostEqual(combined["IDF1"], 40)

    def test_ignored_regions_filter_both_sides(self) -> None:
        """Remove boxes whose centre is in an ignored region from GT and tracks, and keep all frames."""
        # Centre-in-region rule, border included: only the box at (9, 9) has its centre outside (0, 0, 10, 10)
        with tempfile.TemporaryDirectory() as folder:
            gt, pred = Path(folder) / "gt.txt", Path(folder) / "pred.txt"
            gt.write_text("1,1,9,9,10,10,1,0\n1,2,0,0,10,10,1,1\n2,2,5,5,10,10,1,1\n3,2,0,0,10,10,1,1\n",
                          encoding="utf-8")
            pred.write_text("1,10,9,9,10,10,1,0\n2,20,0,0,10,10,1,2\n3,20,5,5,10,10,1,2\n", encoding="utf-8")

            plain = evaluate_sequence(gt, pred, frame_ids=[1, 2, 3])["all"]
            filtered = evaluate_sequence(gt, pred, frame_ids=[1, 2, 3], ignored_regions=[(0, 0, 10, 10)])["all"]

        self.assertEqual((plain["GT_dets"], plain["TRK_dets"]), (4, 3))
        for name in ("HOTA", "MOTA", "IDF1"):
            self.assertAlmostEqual(filtered[name], 100)
        self.assertEqual((filtered["GT_dets"], filtered["TRK_dets"], filtered["frames"]), (1, 1, 3))


class SplitTests(unittest.TestCase):
    """The stratified dev/test split."""

    def test_quotas_add_up(self) -> None:
        """Give quotas that sum to the requested number and never exceed a stratum."""
        rng = random.Random(42)
        for _ in range(50):
            groups = {s: list(range(rng.randint(1, 20))) for s in ("day", "night", "unknown")}
            k = rng.randint(1, sum(len(v) for v in groups.values()))

            quota = _quotas(groups, k)

            self.assertEqual(sum(quota.values()), k)
            self.assertTrue(all(quota[s] <= len(groups[s]) for s in groups))

    def test_split_is_deterministic_and_disjoint(self) -> None:
        """Repeat the same 5 dev + 7 test split from the same input, with no sequence in both."""
        metas = {f"MVI_{i}": SequenceMeta(f"MVI_{i}", weather="night" if i % 5 == 0 else "day") for i in range(40)}

        first, second = select_and_split(metas), select_and_split(metas)

        self.assertEqual(first, second)
        self.assertEqual((len(first["dev"]), len(first["test"])), (5, 7))
        self.assertFalse(set(first["dev"]) & set(first["test"]))


class VerificationInputTests(unittest.TestCase):
    """Input checks of the run verification script."""

    def test_missing_gt_split(self) -> None:
        """Report a GT file without the requested split as a failed check."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "gt_counts.json").write_text(json.dumps({"test": {}}), encoding="utf-8")

            with patch.object(verify_run, "failures", []), contextlib.redirect_stdout(io.StringIO()) as output:
                verify_run.check_gt_counts(root, "dev")

                self.assertIn("GT counts include dev", verify_run.failures)
                self.assertIn("available splits", output.getvalue())


if __name__ == "__main__":
    unittest.main()
