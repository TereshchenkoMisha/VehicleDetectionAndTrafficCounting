"""Tests for the failure analysis: box and event pairing, outcomes, summary, case selection, review checks.

Tests:
    BoxMatchingTests
        test_iou                        IoU of boxes against boxes
        test_pairing_is_one_to_one      the better overlap wins and the competing box makes the pair ambiguous
        test_overlap_below_threshold    boxes that overlap too little are not paired
    EventMatchingTests
        test_ids_and_classes_may_differ   a GT and a predicted crossing pair even with other ids and classes
        test_direction_time_and_support   direction, time distance and box support each reject a pair
        test_ambiguous_boxes_are_not_support   ambiguous box matches do not support a pair
        test_one_gt_event_explains_one_prediction   a GT crossing pairs with one prediction; the pair is ambiguous
    DiagnosisTests
        test_outcomes                   matched, missed, extra and class-confusion candidates
        test_gap_and_duplicate_track_flags   evidence flags for a gap across the line and several track ids
        test_summary_denominators       counts of GT events, predicted events, paired, missed and extra
        test_saved_events               saved event lists become dictionaries with named fields
        test_suggestion                 cases that another configuration solved are suggested first
        test_peers                      peers follow the same GT vehicle across configurations
    ReviewTests
        test_valid_review               a valid review passes and the report links the pictures
        test_assessment_must_match_saved_events   a review that contradicts the saved events is rejected
        test_needs_text_and_three_vehicles   empty fields and fewer than three vehicles are rejected
    DrawingHelperTests
        test_crop                       the crop has a margin and stays inside the image
        test_tile                       the last row of a tile is padded
"""

import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.failure.diagnosis import classify_events, peers_of, saved_events, suggest_cases, summarize
from src.failure.matching import MatchSettings, iou_matrix, match_boxes, match_events
from src.failure.review import check_review, markdown_report
from src.failure.viz import crop_around, tile

SETTINGS = MatchSettings()


def event(target: int = 1, frame: int = 10, cls: int = 0, direction: int = 1) -> dict[str, int]:
    """One crossing event as the counter saves it."""
    return {"target": target, "frame": frame, "cls_id": cls, "direction": direction}


def box_matches(track: int = 99, frames: tuple[int, ...] = (8, 9, 10, 11), ambiguous: bool = False,
                target: int = 1) -> dict[int, list[dict[str, Any]]]:
    """Box pairs between the GT vehicle `target` and the tracker track `track` in the given frames."""
    return {target: [{"frame": f, "track": track, "iou": 0.8, "ambiguous": ambiguous} for f in frames]}


class BoxMatchingTests(unittest.TestCase):
    """Pairing of GT boxes with tracker boxes."""

    def test_iou(self) -> None:
        """Compute IoU of one box against two others."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        other = np.array([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])

        np.testing.assert_allclose(iou_matrix(boxes, other), [[1 / 3, 0.0]])

    def test_pairing_is_one_to_one(self) -> None:
        """Pair a GT box with the better overlap and flag that another box could have been the partner."""
        gt = {1: [(7, [0, 0, 10, 10], 0, 1.0)]}
        pred = {1: [(100, [0, 0, 10, 10], 0, 0.9), (101, [1, 0, 11, 10], 0, 0.8)]}

        [match] = match_boxes(gt, pred, 0.3)[7]

        self.assertEqual(match["track"], 100)
        self.assertTrue(match["ambiguous"])

    def test_overlap_below_threshold(self) -> None:
        """Leave boxes unpaired when their overlap is below the threshold."""
        gt = {1: [(7, [0, 0, 10, 10], 0, 1.0)]}
        pred = {1: [(100, [9, 9, 19, 19], 0, 0.9)]}

        self.assertEqual(match_boxes(gt, pred, 0.3), {})


class EventMatchingTests(unittest.TestCase):
    """Pairing of GT crossings with predicted crossings."""

    def test_ids_and_classes_may_differ(self) -> None:
        """Pair crossings of different track ids and classes when the boxes support it."""
        pairs = match_events([event()], [event(99, cls=2)], box_matches(), SETTINGS)

        self.assertEqual(pairs[0]["pred"], 0)
        self.assertEqual(pairs[0]["support_frames"], 4)

    def test_direction_time_and_support(self) -> None:
        """Reject a pair with another direction, a late crossing, or too little box support."""
        for predicted in (event(99, direction=-1), event(99, frame=25)):
            self.assertEqual(match_events([event()], [predicted], box_matches(), SETTINGS), {})
        self.assertEqual(match_events([event()], [event(99)], box_matches(frames=(10,)), SETTINGS), {})

    def test_ambiguous_boxes_are_not_support(self) -> None:
        """Reject a pair whose only box support is ambiguous."""
        self.assertEqual(match_events([event()], [event(99)], box_matches(ambiguous=True), SETTINGS), {})

    def test_one_gt_event_explains_one_prediction(self) -> None:
        """Keep one pair when two predictions fit one GT crossing, and mark it ambiguous."""
        matches = box_matches()
        matches[1] += box_matches(track=100, frames=(7, 12))[1]

        pairs = match_events([event()], [event(99), event(100)], matches, SETTINGS)

        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["ambiguous"])


class DiagnosisTests(unittest.TestCase):
    """Outcomes, evidence flags, summaries and case selection."""

    def rows(self, gt: list[dict[str, int]], pred: list[dict[str, int]], matches: dict | None = None) -> list[Any]:
        """Classify the events of sequence MVI_1, configuration "cfg"."""
        return classify_events("MVI_1", "cfg", gt, pred, matches or {}, SETTINGS, max_frame_gap=1)

    def test_outcomes(self) -> None:
        """Label every GT and predicted crossing as matched, missed, extra or class confusion."""
        gt = [event(1, 10), event(2, 50), event(3, 90)]
        pred = [event(91, 10, cls=2), event(92, 90), event(93, 200)]
        matches = {**box_matches(91, target=1), **box_matches(92, frames=(88, 89, 90), target=3)}

        outcomes = {row["case_id"]: row["outcome"] for row in self.rows(gt, pred, matches)}

        self.assertEqual(outcomes, {"MVI_1__cfg__gt1__f10": "class_confusion_candidate",
                                    "MVI_1__cfg__gt2__f50": "missed_candidate",
                                    "MVI_1__cfg__gt3__f90": "matched",
                                    "MVI_1__cfg__pred93__f200": "extra_candidate"})

    def test_gap_and_duplicate_track_flags(self) -> None:
        """Flag a gap in the matched boxes across the line and several track ids near the crossing."""
        matches = {1: [{"frame": f, "track": 5, "iou": 0.8, "ambiguous": False} for f in (5, 6, 11, 12)]
                   + [{"frame": 10, "track": 6, "iou": 0.8, "ambiguous": False}]}

        [row] = self.rows([event(1, 10)], [], matches)

        self.assertEqual(row["outcome"], "missed_candidate")
        self.assertIn("matched_observation_gap_across_line", row["flags"])
        self.assertIn("multiple_track_ids_in_context", row["flags"])

    def test_summary_denominators(self) -> None:
        """Count GT events, predicted events, pairs, misses and extras over all sequences."""
        rows = self.rows([event(1, 10), event(2, 50)], [event(91, 10), event(93, 200)], box_matches(91))

        [summary] = [r for r in summarize(rows) if r["sequence"] == "ALL"]

        self.assertEqual((summary["gt_events"], summary["pred_events"]), (2, 2))
        self.assertEqual((summary["paired"], summary["missed"], summary["extra"]), (1, 1, 1))
        self.assertEqual(summary["signed_error"], 0)

    def test_saved_events(self) -> None:
        """Turn a saved event list into a dictionary with named fields."""
        self.assertEqual(saved_events([[12, 3, 0, -1]]), [{"frame": 12, "target": 3, "cls_id": 0, "direction": -1}])

    def test_suggestion(self) -> None:
        """Suggest first a vehicle that another configuration counted correctly."""
        solved = classify_events("MVI_1", "a", [event(1, 10), event(2, 50)], [event(91, 10)], box_matches(91),
                                 SETTINGS, 1)
        failed = classify_events("MVI_1", "b", [event(1, 10), event(2, 50)], [], {}, SETTINGS, 1)

        self.assertEqual(suggest_cases(solved + failed, limit=1), ["MVI_1__b__gt1__f10"])

    def test_peers(self) -> None:
        """Find the configurations that failed on the same GT vehicle."""
        rows = [*classify_events("MVI_1", "a", [event(1, 10)], [], {}, SETTINGS, 1),
                *classify_events("MVI_1", "b", [event(1, 10)], [], {}, SETTINGS, 1),
                *classify_events("MVI_1", "c", [event(2, 10)], [], {}, SETTINGS, 1)]

        self.assertEqual(set(peers_of(rows, rows[0])), {"a", "b"})


class ReviewTests(unittest.TestCase):
    """Checks of the manual review file against the saved analysis."""

    def setUp(self) -> None:
        """Three missed crossings of one configuration; the predictions use the saved `count_tracks` format."""
        self.rows = [row for target in (1, 2, 3)
                     for row in classify_events("MVI_1", "cfg", [event(target, 10 * target)], [], {}, SETTINGS, 1)]
        self.predictions = {"cfg": {"MVI_1": {"events": [[10, 7, 0, 1], [20, 8, 2, 1]]}}}

    def case(self, row: dict[str, Any], counted: dict[str, Any] | None = None) -> dict[str, Any]:
        """A review entry for `row`; by default the counterpart counted nothing."""
        counted = counted or {"counted_track_ids": [], "counted_events": 0, "counted_classes": [],
                              "description": "missed"}
        return {"case_id": row["case_id"], "label": "case", "observed_failure": "a", "likely_cause": "b",
                "cause_confidence": "c", "counting_impact": "d", "limitations": "e",
                "counterpart_assessment": {"cfg": counted}}

    def test_valid_review(self) -> None:
        """Accept a complete review and link the comparison pictures in its report."""
        cases = [self.case(row) for row in self.rows]

        self.assertEqual(len(check_review(cases, self.rows, self.predictions)), 3)
        report = markdown_report(cases, self.rows, Path("pictures"), Path("reports/failures.md"))

        self.assertIn("![comparison](../pictures/MVI_1__cfg__gt1__f10/comparison.png)", report)

    def test_assessment_must_match_saved_events(self) -> None:
        """Reject a review that says a track counted a crossing the saved events do not contain."""
        wrong = {"counted_track_ids": [8], "counted_events": 1, "counted_classes": ["car"], "description": "x"}
        cases = [self.case(self.rows[0], wrong), self.case(self.rows[1]), self.case(self.rows[2])]

        with self.assertRaisesRegex(ValueError, "contradicts"):
            check_review(cases, self.rows, self.predictions)

    def test_needs_text_and_three_vehicles(self) -> None:
        """Reject an empty field and a review of fewer than three vehicles."""
        cases = [self.case(row) for row in self.rows]
        cases[0]["likely_cause"] = " "

        with self.assertRaisesRegex(ValueError, "empty review fields"):
            check_review(cases, self.rows, self.predictions)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            check_review(cases[1:], self.rows, self.predictions)


class DrawingHelperTests(unittest.TestCase):
    """Image helpers of the failure pictures."""

    def test_crop(self) -> None:
        """Add a margin around the boxes and keep the crop inside the image."""
        self.assertEqual(crop_around([[40, 40, 60, 60]], (100, 200), margin=10), (30, 30, 70, 70))
        self.assertEqual(crop_around([[5, 5, 195, 95]], (100, 200), margin=10), (0, 0, 200, 100))
        self.assertEqual(crop_around([], (100, 200)), (0, 0, 200, 100))

    def test_tile(self) -> None:
        """Pad the last row of a tile so all rows have the same width."""
        panels = [np.full((2, 3, 3), v, np.uint8) for v in (1, 2, 3)]

        self.assertEqual(tile(panels, 2).shape, (4, 6, 3))
        self.assertEqual(tile(panels, 3).shape, (2, 9, 3))


if __name__ == "__main__":
    unittest.main()
