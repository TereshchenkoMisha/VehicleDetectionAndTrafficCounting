"""Tests for the failure analysis: box and event pairing, outcomes, summary, case selection, review checks."""

import unittest
from pathlib import Path

import numpy as np

from src.failure.diagnosis import classify_events, peers_of, saved_events, suggest_cases, summarize
from src.failure.matching import MatchSettings, iou_matrix, match_boxes, match_events
from src.failure.review import check_review, markdown_report
from src.failure.viz import crop_around, tile

SETTINGS = MatchSettings()


def event(target=1, frame=10, cls=0, direction=1):
    return {"target": target, "frame": frame, "cls_id": cls, "direction": direction}


def box_matches(track=99, frames=(8, 9, 10, 11), ambiguous=False, target=1):
    return {target: [{"frame": f, "track": track, "iou": 0.8, "ambiguous": ambiguous} for f in frames]}


class BoxMatchingTests(unittest.TestCase):
    def test_iou(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        other = np.array([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        np.testing.assert_allclose(iou_matrix(boxes, other), [[1 / 3, 0.0]])

    def test_pairing_is_one_to_one_and_flags_ambiguity(self):
        gt = {1: [(7, [0, 0, 10, 10], 0, 1.0)]}
        pred = {1: [(100, [0, 0, 10, 10], 0, 0.9), (101, [1, 0, 11, 10], 0, 0.8)]}
        [match] = match_boxes(gt, pred, 0.3)[7]
        self.assertEqual(match["track"], 100)                # the better overlap wins, the other box stays unpaired
        self.assertTrue(match["ambiguous"])                  # but both boxes could have been the partner

    def test_overlap_below_threshold_is_not_paired(self):
        gt = {1: [(7, [0, 0, 10, 10], 0, 1.0)]}
        pred = {1: [(100, [9, 9, 19, 19], 0, 0.9)]}
        self.assertEqual(match_boxes(gt, pred, 0.3), {})


class EventMatchingTests(unittest.TestCase):
    def test_ids_and_classes_do_not_need_to_equal(self):
        pairs = match_events([event()], [event(99, cls=2)], box_matches(), SETTINGS)
        self.assertEqual(pairs[0]["pred"], 0)
        self.assertEqual(pairs[0]["support_frames"], 4)

    def test_direction_time_and_support_are_gates(self):
        for predicted in (event(99, direction=-1), event(99, frame=25)):
            self.assertEqual(match_events([event()], [predicted], box_matches(), SETTINGS), {})
        self.assertEqual(match_events([event()], [event(99)], box_matches(frames=(10,)), SETTINGS), {})

    def test_ambiguous_boxes_are_not_support(self):
        self.assertEqual(match_events([event()], [event(99)], box_matches(ambiguous=True), SETTINGS), {})

    def test_one_gt_event_explains_one_prediction(self):
        matches = box_matches()
        matches[1] += box_matches(track=100, frames=(7, 12))[1]
        pairs = match_events([event()], [event(99), event(100)], matches, SETTINGS)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["ambiguous"])


class DiagnosisTests(unittest.TestCase):
    def rows(self, gt, pred, matches=None):
        return classify_events("MVI_1", "cfg", gt, pred, matches or {}, SETTINGS, max_frame_gap=1)

    def test_outcomes(self):
        gt = [event(1, 10), event(2, 50), event(3, 90)]
        pred = [event(91, 10, cls=2), event(92, 90), event(93, 200)]
        matches = {**box_matches(91, target=1), **box_matches(92, frames=(88, 89, 90), target=3)}
        outcomes = {row["case_id"]: row["outcome"] for row in self.rows(gt, pred, matches)}
        self.assertEqual(outcomes, {"MVI_1__cfg__gt1__f10": "class_confusion_candidate",
                                    "MVI_1__cfg__gt2__f50": "missed_candidate",
                                    "MVI_1__cfg__gt3__f90": "matched",
                                    "MVI_1__cfg__pred93__f200": "extra_candidate"})

    def test_gap_and_duplicate_track_flags(self):
        matches = {1: [{"frame": f, "track": 5, "iou": 0.8, "ambiguous": False} for f in (5, 6, 11, 12)]
                   + [{"frame": 10, "track": 6, "iou": 0.8, "ambiguous": False}]}
        [row] = self.rows([event(1, 10)], [], matches)
        self.assertEqual(row["outcome"], "missed_candidate")
        self.assertIn("matched_observation_gap_across_line", row["flags"])
        self.assertIn("multiple_track_ids_in_context", row["flags"])

    def test_summary_denominators(self):
        rows = self.rows([event(1, 10), event(2, 50)], [event(91, 10), event(93, 200)], box_matches(91))
        [summary] = [r for r in summarize(rows) if r["sequence"] == "ALL"]
        self.assertEqual((summary["gt_events"], summary["pred_events"]), (2, 2))
        self.assertEqual((summary["paired"], summary["missed"], summary["extra"]), (1, 1, 1))
        self.assertEqual(summary["signed_error"], 0)

    def test_saved_events_have_the_event_fields(self):
        self.assertEqual(saved_events([[12, 3, 0, -1]]), [{"frame": 12, "target": 3, "cls_id": 0, "direction": -1}])

    def test_suggestion_prefers_vehicles_another_configuration_counted(self):
        solved = classify_events("MVI_1", "a", [event(1, 10), event(2, 50)], [event(91, 10)], box_matches(91),
                                 SETTINGS, 1)
        failed = classify_events("MVI_1", "b", [event(1, 10), event(2, 50)], [], {}, SETTINGS, 1)
        self.assertEqual(suggest_cases(solved + failed, limit=1), ["MVI_1__b__gt1__f10"])

    def test_peers_follow_the_gt_vehicle_across_configurations(self):
        rows = [*classify_events("MVI_1", "a", [event(1, 10)], [], {}, SETTINGS, 1),
                *classify_events("MVI_1", "b", [event(1, 10)], [], {}, SETTINGS, 1),
                *classify_events("MVI_1", "c", [event(2, 10)], [], {}, SETTINGS, 1)]
        self.assertEqual(set(peers_of(rows, rows[0])), {"a", "b"})


class ReviewTests(unittest.TestCase):
    def setUp(self):
        """Three configurations count three vehicles; the events are the saved `count_tracks` format."""
        self.rows = [row for target in (1, 2, 3)
                     for row in classify_events("MVI_1", "cfg", [event(target, 10 * target)], [], {}, SETTINGS, 1)]
        self.predictions = {"cfg": {"MVI_1": {"events": [[10, 7, 0, 1], [20, 8, 2, 1]]}}}

    def case(self, row, counted=None):
        counted = counted or {"counted_track_ids": [], "counted_events": 0, "counted_classes": [],
                              "description": "missed"}
        return {"case_id": row["case_id"], "label": "case", "observed_failure": "a", "likely_cause": "b",
                "cause_confidence": "c", "counting_impact": "d", "limitations": "e",
                "counterpart_assessment": {"cfg": counted}}

    def test_valid_review_passes_and_is_reported(self):
        cases = [self.case(row) for row in self.rows]
        self.assertEqual(len(check_review(cases, self.rows, self.predictions)), 3)
        report = markdown_report(cases, self.rows, Path("pictures"), Path("reports/failures.md"))
        self.assertIn("![comparison](../pictures/MVI_1__cfg__gt1__f10/comparison.png)", report)

    def test_assessment_must_match_saved_events(self):
        wrong = {"counted_track_ids": [8], "counted_events": 1, "counted_classes": ["car"], "description": "x"}
        cases = [self.case(self.rows[0], wrong), self.case(self.rows[1]), self.case(self.rows[2])]
        with self.assertRaisesRegex(ValueError, "contradicts"):
            check_review(cases, self.rows, self.predictions)

    def test_needs_text_and_three_vehicles(self):
        cases = [self.case(row) for row in self.rows]
        cases[0]["likely_cause"] = " "
        with self.assertRaisesRegex(ValueError, "empty review fields"):
            check_review(cases, self.rows, self.predictions)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            check_review(cases[1:], self.rows, self.predictions)


class DrawingHelperTests(unittest.TestCase):
    def test_crop_has_margin_and_stays_in_the_image(self):
        self.assertEqual(crop_around([[40, 40, 60, 60]], (100, 200), margin=10), (30, 30, 70, 70))
        self.assertEqual(crop_around([[5, 5, 195, 95]], (100, 200), margin=10), (0, 0, 200, 100))
        self.assertEqual(crop_around([], (100, 200)), (0, 0, 200, 100))

    def test_tile_pads_the_last_row(self):
        panels = [np.full((2, 3, 3), v, np.uint8) for v in (1, 2, 3)]
        self.assertEqual(tile(panels, 2).shape, (4, 6, 3))
        self.assertEqual(tile(panels, 3).shape, (2, 9, 3))


if __name__ == "__main__":
    unittest.main()
