"""Tests for line-crossing detection, GT/track counting parity and line placement helpers.

Tests:
    CrossingTests                       one trajectory against the vertical segment x=100, y=0..10
        test_direct_crossing            a step across the line is counted at the first frame beyond it
        test_contact_then_crossing      touching the line and then leaving on the far side is a crossing
        test_contact_then_return        touching the line and going back is not a crossing
        test_start_or_end_on_line       a trajectory that starts or ends exactly on the line is not decided
        test_contact_outside_segment    touching the infinite line outside the segment is not a crossing
        test_endpoints_and_direction    a crossing at the segment end counts; opposite steps give opposite directions
        test_gap_policy                 a crossing inside a frame gap is lost unless the gap is allowed
        test_resume_after_gap           counting resumes with the points that follow a gap
        test_only_first_crossing        a target that crosses twice produces one event
        test_tally                      events are counted per class and direction, with an "all" total
        test_invalid_line               a line with equal endpoints is rejected
        test_repeated_frames            frames that do not increase are rejected
        test_nan_coordinates            non-finite coordinates are rejected
    GtAndTrackCountingTests
        test_gt_and_tracks_agree        GT and tracker counting use the same ignored-region and gap rules
    LinePlacementTests
        test_dominant_axis_wraps        opposite travel directions belong to one axis; no moving targets is an error
        test_clip_segment               a segment is cut to the image
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.counting.line_crossing import Point, count_events, first_crossing, tally, validate_line
from src.counting.lines import clip_segment, dominant_direction
from src.counting.trajectories import count_gt, count_tracks
from src.data.detrac import Box, FrameAnn, SequenceAnn, Target

LINE_START: Point = (100, 0)        # the counting segment in the crossing tests: vertical, x=100, y from 0 to 10
LINE_END: Point = (100, 10)


class CrossingTests(unittest.TestCase):
    """Crossings of one trajectory with a fixed segment."""

    def hit(self, coordinates: list[Point], frames: list[int] | None = None, gap: int = 1
            ) -> tuple[int, int, Point] | None:
        """Run `first_crossing` on (x, y) points; frames default to 1, 2, 3, ..."""
        frames = range(1, len(coordinates) + 1) if frames is None else frames
        points = [(frame, x, y) for frame, (x, y) in zip(frames, coordinates, strict=True)]
        return first_crossing(points, LINE_START, LINE_END, max_frame_gap=gap)

    def test_direct_crossing(self) -> None:
        """Count a step across the line at the first frame beyond it."""
        self.assertEqual(self.hit([(90, 5), (110, 5)])[0], 2)

    def test_contact_then_crossing(self) -> None:
        """Count a trajectory that touches the line and then leaves on the far side."""
        self.assertEqual(self.hit([(90, 5), (100, 5), (110, 5)])[0], 3)

    def test_contact_then_return(self) -> None:
        """Ignore a trajectory that touches the line and goes back."""
        self.assertIsNone(self.hit([(90, 5), (100, 5), (90, 5)]))

    def test_start_or_end_on_line(self) -> None:
        """Ignore a trajectory that starts or ends exactly on the line."""
        self.assertIsNone(self.hit([(100, 5), (110, 5)]))
        self.assertIsNone(self.hit([(90, 5), (100, 5)]))

    def test_contact_outside_segment(self) -> None:
        """Ignore a touch of the infinite line outside the segment."""
        self.assertIsNone(self.hit([(90, 5), (100, 20), (110, 5)]))

    def test_endpoints_and_direction(self) -> None:
        """Count a crossing at the segment end; opposite steps give opposite directions."""
        forward = self.hit([(90, 0), (110, 0)])
        reverse = self.hit([(110, 0), (90, 0)])

        self.assertEqual(forward[1], -reverse[1])
        self.assertEqual(forward[2], LINE_START)

    def test_gap_policy(self) -> None:
        """Lose a crossing inside a frame gap, unless the allowed gap is large enough."""
        points = [(90, 5), (110, 5)]

        self.assertIsNone(self.hit(points, frames=[1, 20]))
        self.assertIsNotNone(self.hit(points, frames=[1, 20], gap=19))

    def test_resume_after_gap(self) -> None:
        """Forget the points before a gap and count a crossing made after it."""
        self.assertEqual(self.hit([(90, 5), (90, 5), (110, 5)], frames=[1, 20, 21])[0], 21)

    def test_only_first_crossing(self) -> None:
        """Produce one event for a target that crosses the line twice."""
        trajectory = ("a", 0, [(1, 90, 5), (2, 110, 5), (3, 90, 5), (4, 110, 5)])

        [event] = count_events([trajectory], LINE_START, LINE_END)

        self.assertEqual((event["frame"], event["direction"]), (2, -1))

    def test_tally(self) -> None:
        """Count events per class and direction, plus the "all" total."""
        events = [{"cls_id": 0, "direction": 1}, {"cls_id": 0, "direction": -1}, {"cls_id": 2, "direction": 1}]

        counts = tally(events)

        self.assertEqual(counts["car"], {"total": 2, "ab": 1, "ba": 1})
        self.assertEqual(counts["bus"]["total"], 0)
        self.assertEqual(counts["all"], {"total": 3, "ab": 2, "ba": 1})

    def test_invalid_line(self) -> None:
        """Reject a line whose endpoints are equal."""
        with self.assertRaises(ValueError):
            validate_line((1, 1), (1, 1))

    def test_repeated_frames(self) -> None:
        """Reject a trajectory whose frames do not increase."""
        with self.assertRaises(ValueError):
            self.hit([(90, 5), (110, 5)], frames=[2, 2])

    def test_nan_coordinates(self) -> None:
        """Reject a trajectory with a non-finite coordinate."""
        with self.assertRaises(ValueError):
            self.hit([(float("nan"), 5)])


class GtAndTrackCountingTests(unittest.TestCase):
    """GT counts and tracker counts must come out of the same rules."""

    def test_gt_and_tracks_agree(self) -> None:
        """Give equal counts and protocols for GT and tracker rows that describe the same motion."""
        # One car moves across x=100, and its middle observation lies in an ignored region
        ann = SequenceAnn("MVI_1", ignored_regions=[Box(99, 0, 2, 10)])
        for fid, x in enumerate((90, 100, 110), 1):
            ann.frames[fid] = FrameAnn(fid, [Target(1, "car", Box(x - 1, 4, 2, 2), 0, 0)])
        line = {"p1": [100, 0], "p2": [100, 10]}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.txt"
            path.write_text("".join(f"{fid},1,{x - 1},4,2,2,1,0\n" for fid, x in enumerate((90, 100, 110), 1)),
                            encoding="utf-8")

            # The hole left by the ignored observation breaks the path unless the frame gap is 2
            for gap, expected_total in ((1, 0), (2, 1)):
                gt, _ = count_gt(ann, line, max_frame_gap=gap)
                pred = count_tracks(path, line, [(99, 0, 101, 10)], max_frame_gap=gap)

                self.assertEqual(gt["exclude_ignored"], pred["counts"])
                self.assertEqual(pred["counts"]["all"]["total"], expected_total)
                self.assertEqual(gt["protocol"], pred["protocol"])


class LinePlacementTests(unittest.TestCase):
    """Helpers that propose a counting line from the traffic flow."""

    def test_dominant_axis_wraps(self) -> None:
        """Treat angles just above 0 and just below pi as one axis, and refuse an empty flow."""
        streams = [{"ang": angle, "len": 100, "pts": np.zeros((5, 2))} for angle in (0.01, math.pi - 0.01)]

        selected, angle = dominant_direction(streams)

        self.assertEqual(len(selected), 2)
        self.assertLess(min(angle, math.pi - angle), 0.02)
        with self.assertRaises(ValueError):
            dominant_direction([])

    def test_clip_segment(self) -> None:
        """Cut a segment that leaves the image at the image border."""
        a, b = clip_segment(np.array([-5., 0.]), np.array([15., 10.]), (11, 11))

        np.testing.assert_allclose(a, [0, 2.5])
        np.testing.assert_allclose(b, [10, 7.5])


if __name__ == "__main__":
    unittest.main()
