import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.counting.line_crossing import first_crossing, validate_line
from src.counting.lines import clip_segment, dominant_direction
from src.counting.trajectories import count_gt, count_tracks
from src.data.detrac import Box, FrameAnn, SequenceAnn, Target


class CrossingTests(unittest.TestCase):
    def hit(self, coordinates, frames=None, gap=1):
        frames = range(1, len(coordinates) + 1) if frames is None else frames
        points = [(frame, x, y) for frame, (x, y) in zip(frames, coordinates, strict=True)]
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


class GtAndTrackCountingTests(unittest.TestCase):
    def test_gt_and_prediction_ignore_observations_equally(self):
        ann = SequenceAnn("MVI_1", ignored_regions=[Box(99, 0, 2, 10)])
        for fid, x in enumerate((90, 100, 110), 1):
            ann.frames[fid] = FrameAnn(fid, [Target(1, "car", Box(x - 1, 4, 2, 2), 0, 0)])
        line = {"p1": [100, 0], "p2": [100, 10]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.txt"
            path.write_text("".join(f"{fid},1,{x - 1},4,2,2,1,0\n" for fid, x in enumerate((90, 100, 110), 1)))
            for gap, expected in ((1, 0), (2, 1)):
                gt, _ = count_gt(ann, line, max_frame_gap=gap)
                pred = count_tracks(path, line, [(99, 0, 101, 10)], max_frame_gap=gap)
                self.assertEqual(gt["exclude_ignored"], pred["counts"])
                self.assertEqual(pred["counts"]["all"]["total"], expected)
                self.assertEqual(gt["protocol"], pred["protocol"])


class LinePlacementTests(unittest.TestCase):
    def test_circular_flow_direction(self):
        streams = [{"ang": angle, "len": 100, "pts": np.zeros((5, 2))} for angle in (0.01, math.pi - 0.01)]
        selected, angle = dominant_direction(streams)
        self.assertEqual(len(selected), 2)
        self.assertLess(min(angle, math.pi - angle), 0.02)
        with self.assertRaises(ValueError):
            dominant_direction([])
        a, b = clip_segment(np.array([-5., 0.]), np.array([15., 10.]), (11, 11))
        np.testing.assert_allclose(a, [0, 2.5])
        np.testing.assert_allclose(b, [10, 7.5])


if __name__ == "__main__":
    unittest.main()
