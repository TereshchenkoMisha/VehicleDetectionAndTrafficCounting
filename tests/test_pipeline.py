import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.counting.trajectories import build_trajectories
from src.data.regions import inside_any
from src.tracking.pipeline import detector_conf, keep_mask, run_sequence, tracker_settings


class RegionTests(unittest.TestCase):
    def test_borders_are_inside(self):
        points = [(0, 0), (10, 10), (10.1, 5), (5, -0.1)]
        self.assertEqual(inside_any(points, [(0, 0, 10, 10)]).tolist(), [True, True, False, False])
        self.assertEqual(inside_any(points, []).tolist(), [False] * 4)

    def test_keep_mask_uses_the_centre(self):
        xywh = np.array([[5.0, 5.0, 4.0, 4.0], [50.0, 50.0, 4.0, 4.0]])      # centre x, centre y, w, h
        self.assertEqual(keep_mask(xywh, [(0, 0, 10, 10)]).tolist(), [False, True])


class TrajectoryTests(unittest.TestCase):
    def test_class_is_majority_vote_and_regions_drop_rows(self):
        rows = [(1, 7, 0, 5, 5), (2, 7, 1, 6, 5), (3, 7, 1, 7, 5), (4, 7, 1, 50, 50)]
        [(key, cls, points)] = build_trajectories(rows)
        self.assertEqual((key, cls, len(points)), (7, 1, 4))
        [(_, _, points)] = build_trajectories(rows, regions=[(0, 0, 10, 10)])
        self.assertEqual([p[0] for p in points], [4])


class TrackerConfigTests(unittest.TestCase):
    def test_detector_runs_at_the_low_threshold(self):
        for name in ("bytetrack", "botsort"):
            settings = tracker_settings(name, conf=0.25)
            self.assertEqual(settings["track_high_thresh"], 0.25)
            self.assertEqual(detector_conf(settings), settings["track_low_thresh"])
            self.assertLess(detector_conf(settings), settings["track_high_thresh"])

    def test_shared_association_settings_match(self):
        shared = ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh")
        a, b = tracker_settings("bytetrack"), tracker_settings("botsort")
        self.assertEqual([a[k] for k in shared], [b[k] for k in shared])


class FakeBoxes:
    def __init__(self, xywh):
        self.xywh = xywh

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.xywh)

    def __getitem__(self, mask):
        return FakeBoxes(self.xywh[mask])


class RunSequenceTests(unittest.TestCase):
    def test_ignored_detections_never_reach_the_tracker(self):
        received = []

        class Model:
            def predict(self, *args, **kwargs):
                xywh = np.array([[5.0, 5.0, 2.0, 2.0], [50.0, 50.0, 2.0, 2.0]])
                return [SimpleNamespace(boxes=FakeBoxes(xywh))]

        class Tracker:
            def update(self, det, img):
                received.append(det.xywh.tolist())
                return np.zeros((0, 8))

        frames = [(1, "a.jpg"), (2, "b.jpg")]
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("src.tracking.pipeline.read_frame", return_value=image), \
                patch("src.tracking.pipeline.sync"):
            rows, _, _ = run_sequence(Model(), Tracker(), frames, [(0, 0, 10, 10)], 0.1, 960, 0.7, "cpu")
        self.assertEqual(rows, [])
        self.assertEqual(received, [[[50.0, 50.0, 2.0, 2.0]]] * 2)


if __name__ == "__main__":
    unittest.main()
