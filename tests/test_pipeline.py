"""Tests for the ignored-region rule, trajectory building, tracker settings and the detect-track loop."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.counting.trajectories import build_trajectories
from src.data.regions import inside_any
from src.tracking.pipeline import build_tracker, detector_conf, keep_mask, run_sequence, tracker_settings, warmup


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
    def test_tracker_thresholds_and_low_score_input(self):
        for name in ("bytetrack", "botsort"):
            for conf in (0.1, 0.25, 0.6):
                settings = tracker_settings(name, conf)
                self.assertEqual(settings["track_high_thresh"], conf)
                self.assertEqual(settings["new_track_thresh"], conf)
                self.assertEqual(settings["track_low_thresh"], 0.1)
                self.assertEqual(detector_conf(settings), 0.1)

    def test_low_score_observation_preserves_identity(self):
        from ultralytics.engine.results import Boxes

        image = np.zeros((30, 30, 3), dtype=np.uint8)
        for name in ("bytetrack", "botsort"):
            tracker = build_tracker(tracker_settings(name, 0.25), "cpu")

            def boxes(score):
                return Boxes(np.array([[5, 5, 15, 15, score, 2]], dtype=np.float32), image.shape[:2])

            first = tracker.update(boxes(0.9).numpy(), image)
            recovered = tracker.update(boxes(0.18).numpy(), image)
            with self.subTest(tracker=name):
                self.assertEqual(len(first), 1)
                self.assertEqual(len(recovered), 1)
                self.assertEqual(first[0, 4], recovered[0, 4])

    def test_seed_is_repeatable_and_deterministic(self):
        import cv2
        import torch

        from src.utils import set_seed

        set_seed(42)
        first_numpy, first_torch = np.random.rand(3), torch.rand(3)
        set_seed(42)
        np.testing.assert_array_equal(first_numpy, np.random.rand(3))
        self.assertTrue(torch.equal(first_torch, torch.rand(3)))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(cv2.getNumThreads(), 1)

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
    def test_selected_nms_policy_is_class_aware(self):
        calls = []

        class Model:
            def predict(self, *_args, **kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(boxes=FakeBoxes(np.zeros((0, 4))))]

        class Tracker:
            def update(self, _det, _img):
                return np.zeros((0, 8))

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("src.tracking.pipeline.read_frame", return_value=image), \
                patch("src.tracking.pipeline.sync"):
            run_sequence(Model(), Tracker(), [(1, "a.jpg")], [], 0.1, 960, 0.7, "cpu")
            warmup(Model(), image, 960, "cpu", 0.1, n=1)
        self.assertEqual([c["agnostic_nms"] for c in calls], [False, False])

    def test_ignored_detections_never_reach_the_tracker(self):
        received = []

        class Model:
            def predict(self, *_args, **_kwargs):
                xywh = np.array([[5.0, 5.0, 2.0, 2.0], [50.0, 50.0, 2.0, 2.0]])
                return [SimpleNamespace(boxes=FakeBoxes(xywh))]

        class Tracker:
            def update(self, det, _img):
                received.append(det.xywh.tolist())
                return np.zeros((0, 8))

        frames = [(1, "a.jpg"), (2, "b.jpg")]
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("src.tracking.pipeline.read_frame", return_value=image), \
                patch("src.tracking.pipeline.sync"):
            rows, _, _ = run_sequence(Model(), Tracker(), frames, [(0, 0, 10, 10)], 0.1, 960, 0.7, "cpu")
        self.assertEqual(rows, [])
        self.assertEqual(received, [[[50.0, 50.0, 2.0, 2.0]]] * 2)




class PathTests(unittest.TestCase):
    @staticmethod
    def notebook_functions(name, names):
        import ast
        import json
        from pathlib import Path
        from src.config import ROOT
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                functions.extend(node for node in ast.parse("".join(cell["source"])).body
                                 if isinstance(node, ast.FunctionDef) and node.name in names)
        namespace = {"Path": Path, "read_json": lambda p: json.loads(Path(p).read_text(encoding="utf-8"))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), name, "exec"), namespace)
        return namespace

    def test_notebook_roots_from_workspace_and_project_folders(self):
        from src.config import ROOT
        for name in ("results.ipynb", "ablation.ipynb"):
            find = self.notebook_functions(name, {"find_root"})["find_root"]
            for directory in (ROOT, ROOT / "notebooks", ROOT.parent, ROOT.parent / "failure_analysis"):
                with self.subTest(notebook=name, cwd=directory):
                    self.assertEqual(find(directory), ROOT)

    def test_results_run_is_selected_as_a_matching_pair(self):
        import tempfile
        from pathlib import Path
        ns = self.notebook_functions("results.ipynb", {"results_inputs", "resolve_results_run"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports, tracks = root / "reports/run_one", root / "outputs/tracks_run_one"
            for p in ns["results_inputs"](reports, tracks, "dev", .25):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}", encoding="utf-8")
            selected = ns["resolve_results_run"](root, "dev", .25)
            self.assertEqual(selected[:2], (reports, tracks))
            (root / "outputs/tracks/dev").mkdir(parents=True)
            # An incomplete new standard run must not silently fall back to an archive.
            self.assertEqual(ns["resolve_results_run"](root, "dev", .25)[:2],
                             (root / "reports", root / "outputs/tracks"))

    def test_results_run_ambiguity_requires_an_explicit_choice(self):
        import tempfile
        from pathlib import Path
        ns = self.notebook_functions("results.ipynb", {"results_inputs", "resolve_results_run"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tag in ("one", "two"):
                for p in ns["results_inputs"](root / "reports" / tag, root / "outputs" / f"tracks_{tag}", "dev", .25):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "REPORTS_DIR"):
                ns["resolve_results_run"](root, "dev", .25)

    def test_dataset_yaml_uses_utf8_even_with_ansi_default(self):
        import tempfile
        from pathlib import Path
        import yaml
        from src.data.export import convert
        original_write = Path.write_text

        def ansi_default(target, data, *args, **kwargs):
            kwargs.setdefault("encoding", "cp1251")
            return original_write(target, data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "3 курс"
            out.mkdir()
            with patch.object(Path, "write_text", ansi_default):
                convert({}, {"dev": [], "test": []}, out_dir=out)
            dataset = yaml.safe_load((out / "detrac.yaml").read_text(encoding="utf-8"))
            self.assertEqual(Path(dataset["path"]), out.resolve())


if __name__ == "__main__":
    unittest.main()
