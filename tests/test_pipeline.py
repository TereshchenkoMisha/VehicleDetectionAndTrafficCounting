"""Tests for the ignored-region rule, trajectories, tracker settings, the detect-track loop and the notebook helpers.

Tests:
    RegionTests
        test_borders_are_inside                 a point on the border of an ignored region is inside it
        test_keep_mask_uses_the_centre          a detection is dropped when its centre is in an ignored region
    TrajectoryTests
        test_class_is_majority_vote             a track gets its most frequent class
        test_regions_drop_rows                  observations in ignored regions are removed before grouping
    TrackerConfigTests
        test_tracker_thresholds                 one confidence sets the high and new-track thresholds
        test_low_score_observation_preserves_identity   a low-score box keeps the track id in both real trackers
        test_seed_is_repeatable_and_deterministic       the same seed gives the same random numbers
        test_shared_association_settings_match  ByteTrack and BoT-SORT share the association settings
    RunSequenceTests
        test_nms_policy_is_class_aware          the detector is called with class-aware NMS
        test_ignored_detections_never_reach_the_tracker   detections in ignored regions are filtered before tracking
        test_track_rows_use_xywh_and_project_classes     tracker rows become x, y, w, h and project class ids
    TrackConfigTests
        test_video_gets_the_counting_line_of_its_sequence   each video receives the line of its own sequence
    NotebookTests
        test_roots_from_workspace_and_project_folders   notebooks find the repository root from several folders
        test_results_run_is_selected_as_a_matching_pair   the results notebook picks one reports and tracks pair
        test_results_run_ambiguity_requires_a_choice    several complete runs need an explicit choice
    ExportEncodingTests
        test_dataset_yaml_uses_utf8             detrac.yaml is UTF-8 even when the system default is not
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from ultralytics.engine.results import Boxes

from src.config import ROOT
from src.counting.trajectories import build_trajectories
from src.data.export import convert
from src.data.regions import inside_any
from src.tracking.pipeline import (
    build_tracker,
    detector_conf,
    keep_mask,
    run_sequence,
    track_config,
    tracker_settings,
    warmup,
)
from src.utils import set_seed

CONFIDENCES: tuple[float, ...] = (0.1, 0.25, 0.6)       # tracker thresholds used in the threshold experiment


class RegionTests(unittest.TestCase):
    """The ignored-region rule."""

    def test_borders_are_inside(self) -> None:
        """Count a point on the region border as inside, and a point just outside as outside."""
        points = [(0, 0), (10, 10), (10.1, 5), (5, -0.1)]

        self.assertEqual(inside_any(points, [(0, 0, 10, 10)]).tolist(), [True, True, False, False])
        self.assertEqual(inside_any(points, []).tolist(), [False] * 4)

    def test_keep_mask_uses_the_centre(self) -> None:
        """Drop a detection when its box centre is in an ignored region."""
        xywh = np.array([[5.0, 5.0, 4.0, 4.0], [50.0, 50.0, 4.0, 4.0]])      # centre x, centre y, w, h

        self.assertEqual(keep_mask(xywh, [(0, 0, 10, 10)]).tolist(), [False, True])


class TrajectoryTests(unittest.TestCase):
    """Trajectories built from observations."""

    # (frame, track id, class id, centre x, centre y): class 1 is the majority; the last row is far from the others
    ROWS = [(1, 7, 0, 5, 5), (2, 7, 1, 6, 5), (3, 7, 1, 7, 5), (4, 7, 1, 50, 50)]

    def test_class_is_majority_vote(self) -> None:
        """Give a track the class that most of its observations have."""
        [(key, cls, points)] = build_trajectories(self.ROWS)

        self.assertEqual((key, cls, len(points)), (7, 1, 4))

    def test_regions_drop_rows(self) -> None:
        """Remove the observations that lie in an ignored region."""
        [(_, _, points)] = build_trajectories(self.ROWS, regions=[(0, 0, 10, 10)])

        self.assertEqual([p[0] for p in points], [4])


class TrackerConfigTests(unittest.TestCase):
    """Tracker settings and the behaviour of the real trackers."""

    def test_tracker_thresholds(self) -> None:
        """Set the high and new-track thresholds from one confidence and keep the detector at the low one."""
        for name in ("bytetrack", "botsort"):
            for conf in CONFIDENCES:
                settings = tracker_settings(name, conf)

                self.assertEqual(settings["track_high_thresh"], conf)
                self.assertEqual(settings["new_track_thresh"], conf)
                self.assertEqual(settings["track_low_thresh"], 0.1)
                self.assertEqual(detector_conf(settings), 0.1)

    def test_low_score_observation_preserves_identity(self) -> None:
        """Keep the track id when the next detection has a low score, in ByteTrack and in BoT-SORT."""
        image = np.zeros((30, 30, 3), dtype=np.uint8)

        def boxes(score: float) -> Any:
            """One box of class 2 with the given score, in the format the trackers read."""
            return Boxes(np.array([[5, 5, 15, 15, score, 2]], dtype=np.float32), image.shape[:2]).numpy()

        for name in ("bytetrack", "botsort"):
            tracker = build_tracker(tracker_settings(name, 0.25), "cpu")

            # A confident detection starts the track; a score below 0.25 but above 0.1 must keep it
            first = tracker.update(boxes(0.9), image)
            recovered = tracker.update(boxes(0.18), image)

            with self.subTest(tracker=name):
                self.assertEqual(len(first), 1)
                self.assertEqual(len(recovered), 1)
                self.assertEqual(first[0, 4], recovered[0, 4])

    def test_seed_is_repeatable_and_deterministic(self) -> None:
        """Repeat random numbers after the same seed and switch on deterministic algorithms."""
        set_seed(42)
        first_numpy, first_torch = np.random.rand(3), torch.rand(3)
        set_seed(42)

        np.testing.assert_array_equal(first_numpy, np.random.rand(3))
        self.assertTrue(torch.equal(first_torch, torch.rand(3)))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(cv2.getNumThreads(), 1)

    def test_shared_association_settings_match(self) -> None:
        """Use the same association settings in ByteTrack and BoT-SORT, so only the tracker differs."""
        shared = ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh")
        a, b = tracker_settings("bytetrack"), tracker_settings("botsort")

        self.assertEqual([a[k] for k in shared], [b[k] for k in shared])


class FakeBoxes:
    """The part of Ultralytics `Boxes` that the pipeline uses: `xywh`, `cpu`, `numpy`, length and indexing."""

    def __init__(self, xywh: NDArray[np.floating]) -> None:
        self.xywh = xywh

    def cpu(self) -> "FakeBoxes":
        """Return self: the data is already on the CPU."""
        return self

    def numpy(self) -> "FakeBoxes":
        """Return self: the data is already a NumPy array."""
        return self

    def __len__(self) -> int:
        """Number of boxes."""
        return len(self.xywh)

    def __getitem__(self, mask: NDArray[np.bool_]) -> "FakeBoxes":
        """The boxes selected by a boolean mask."""
        return FakeBoxes(self.xywh[mask])


class EmptyModel:
    """A detector that finds nothing and records the keyword arguments of every `predict` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, *_args: Any, **kwargs: Any) -> list[Any]:
        """Record the call and return no detections."""
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes(np.zeros((0, 4))))]


class RunSequenceTests(unittest.TestCase):
    """The detect-and-track loop with a fake detector and a fake tracker."""

    def test_nms_policy_is_class_aware(self) -> None:
        """Call the detector with class-aware NMS, in the loop and in the warm-up."""
        class Tracker:
            def update(self, _det: Any, _img: Any) -> NDArray[np.floating]:
                """Return no tracks."""
                return np.zeros((0, 8))

        model = EmptyModel()
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("src.tracking.pipeline.read_frame", return_value=image), patch("src.tracking.pipeline.sync"):
            run_sequence(model, Tracker(), [(1, "a.jpg")], [], 0.1, 960, 0.7, "cpu")
            warmup(model, image, 960, "cpu", 0.1, n=1)

        self.assertEqual([call["agnostic_nms"] for call in model.calls], [False, False])

    def test_ignored_detections_never_reach_the_tracker(self) -> None:
        """Remove the detection inside the ignored region before the tracker update, in every frame."""
        received: list[list[list[float]]] = []

        class Model:
            def predict(self, *_args: Any, **_kwargs: Any) -> list[Any]:
                """Return two detections."""
                # One detection inside the ignored region (0, 0, 10, 10) and one outside it
                xywh = np.array([[5.0, 5.0, 2.0, 2.0], [50.0, 50.0, 2.0, 2.0]])
                return [SimpleNamespace(boxes=FakeBoxes(xywh))]

        class Tracker:
            def update(self, det: FakeBoxes, _img: Any) -> NDArray[np.floating]:
                """Record the detections received and return no tracks."""
                received.append(det.xywh.tolist())
                return np.zeros((0, 8))

        frames = [(1, "a.jpg"), (2, "b.jpg")]
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("src.tracking.pipeline.read_frame", return_value=image), patch("src.tracking.pipeline.sync"):
            rows, _, _ = run_sequence(Model(), Tracker(), frames, [(0, 0, 10, 10)], 0.1, 960, 0.7, "cpu")

        self.assertEqual(rows, [])
        self.assertEqual(received, [[[50.0, 50.0, 2.0, 2.0]]] * 2)

    def test_track_rows_use_xywh_and_project_classes(self) -> None:
        """Turn a tracker row into x, y, width, height and a project class id."""
        class Tracker:
            def update(self, _det: Any, _img: Any) -> NDArray[np.floating]:
                """Return one confirmed track."""
                # x1, y1, x2, y2, track id, score, COCO class (5 = bus), detection index
                return np.array([[10.0, 20.0, 30.0, 60.0, 5.0, 0.9, 5.0, 0.0]])

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("src.tracking.pipeline.read_frame", return_value=image), patch("src.tracking.pipeline.sync"):
            rows, _, _ = run_sequence(EmptyModel(), Tracker(), [(3, "a.jpg")], [], 0.1, 960, 0.7, "cpu")

        # Width is 30 - 10, height is 60 - 20, and the project id of a bus is 1
        self.assertEqual(rows, [(3, 5, 10.0, 20.0, 20.0, 40.0, 0.9, 1)])


class TrackConfigTests(unittest.TestCase):
    """The wiring of `track_config` with the model, the tracker and the video writer replaced."""

    def test_video_gets_the_counting_line_of_its_sequence(self) -> None:
        """Pass the line of each sequence to its video, and no line to a sequence that has none."""
        line = {"p1": [0, 0], "p2": [0, 9]}
        frames = {"A": [(1, Path("a.jpg"))], "B": [(1, Path("b.jpg"))]}

        with tempfile.TemporaryDirectory() as temporary, \
                patch("src.tracking.pipeline.YOLO"), patch("src.tracking.pipeline.weights_path"), \
                patch("src.tracking.pipeline.read_frame"), patch("src.tracking.pipeline.warmup"), \
                patch("src.tracking.pipeline.build_tracker"), patch("src.tracking.pipeline.set_seed"), \
                patch("src.tracking.pipeline.run_sequence", return_value=([], 0.01, 0.01)), \
                patch("src.tracking.pipeline.write_tracking_video") as video:
            out_dir = Path(temporary) / "cfg.partial"
            out_dir.mkdir()
            track_config("yolov8n.pt", "bytetrack", frames, {"A": [], "B": []}, out_dir, imgsz=960, iou=0.7,
                         device="cpu", seed=42, vis_dir=out_dir / "videos", lines={"A": line})

        # Positional arguments of write_tracking_video: frames, MOT file, regions, line, label, video path
        self.assertEqual([call.args[3] for call in video.call_args_list], [line, None])
        self.assertEqual(video.call_args_list[0].args[4], "yolov8n + bytetrack")
        self.assertEqual([call.args[5].name for call in video.call_args_list], ["cfg_A.mp4", "cfg_B.mp4"])


class NotebookTests(unittest.TestCase):
    """Functions taken out of the notebooks and run on their own."""

    @staticmethod
    def notebook_functions(name: str, names: set[str]) -> dict[str, Any]:
        """Compile only the named top-level functions of a notebook, so they run without running the notebook."""
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                functions.extend(node for node in ast.parse("".join(cell["source"])).body
                                 if isinstance(node, ast.FunctionDef) and node.name in names)
        namespace = {"Path": Path, "read_json": lambda p: json.loads(Path(p).read_text(encoding="utf-8"))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), name, "exec"), namespace)
        return namespace

    def test_roots_from_workspace_and_project_folders(self) -> None:
        """Find the repository root from the root, the notebooks folder, the workspace and a sibling folder."""
        for name in ("results.ipynb", "ablation.ipynb"):
            find = self.notebook_functions(name, {"find_root"})["find_root"]
            for directory in (ROOT, ROOT / "notebooks", ROOT.parent, ROOT.parent / "failure_analysis"):
                with self.subTest(notebook=name, cwd=directory):
                    self.assertEqual(find(directory), ROOT)

    def test_results_run_is_selected_as_a_matching_pair(self) -> None:
        """Select the tagged reports and tracks together, and never fall back to an archive."""
        ns = self.notebook_functions("results.ipynb", {"results_inputs", "resolve_results_run"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports, tracks = root / "reports/run_one", root / "outputs/tracks_run_one"
            for path in ns["results_inputs"](reports, tracks, "dev", .25):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            selected = ns["resolve_results_run"](root, "dev", .25)
            self.assertEqual(selected[:2], (reports, tracks))

            # An incomplete new standard run must not silently fall back to an archive
            (root / "outputs/tracks/dev").mkdir(parents=True)
            self.assertEqual(ns["resolve_results_run"](root, "dev", .25)[:2],
                             (root / "reports", root / "outputs/tracks"))

    def test_results_run_ambiguity_requires_a_choice(self) -> None:
        """Ask for an explicit REPORTS_DIR when two complete runs exist."""
        ns = self.notebook_functions("results.ipynb", {"results_inputs", "resolve_results_run"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tag in ("one", "two"):
                for path in ns["results_inputs"](root / "reports" / tag, root / "outputs" / f"tracks_{tag}",
                                                 "dev", .25):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "REPORTS_DIR"):
                ns["resolve_results_run"](root, "dev", .25)


class ExportEncodingTests(unittest.TestCase):
    """Text encoding of the exported dataset description."""

    def test_dataset_yaml_uses_utf8(self) -> None:
        """Write detrac.yaml in UTF-8 even when `Path.write_text` defaults to a Windows code page."""
        original_write = Path.write_text

        def ansi_default(target: Path, data: str, *args: Any, **kwargs: Any) -> int:
            """`Path.write_text` that behaves as if the system default encoding were cp1251."""
            kwargs.setdefault("encoding", "cp1251")
            return original_write(target, data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            # Cyrillic folder name: the path itself must survive the round trip through the yaml file
            out = Path(temporary) / "3 курс"
            out.mkdir()
            with patch.object(Path, "write_text", ansi_default):
                convert({}, {"dev": [], "test": []}, out_dir=out)

            dataset = yaml.safe_load((out / "detrac.yaml").read_text(encoding="utf-8"))

        self.assertEqual(Path(dataset["path"]), out.resolve())


if __name__ == "__main__":
    unittest.main()
