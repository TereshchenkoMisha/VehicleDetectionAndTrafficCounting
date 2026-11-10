"""Tests for MOT parsing, frame naming, JSON writing, GT export, GT and tracking videos and run validation.

Tests:
    FileTests
        test_unicode_image_paths_are_read       images in a Cyrillic and Chinese folder are read in BGR order
        test_preview_is_written_to_a_unicode_path   the line preview image is saved to such a folder
        test_unreadable_images                  empty, broken and missing images raise a clear error
        test_mot_validation                     invalid MOT rows raise ValueError; an empty file is empty
        test_frame_index_and_offset             frame numbers come from file names; the offset is detected; repeats fail
        test_atomic_json_rejects_nan            a failed JSON write leaves the old file untouched
        test_gt_mot_preflight                   GT export fails before writing when a sequence has no frames
        test_gt_video_keeps_unannotated_frames  every frame is written to the GT video; the writer is always released
        test_tracking_video_marks_the_crossing_frame   the crossing circle appears in the crossing frame only
        test_tracking_video_without_line        a video without a line is still written, with no line colour
        test_run_validation_and_fingerprint     a changed MOT file or incomplete timing makes a run invalid
        test_run_must_use_the_same_ignored_regions   a run made with other ignored regions is refused
        test_unfinished_runs_are_skipped        `.partial` folders are not listed as runs
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.data.detrac import FrameAnn, SequenceAnn
from src.data.export import convert_all
from src.data.mot import load_mot, write_mot
from src.data.sequences import frame_paths, read_frame, resolve_naming
from src.data.split import mean_brightness
from src.eval.validation import run_dirs, validate_tracking_run
from src.tracking.viz import LINE_COLOR, write_gt_video, write_preview, write_tracking_video
from src.utils import file_info, write_json

TRACK_IMAGE_SIZE: tuple[int, int] = (150, 120)      # height, width of the frames in the tracking video tests


class FileTests(unittest.TestCase):
    """File readers, writers and checks, each on a temporary folder."""

    def setUp(self) -> None:
        """Create a temporary folder that is removed after the test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def mot(self, text: str) -> Path:
        """Write `text` to a MOT file and return its path."""
        path = self.root / "track.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def unicode_folder_with_image(self) -> tuple[Path, Path]:
        """A folder named in Cyrillic and Chinese with one image: OpenCV cannot open such paths on Windows."""
        folder = self.root / "кадры 课程"
        folder.mkdir()
        image_path = folder / "img00001.jpg"
        Image.new("RGB", (80, 60), (10, 20, 30)).save(image_path)
        return folder, image_path

    def test_unicode_image_paths_are_read(self) -> None:
        """Read an image from a Unicode folder as a BGR array."""
        folder, image_path = self.unicode_folder_with_image()

        frame = read_frame(image_path)

        self.assertEqual(frame.shape, (60, 80, 3))
        np.testing.assert_allclose(frame[0, 0], [30, 20, 10], atol=2)      # the RGB colour (10, 20, 30) in BGR
        self.assertAlmostEqual(mean_brightness(folder), float(frame.mean()))

    def test_preview_is_written_to_a_unicode_path(self) -> None:
        """Save the counting-line preview into a Unicode folder and read it back."""
        folder, image_path = self.unicode_folder_with_image()
        preview = folder / "линия.png"

        write_preview([(1, image_path)], "MVI_1", {"p1": [10, 0], "p2": [10, 59]}, preview)

        self.assertEqual(read_frame(preview).shape, (60, 80, 3))

    def test_unreadable_images(self) -> None:
        """Raise a clear error for an empty, a broken and a missing image."""
        for name, content in (("empty.jpg", b""), ("broken.jpg", b"invalid")):
            file = self.root / name
            file.write_bytes(content)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Cannot read image"):
                read_frame(file)

        with self.assertRaisesRegex(RuntimeError, "Cannot read image"):
            read_frame(self.root / "missing.jpg")

    def test_mot_validation(self) -> None:
        """Reject a duplicate track, a NaN, a zero-width box and an unknown class; accept an empty file."""
        invalid_files = ("1,1,0,0,10,10,1,0\n1,1,1,1,10,10,1,0\n", "1,1,nan,0,10,10,1,0\n",
                         "1,1,0,0,0,10,1,0\n", "1,1,0,0,10,10,1,3\n")
        for text in invalid_files:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_mot(self.mot(text))

        self.assertEqual(load_mot(self.mot("")), {})

    def test_frame_index_and_offset(self) -> None:
        """Take frame numbers from file names, detect the numbering offset, and reject a repeated number."""
        (self.root / "img00001.JPEG").touch()
        (self.root / "IMG00002.jpg").touch()

        self.assertEqual(list(frame_paths(self.root)), [1, 2])
        self.assertEqual(resolve_naming(self.root, [0, 1]), 1)
        with self.assertRaises(ValueError):
            resolve_naming(self.root, [1])

        # IMG1.jpg is frame 1 again, and a repeated frame number would shift the annotations
        (self.root / "IMG1.jpg").touch()
        with self.assertRaises(ValueError):
            frame_paths(self.root)

    def test_atomic_json_rejects_nan(self) -> None:
        """Keep the old JSON file when a new value cannot be written."""
        path = self.root / "out.json"
        write_json(path, {"valid": 1})
        original = path.read_bytes()

        with self.assertRaises(ValueError):
            write_json(path, {"invalid": float("nan")})

        self.assertEqual(path.read_bytes(), original)

    def test_gt_mot_preflight(self) -> None:
        """Fail before writing anything when a requested sequence has no frames."""
        sequence = self.root / "MVI_1"
        sequence.mkdir()
        (sequence / "IMG00001.jpg").touch()
        ignored = self.root / "ignored.json"
        write_json(ignored, {"MVI_1": []})
        ann = SequenceAnn("MVI_1", frames={1: FrameAnn(1, [])})

        with (patch("src.data.export.load_all", return_value=({"MVI_1": ann}, {})),
              self.assertRaises(FileNotFoundError)):
            convert_all(["MVI_1", "MVI_2"], self.root / "out", self.root, ignored)

        self.assertFalse((self.root / "out").exists())

    def test_gt_video_keeps_unannotated_frames(self) -> None:
        """Write every frame to the GT video, also those without annotation, and always release the writer."""
        sequence = self.root / "MVI_1"
        sequence.mkdir()
        for fid in (1, 2, 3):
            (sequence / f"IMG{fid:05d}.jpg").touch()
        ann = SequenceAnn("MVI_1", frames={1: FrameAnn(1, []), 3: FrameAnn(3, [])})
        line = {"p1": [10, 0], "p2": [10, 9]}
        frame = np.zeros((10, 20, 3), dtype=np.uint8)

        with patch("src.tracking.viz.cv2.VideoWriter") as writer_class:
            writer = writer_class.return_value
            writer.isOpened.return_value = True

            # All three frames are written, although frame 2 has no annotation entry
            with patch("src.tracking.viz.read_frame", return_value=frame):
                write_gt_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            self.assertEqual(writer.write.call_count, 3)
            writer.release.assert_called_once()
            writer.reset_mock()

            # A frame that cannot be read stops the video, and the writer is still released
            failing = [frame, frame, RuntimeError("Unreadable image")]
            with (patch("src.tracking.viz.read_frame", side_effect=failing),
                  self.assertRaisesRegex(RuntimeError, "Unreadable image")):
                write_gt_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            writer.release.assert_called_once()

    def written_tracking_frames(self, line: dict[str, list[int]] | None) -> list[NDArray[np.uint8]]:
        """Frames written by `write_tracking_video` for one car that crosses the vertical line x=60 in frame 2."""
        mot = self.root / "MVI_1.txt"
        write_mot([(1, 7, 40, 55, 10, 10, 0.9, 0), (2, 7, 70, 55, 10, 10, 0.9, 0)], mot)
        frames = [(1, Path("a.jpg")), (2, Path("b.jpg"))]
        black_image = np.zeros((*TRACK_IMAGE_SIZE, 3), dtype=np.uint8)

        with patch("src.tracking.viz.cv2.VideoWriter") as writer_class:
            writer_class.return_value.isOpened.return_value = True
            with patch("src.tracking.viz.read_frame", return_value=black_image):
                write_tracking_video(frames, mot, [], line, "yolov8n + bytetrack", self.root / "out.mp4")
            return [call.args[0] for call in writer_class.return_value.write.call_args_list]

    def test_tracking_video_marks_the_crossing_frame(self) -> None:
        """Draw the crossing circle in the crossing frame and not before it."""
        before, crossing = self.written_tracking_frames({"p1": [60, 0], "p2": [60, 149]})

        def yellow_pixels(frame: NDArray[np.uint8]) -> int:
            # Left of the line only the crossing circle can be yellow: the line itself is at x=60
            return int((frame[:, :55] == LINE_COLOR).all(axis=2).sum())

        self.assertEqual(yellow_pixels(before), 0)
        self.assertGreater(yellow_pixels(crossing), 0)

    def test_tracking_video_without_line(self) -> None:
        """Write all frames of a video without a counting line, and draw no line colour in them."""
        frames = self.written_tracking_frames(None)

        self.assertEqual(len(frames), 2)
        self.assertFalse(any((frame == LINE_COLOR).all(axis=2).any() for frame in frames))

    def test_run_validation_and_fingerprint(self) -> None:
        """Refuse a run whose MOT file changed after timing, or whose frame count is incomplete."""
        sequence = self.root / "raw" / "MVI_1"
        sequence.mkdir(parents=True)
        for fid in (1, 2):
            (sequence / f"IMG{fid:05d}.jpg").touch()
        cfg = self.root / "run"
        cfg.mkdir()
        mot = cfg / "MVI_1.txt"
        mot.write_text("1,1,0,0,10,10,1,0\n", encoding="utf-8")
        timing = {"split": "dev", "frames": 2,
                  "per_seq": {"MVI_1": {"frames": 2, "rows": 1, "mot_sha256": file_info(mot)["sha256"]}}}
        write_json(cfg / "timing.json", timing)

        validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")

        # The MOT file changes after timing.json was written
        mot.write_text("2,1,0,0,10,10,1,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")

        # Timing that covers one frame of a two-frame sequence
        timing["per_seq"]["MVI_1"].pop("mot_sha256")
        timing["per_seq"]["MVI_1"]["frames"] = 1
        write_json(cfg / "timing.json", timing)
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")

    def test_run_must_use_the_same_ignored_regions(self) -> None:
        """Refuse a run whose recorded ignored regions differ from the current ones."""
        (self.root / "raw" / "MVI_1").mkdir(parents=True)
        (self.root / "raw" / "MVI_1" / "IMG00001.jpg").touch()
        cfg = self.root / "run"
        cfg.mkdir()
        (cfg / "MVI_1.txt").write_text("", encoding="utf-8")
        write_json(cfg / "timing.json", {"split": "dev", "frames": 1, "per_seq": {"MVI_1": {"frames": 1, "rows": 0}},
                                         "provenance": {"ignored_regions": {"MVI_1": [[0, 0, 10, 10]]}}})

        validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev", ignored={"MVI_1": [[0, 0, 10, 10]]})

        with self.assertRaisesRegex(ValueError, "Ignored regions differ"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev", ignored={"MVI_1": []})

    def test_unfinished_runs_are_skipped(self) -> None:
        """List only finished configuration folders, not `.partial` ones."""
        for name in ("yolov8n_bytetrack_conf0.25", "yolov8n_botsort_conf0.25.partial"):
            (self.root / name).mkdir()

        self.assertEqual([p.name for p in run_dirs(self.root)], ["yolov8n_bytetrack_conf0.25"])


if __name__ == "__main__":
    unittest.main()
