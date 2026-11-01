import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.detrac import FrameAnn, SequenceAnn
from src.data.mot import load_mot
from src.data.sequences import frame_paths, resolve_naming
from src.eval.validation import run_dirs, validate_tracking_run
from src.utils import file_info, write_json


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def mot(self, text):
        path = self.root / "track.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_mot_validation(self):
        for text in ("1,1,0,0,10,10,1,0\n1,1,1,1,10,10,1,0\n", "1,1,nan,0,10,10,1,0\n",
                     "1,1,0,0,0,10,1,0\n", "1,1,0,0,10,10,1,3\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_mot(self.mot(text))
        self.assertEqual(load_mot(self.mot("")), {})

    def test_frame_index_and_offset(self):
        (self.root / "img00001.JPEG").touch()
        (self.root / "IMG00002.jpg").touch()
        self.assertEqual(list(frame_paths(self.root)), [1, 2])
        self.assertEqual(resolve_naming(self.root, [0, 1]), 1)
        with self.assertRaises(ValueError):
            resolve_naming(self.root, [1])
        (self.root / "IMG1.jpg").touch()
        with self.assertRaises(ValueError):
            frame_paths(self.root)

    def test_atomic_json_rejects_nan(self):
        path = self.root / "out.json"
        write_json(path, {"valid": 1})
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            write_json(path, {"invalid": float("nan")})
        self.assertEqual(path.read_bytes(), original)

    def test_gt_mot_preflight(self):
        from src.data.export import convert_all
        sequence = self.root / "MVI_1"
        sequence.mkdir()
        (sequence / "IMG00001.jpg").touch()
        ignored = self.root / "ignored.json"
        write_json(ignored, {"MVI_1": []})
        ann = SequenceAnn("MVI_1", frames={1: FrameAnn(1, [])})
        with patch("src.data.export.load_all", return_value=({"MVI_1": ann}, {})):
            with self.assertRaises(FileNotFoundError):
                convert_all(["MVI_1", "MVI_2"], self.root / "out", self.root, ignored)
        self.assertFalse((self.root / "out").exists())

    def test_gt_video_keeps_unannotated_frames(self):
        from src.tracking.viz import write_gt_video
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
            with patch("src.tracking.viz.read_frame", return_value=frame):
                write_gt_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            self.assertEqual(writer.write.call_count, 3)
            writer.release.assert_called_once()
            writer.reset_mock()
            failing = [frame, frame, RuntimeError("Unreadable image")]
            with patch("src.tracking.viz.read_frame", side_effect=failing):
                with self.assertRaisesRegex(RuntimeError, "Unreadable image"):
                    write_gt_video(ann, line, "MVI_1", [], self.root / "out.mp4", raw_root=self.root)
            writer.release.assert_called_once()

    def test_run_validation_and_fingerprint(self):
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
        mot.write_text("2,1,0,0,10,10,1,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")
        timing["per_seq"]["MVI_1"].pop("mot_sha256")
        timing["per_seq"]["MVI_1"]["frames"] = 1
        write_json(cfg / "timing.json", timing)
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            validate_tracking_run(cfg, ["MVI_1"], self.root / "raw", "dev")

    def test_unfinished_runs_are_skipped(self):
        for name in ("yolov8n_bytetrack_conf0.25", "yolov8n_botsort_conf0.25.partial"):
            (self.root / name).mkdir()
        self.assertEqual([p.name for p in run_dirs(self.root)], ["yolov8n_bytetrack_conf0.25"])


if __name__ == "__main__":
    unittest.main()
