"""Seeds, atomic JSON output, file fingerprints, environment info and CLI argument types."""
import argparse
import hashlib
import json
import os
import platform
import random
import tempfile
import time
from pathlib import Path

from .config import SEED


def set_seed(seed=SEED):
    """Seed Python, NumPy, OpenCV and PyTorch. Does not change speed."""
    import cv2
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """Bit-repeatable GPU results. Slower: do NOT use for speed measurements."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # must be set before the first CUDA matmul
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_info(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def write_json(path, value):
    """Write JSON through a temporary file so a crash never leaves a half-written file."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_info(device=None):
    """Software and hardware versions recorded next to every result."""
    import cv2
    import numpy as np
    import torch
    import ultralytics

    return {"device": str(device), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def fraction(text):
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def seed_value(text):
    value = int(text)
    if not 0 <= value < 2**31:
        raise argparse.ArgumentTypeError("must be between 0 and 2**31 - 1")
    return value
