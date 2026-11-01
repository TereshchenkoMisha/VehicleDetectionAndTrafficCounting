"""Small helpers used across the project: reproducibility switches, JSON and hash helpers, the
environment record stored next to results, and argparse value validators."""
import argparse
import hashlib
import json
import os
import platform
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import SEED


# ---------------------------------------------------------------- reproducibility

def set_seed(seed: int = SEED) -> None:
    """Seed every random generator the project touches.

    Covers Python's `random`, NumPy, OpenCV and PyTorch (CPU and all GPUs). This only fixes the
    random streams; it does not force deterministic GPU kernels and does not change speed. For
    bit-repeatable GPU results call `set_deterministic` as well.
    """
    import cv2
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic() -> None:
    """Make GPU computation bit-repeatable between runs on the same machine and software.

    Disables cuDNN autotuning and forces deterministic implementations (PyTorch raises an error
    for an operation that has none). This is slower, so it must not be used for speed measurements.
    """
    # cuBLAS reads this variable when it is first used, so it has to be set before any CUDA matmul
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False          # benchmark mode picks algorithms by timing: not repeatable
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------- files

def file_info(path: str | Path) -> dict[str, str]:
    """Absolute path and SHA-256 of a file.

    Stored in result files, so a later reader can tell exactly which input produced a result and
    detect that the input changed afterwards.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # Hash in 1 MiB blocks so a large file (weights, videos) is never fully in memory
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def read_json(path: str | Path) -> Any:
    """Parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Write `value` as indented JSON, replacing the target atomically.

    The text is written to a temporary file in the same folder and then moved over the target, so
    a crash or a full disk never leaves a half-written result. NaN and Infinity are rejected
    because they are not valid JSON. Missing folders are created.
    """
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"      # serialise first: errors leave the target alone
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)     # atomic when source and target are on one filesystem
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)       # no-op after a successful replace


def env_info(device: Any = None) -> dict[str, Any]:
    """Versions of the libraries and the device, saved next to results.

    Timings and sometimes metrics depend on these, so a result is only comparable with another
    when they match.
    """
    import cv2
    import numpy as np
    import torch
    import ultralytics

    return {"device": str(device), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------- argparse validators
# Used as `type=` in add_argument. Raising ArgumentTypeError makes argparse print a clean usage
# error instead of a traceback.

def positive_int(text: str) -> int:
    """Integer >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def fraction(text: str) -> float:
    """Float in [0, 1]: a score or IoU threshold."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def seed_value(text: str) -> int:
    """Integer in [0, 2**31): the range NumPy accepts as a seed."""
    value = int(text)
    if not 0 <= value < 2**31:
        raise argparse.ArgumentTypeError("must be between 0 and 2**31 - 1")
    return value
