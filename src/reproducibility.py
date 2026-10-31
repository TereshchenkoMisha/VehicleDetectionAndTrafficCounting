"""Configure repeatable inference in the same software and hardware environment"""
import os
import random

from .data.config import SEED


def set_seed(seed=SEED):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import cv2
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
