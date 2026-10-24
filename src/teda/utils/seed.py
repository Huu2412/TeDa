import os
import random

import numpy as np
import torch


def set_seed(seed: int, rank: int = 0) -> None:
    """Different (but reproducible) streams per rank, so chunk-size / SpecAug choices differ across GPUs."""
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
