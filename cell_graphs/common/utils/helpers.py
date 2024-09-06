import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seeds all random number generators.

    Args:
    ----
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    """Returns the available device for torch.

    Returns
    -------
    The GPU device if CUDA is available and the CPU device as a fallback.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
