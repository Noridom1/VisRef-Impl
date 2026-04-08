from __future__ import annotations

import random
import importlib

import numpy as np

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
