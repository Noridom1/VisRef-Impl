from __future__ import annotations

import numpy as np


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def should_stop(entropy: float, threshold: float, step: int,
                max_steps: int) -> bool:
    return entropy < threshold or step >= max_steps
