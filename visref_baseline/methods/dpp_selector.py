from __future__ import annotations

import numpy as np


def build_Mk(z_k: np.ndarray) -> np.ndarray:
    return z_k.T @ z_k


def build_kernel(visual_tokens: np.ndarray, M_k: np.ndarray) -> np.ndarray:
    return visual_tokens @ M_k @ visual_tokens.T


def relevance_scores(visual_tokens: np.ndarray, M_k: np.ndarray) -> np.ndarray:
    return np.einsum("nd,df,nf->n", visual_tokens, M_k, visual_tokens)


def _safe_logdet(matrix: np.ndarray, eps: float = 1e-6) -> float:
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    sign, logdet = np.linalg.slogdet(matrix + eps * eye)
    if sign <= 0:
        return -1e30
    return float(logdet)


def greedy_logdet_select(L: np.ndarray, m: int, eps: float = 1e-6) -> list[int]:
    n = L.shape[0]
    if m <= 0 or n == 0:
        return []
    m = min(m, n)
    selected: list[int] = []
    candidates = set(range(n))

    for _ in range(m):
        best_idx = None
        best_gain = -1e30
        base_logdet = 0.0
        if selected:
            base_L = L[np.ix_(selected, selected)]
            base_logdet = _safe_logdet(base_L, eps=eps)

        for idx in candidates:
            trial = selected + [idx]
            trial_L = L[np.ix_(trial, trial)]
            gain = _safe_logdet(trial_L, eps=eps) - base_logdet
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
