from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _probabilities(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("predictions contain non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("predictions must lie in [0, 1]")
    return p


def evaluate_binary(y, p) -> dict[str, float]:
    """Return the primary probability-classification metrics used in the project."""
    y = np.asarray(y, dtype=int)
    p = _probabilities(p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "pred_mean": float(p.mean()),
    }


def competition_score(auc: float, logloss: float) -> float:
    """Working exact composite recovered during the competition endgame audit."""
    return float(0.6 + 0.4 * auc - (0.6 / 0.595) * logloss)
