from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def fit_platt(p, y) -> dict:
    p = _clip(p)
    y = np.asarray(y, dtype=int)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
    return {
        "type": "platt",
        "coef": model.coef_.ravel().tolist(),
        "intercept": model.intercept_.ravel().tolist(),
    }


def fit_beta(p, y) -> dict:
    p = _clip(p)
    y = np.asarray(y, dtype=int)
    x = np.column_stack([np.log(p), -np.log(1.0 - p)])
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
    return {
        "type": "beta",
        "coef": model.coef_.ravel().tolist(),
        "intercept": model.intercept_.ravel().tolist(),
    }


def apply_calibrator(calibrator: dict, p) -> np.ndarray:
    p = _clip(p)

    if calibrator["type"] == "platt":
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
    elif calibrator["type"] == "beta":
        x = np.column_stack([np.log(p), -np.log(1.0 - p)])
    else:
        raise ValueError(f"unsupported calibrator: {calibrator['type']}")

    coef = np.asarray(calibrator["coef"], dtype=float)
    intercept = float(calibrator["intercept"][0])
    score = x @ coef + intercept
    return 1.0 / (1.0 + np.exp(-score))


# The competition pipeline used nested model refits to generate calibration
# training predictions without outer-fold leakage. This module intentionally
# exposes only the calibration primitives; see docs/methodology.md.
