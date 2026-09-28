import numpy as np

from financial_stress.metrics import competition_score, evaluate_binary


def test_competition_score_known_point():
    score = competition_score(
        0.910836818627451,
        0.24402171293452257,
    )
    assert abs(score - 0.718262411886756) < 1e-12


def test_binary_metrics_are_finite():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.7, 0.9])
    result = evaluate_binary(y, p)
    assert all(np.isfinite(v) for v in result.values())
