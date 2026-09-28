import numpy as np
import pandas as pd

from financial_stress.validation import construction_slot_folds, latent_group_id


def _synthetic_group_frame():
    group = np.repeat(np.arange(10_000), 4)
    return pd.DataFrame(
        {
            "arpu": group.astype(float),
            "x_90_d_activity_rate": (group % 5_803).astype(float),
        }
    )


def test_exact_group_reconstruction():
    frame = _synthetic_group_frame()
    groups = latent_group_id(frame)
    counts = pd.Series(groups).value_counts().to_numpy()

    assert len(np.unique(groups)) == 10_000
    assert np.all(counts == 4)


def test_construction_slots_cover_each_position():
    frame = _synthetic_group_frame()
    folds = construction_slot_folds(frame)

    assert np.array_equal(np.bincount(folds), np.array([10_000] * 4))
    assert np.array_equal(folds[:8], np.array([0, 1, 2, 3, 0, 1, 2, 3]))
