from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GROUP_KEYS = ["arpu", "x_90_d_activity_rate"]


def latent_group_id(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct the exact latent profile grouping discovered in the data audit."""
    missing = [c for c in GROUP_KEYS if c not in df.columns]
    if missing:
        raise KeyError(f"missing group key columns: {missing}")

    key = pd.MultiIndex.from_frame(df[GROUP_KEYS])
    codes, _ = pd.factorize(key, sort=False)
    return codes.astype(np.int32)


def construction_slot_folds(df: pd.DataFrame) -> np.ndarray:
    """Create the four deployment-like folds from within-group row position.

    The official Train file contains exactly four contiguous rows per latent group.
    """
    groups = latent_group_id(df)
    counts = pd.Series(groups).value_counts().to_numpy()
    if len(counts) != 10_000 or not np.all(counts == 4):
        raise ValueError(
            "expected exactly 10,000 latent groups with four Train rows each"
        )

    slots = (
        pd.DataFrame({"group": groups})
        .groupby("group", sort=False)
        .cumcount()
        .to_numpy(np.int8)
    )

    if set(np.unique(slots)) != {0, 1, 2, 3}:
        raise RuntimeError("unexpected construction-slot structure")
    return slots


def group_disjoint_folds(df: pd.DataFrame, n_splits: int = 4) -> np.ndarray:
    """Deterministic group-disjoint stress folds.

    This is a compact portfolio implementation. The research archive retains the
    original frozen split manifest used for the reported stress-test metrics.
    """
    groups = latent_group_id(df)
    folds = np.full(len(df), -1, dtype=np.int8)

    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(df), dtype=np.int8)

    for fold, (_, valid_idx) in enumerate(splitter.split(dummy, dummy, groups)):
        folds[valid_idx] = fold

    if np.any(folds < 0):
        raise RuntimeError("incomplete group-disjoint fold assignment")
    return folds
