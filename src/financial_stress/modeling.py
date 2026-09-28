from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .features import STATIC_CATS
from .metrics import evaluate_binary

SEEDS = [20260911, 20260917, 20260923]

XGB_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 10.0,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "reg_alpha": 0.05,
    "gamma": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_bin": 256,
    "n_jobs": 4,
}

CAT_PARAMS = {
    "iterations": 250,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.9,
    "rsm": 0.8,
    "border_count": 64,
    "boosting_type": "Plain",
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "n_jobs": 4,
    "verbosity": -1,
}


def _one_hot(train: pd.DataFrame, valid: pd.DataFrame):
    numeric = [c for c in train.columns if c not in STATIC_CATS]
    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    train_cat = enc.fit_transform(train[STATIC_CATS])
    valid_cat = enc.transform(valid[STATIC_CATS])

    x_train = np.concatenate(
        [train_cat, train[numeric].to_numpy(np.float32)], axis=1
    )
    x_valid = np.concatenate(
        [valid_cat, valid[numeric].to_numpy(np.float32)], axis=1
    )
    return x_train, x_valid


def _catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STATIC_CATS:
        out[col] = out[col].fillna("__NA__").astype(str)
    return out


def fit_predict_fold(
    x_base: pd.DataFrame,
    x_sign: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit the three model families and return seed-averaged probabilities."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb_train, xgb_valid = _one_hot(
        x_base.iloc[train_idx], x_base.iloc[valid_idx]
    )
    xgb_predictions = []

    for seed in SEEDS:
        params = dict(XGB_PARAMS, random_state=seed)
        model = XGBClassifier(**params)
        model.fit(xgb_train, y[train_idx])
        xgb_predictions.append(model.predict_proba(xgb_valid)[:, 1])

    cat_train = _catboost_frame(x_base.iloc[train_idx])
    cat_valid = _catboost_frame(x_base.iloc[valid_idx])
    cat_predictions = []

    for seed in SEEDS:
        params = dict(CAT_PARAMS, random_seed=seed)
        model = CatBoostClassifier(**params)
        model.fit(cat_train, y[train_idx], cat_features=STATIC_CATS)
        cat_predictions.append(model.predict_proba(cat_valid)[:, 1])

    lgb_train, lgb_valid = _one_hot(
        x_sign.iloc[train_idx], x_sign.iloc[valid_idx]
    )
    lgb_predictions = []

    for seed in SEEDS:
        params = dict(LGBM_PARAMS, random_state=seed)
        model = LGBMClassifier(**params)
        model.fit(lgb_train, y[train_idx])
        lgb_predictions.append(model.predict_proba(lgb_valid)[:, 1])

    p_xgb = np.mean(np.column_stack(xgb_predictions), axis=1)
    p_cat = np.mean(np.column_stack(cat_predictions), axis=1)
    p_lgb = np.mean(np.column_stack(lgb_predictions), axis=1)
    p_ensemble = (p_xgb + p_cat + p_lgb) / 3.0

    return {
        "xgboost": p_xgb,
        "catboost": p_cat,
        "sign_lightgbm": p_lgb,
        "ensemble": p_ensemble,
    }


def run_oof(
    x_base: pd.DataFrame,
    x_sign: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run complete out-of-fold evaluation for a supplied fold assignment."""
    y = np.asarray(y, dtype=int)
    folds = np.asarray(folds, dtype=int)

    components = {
        name: np.full(len(y), np.nan, dtype=float)
        for name in ["xgboost", "catboost", "sign_lightgbm", "ensemble"]
    }

    for fold in sorted(np.unique(folds)):
        valid_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]

        pred = fit_predict_fold(
            x_base=x_base,
            x_sign=x_sign,
            y=y,
            train_idx=train_idx,
            valid_idx=valid_idx,
        )

        for name, values in pred.items():
            components[name][valid_idx] = values

    for name, values in components.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"incomplete OOF predictions for {name}")

    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "fold_id": folds,
            "y": y,
            **{f"pred_{name}": values for name, values in components.items()},
        }
    )

    metrics = evaluate_binary(y, components["ensemble"])
    return frame, metrics
