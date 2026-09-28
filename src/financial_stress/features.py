from __future__ import annotations

import re

import numpy as np
import pandas as pd

STATIC = [
    "arpu",
    "age",
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
    "x_90_d_activity_rate",
]
STATIC_CATS = ["gender", "region", "smartphone", "segment", "earning_pattern"]

CHANNELS = [
    "paybill",
    "merchantpay",
    "transfer_from_bank",
    "mm_send",
    "received",
    "deposit",
    "withdraw",
]
INFLOW = ["deposit", "received", "transfer_from_bank"]
OUTFLOW = ["withdraw", "mm_send", "merchantpay", "paybill"]

_MONTH_PATTERN = re.compile(r"^m([1-6])_(.+)$")


def temporal_bases(df: pd.DataFrame) -> list[str]:
    return sorted(
        {
            m.group(2)
            for c in df.columns
            if (m := _MONTH_PATTERN.match(c)) is not None
        }
    )


def advanced_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Target-free temporal and cross-channel representation.

    M1 is the most recent month and M6 is the oldest month.
    """
    bases = temporal_bases(df)
    out = df[STATIC].copy().reset_index(drop=True)

    x_centered = np.arange(1, 7, dtype=float) - 3.5
    slope_den = float((x_centered**2).sum())
    recency_weights = np.array(
        [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768], dtype=float
    )

    feats: dict[str, np.ndarray] = {}

    for base in bases:
        a = df[[f"m{i}_{base}" for i in range(1, 7)]].to_numpy(float)
        reversed_a = a[:, ::-1]
        d1 = a[:, :-1] - a[:, 1:]
        d2 = d1[:, :-1] - d1[:, 1:]

        mean = a.mean(axis=1)
        std = a.std(axis=1)
        maximum = a.max(axis=1)
        minimum = a.min(axis=1)

        feats[f"{base}__mean"] = mean
        feats[f"{base}__std"] = std
        feats[f"{base}__min"] = minimum
        feats[f"{base}__max"] = maximum
        feats[f"{base}__zero_n"] = (a == 0).sum(axis=1)
        feats[f"{base}__m1_minus_m6"] = a[:, 0] - a[:, -1]
        feats[f"{base}__recent3_minus_old3"] = (
            a[:, :3].mean(axis=1) - a[:, 3:].mean(axis=1)
        )
        feats[f"{base}__slope_old_to_recent"] = (
            (reversed_a - reversed_a.mean(axis=1, keepdims=True))
            * x_centered
        ).sum(axis=1) / slope_den

        q25 = np.quantile(a, 0.25, axis=1)
        q75 = np.quantile(a, 0.75, axis=1)
        feats[f"{base}__median"] = np.median(a, axis=1)
        feats[f"{base}__iqr"] = q75 - q25
        feats[f"{base}__recent2_minus_old2"] = (
            a[:, :2].mean(axis=1) - a[:, -2:].mean(axis=1)
        )
        feats[f"{base}__m1_minus_m2"] = d1[:, 0]
        feats[f"{base}__recent_accel"] = d2[:, 0]
        feats[f"{base}__mean_abs_diff"] = np.abs(d1).mean(axis=1)
        feats[f"{base}__max_abs_diff"] = np.abs(d1).max(axis=1)
        feats[f"{base}__diff_std"] = d1.std(axis=1)
        feats[f"{base}__sign_changes"] = np.sum(
            np.sign(d1[:, :-1]) * np.sign(d1[:, 1:]) < 0, axis=1
        )
        feats[f"{base}__monotonic_score"] = np.sign(d1).sum(axis=1)
        feats[f"{base}__argmax_month"] = np.argmax(a, axis=1) + 1
        feats[f"{base}__argmin_month"] = np.argmin(a, axis=1) + 1
        feats[f"{base}__cv"] = std / (np.abs(mean) + 1.0)
        feats[f"{base}__peak_to_current_frac"] = (
            maximum - a[:, 0]
        ) / (maximum + 1.0)
        feats[f"{base}__ewm_recent"] = np.average(
            a, axis=1, weights=recency_weights
        )

        log_old_to_recent = np.log1p(a[:, ::-1])
        feats[f"{base}__log_slope_old_to_recent"] = (
            (
                log_old_to_recent
                - log_old_to_recent.mean(axis=1, keepdims=True)
            )
            * x_centered
        ).sum(axis=1) / slope_den

        zero = a == 0
        active = ~zero
        feats[f"{base}__recent_zero_n3"] = zero[:, :3].sum(axis=1)
        feats[f"{base}__old_zero_n3"] = zero[:, 3:].sum(axis=1)
        feats[f"{base}__zero_transitions"] = np.sum(
            zero[:, :-1] != zero[:, 1:], axis=1
        )

        recent_active_run = np.zeros(len(df), dtype=np.int8)
        for j in range(6):
            recent_active_run += active[:, : j + 1].all(axis=1).astype(np.int8)
        feats[f"{base}__recent_active_run"] = recent_active_run

    monthly_names = [
        "inflow_value",
        "outflow_value",
        "net_value",
        "activity_value",
        "inflow_volume",
        "outflow_volume",
        "net_volume",
        "activity_volume",
        "active_channels",
        "bal_to_activity_log",
        "outflow_to_inflow_log",
    ]
    monthly = {name: [] for name in monthly_names}

    for month in range(1, 7):
        inflow = sum(
            df[f"m{month}_{ch}_total_value"].to_numpy(float)
            for ch in INFLOW
        )
        outflow = sum(
            df[f"m{month}_{ch}_total_value"].to_numpy(float)
            for ch in OUTFLOW
        )
        inflow_volume = sum(
            df[f"m{month}_{ch}_volume"].to_numpy(float)
            for ch in INFLOW
        )
        outflow_volume = sum(
            df[f"m{month}_{ch}_volume"].to_numpy(float)
            for ch in OUTFLOW
        )
        balance = df[f"m{month}_daily_avg_bal"].to_numpy(float)

        activity = inflow + outflow
        volume = inflow_volume + outflow_volume
        active_channels = sum(
            (df[f"m{month}_{ch}_volume"].to_numpy(float) > 0).astype(np.int8)
            for ch in CHANNELS
        )

        values = {
            "inflow_value": inflow,
            "outflow_value": outflow,
            "net_value": inflow - outflow,
            "activity_value": activity,
            "inflow_volume": inflow_volume,
            "outflow_volume": outflow_volume,
            "net_volume": inflow_volume - outflow_volume,
            "activity_volume": volume,
            "active_channels": active_channels,
            "bal_to_activity_log": np.log1p(balance) - np.log1p(activity),
            "outflow_to_inflow_log": np.log1p(outflow) - np.log1p(inflow),
        }

        for name, value in values.items():
            monthly[name].append(value)
            feats[f"m{month}__cross__{name}"] = value

    for name, arrays in monthly.items():
        a = np.column_stack(arrays)
        reversed_a = a[:, ::-1]
        d1 = a[:, :-1] - a[:, 1:]

        feats[f"cross__{name}__mean"] = a.mean(axis=1)
        feats[f"cross__{name}__std"] = a.std(axis=1)
        feats[f"cross__{name}__m1_minus_m6"] = a[:, 0] - a[:, -1]
        feats[f"cross__{name}__recent3_minus_old3"] = (
            a[:, :3].mean(axis=1) - a[:, 3:].mean(axis=1)
        )
        feats[f"cross__{name}__slope"] = (
            (reversed_a - reversed_a.mean(axis=1, keepdims=True))
            * x_centered
        ).sum(axis=1) / slope_den
        feats[f"cross__{name}__recent_diff"] = d1[:, 0]
        feats[f"cross__{name}__mean_abs_diff"] = np.abs(d1).mean(axis=1)

    feature_frame = pd.DataFrame(
        {name: np.asarray(value, dtype=np.float32) for name, value in feats.items()}
    )
    return pd.concat([out, feature_frame], axis=1)


def derivative_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compact sign-state features used by the LightGBM component."""
    feats: dict[str, np.ndarray] = {}

    for base in temporal_bases(df):
        a = df[[f"m{i}_{base}" for i in range(1, 7)]].to_numpy(np.float64)
        diff = a[:, :-1] - a[:, 1:]
        eps = (np.abs(a).mean(axis=1, keepdims=True) + 1.0) * 1e-6
        sign = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))

        feats[f"{base}__diff_pos_n"] = (sign > 0).sum(axis=1)
        feats[f"{base}__diff_neg_n"] = (sign < 0).sum(axis=1)
        feats[f"{base}__diff_zero_n"] = (sign == 0).sum(axis=1)
        feats[f"{base}__recent_dir"] = sign[:, 0]
        feats[f"{base}__old_dir"] = sign[:, -1]

    return pd.DataFrame(
        {name: np.asarray(value, dtype=np.float32) for name, value in feats.items()}
    )


def make_feature_views(predictors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = advanced_temporal(predictors).reset_index(drop=True)
    sign = derivative_direction_features(predictors).reset_index(drop=True)
    augmented = pd.concat([base, sign], axis=1)

    if base.columns.duplicated().any() or augmented.columns.duplicated().any():
        raise RuntimeError("duplicate engineered feature names")

    return base, augmented
