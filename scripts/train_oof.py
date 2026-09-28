#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from financial_stress.features import make_feature_views
from financial_stress.metrics import competition_score
from financial_stress.modeling import run_oof
from financial_stress.validation import construction_slot_folds

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    if TARGET not in train.columns or ID_COL not in train.columns:
        raise ValueError("Train.csv must contain ID and target columns")

    y = train[TARGET].astype(int).to_numpy()
    predictors = train.drop(columns=[TARGET, ID_COL])

    folds = construction_slot_folds(predictors)
    x_base, x_sign = make_feature_views(predictors)

    predictions, metrics = run_oof(
        x_base=x_base,
        x_sign=x_sign,
        y=y,
        folds=folds,
    )

    metrics["composite_score"] = competition_score(
        metrics["auc"], metrics["logloss"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print("OOF complete")
    for key, value in metrics.items():
        print(f"{key}: {value:.12f}")


if __name__ == "__main__":
    main()
