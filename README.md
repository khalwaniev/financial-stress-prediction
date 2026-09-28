# Financial Stress Prediction

A leakage-aware tabular machine-learning pipeline for the **Zindi Financial Stress Prediction Challenge (August 2026)**.

The task is probabilistic binary classification: estimate whether a mobile-money user snapshot will experience liquidity stress in the next 30 days.

This repository is a cleaned portfolio version of the project. It focuses on the modeling, validation, feature engineering, calibration, and reproducibility work rather than the internal experiment-control archive used during the competition.

## Problem

The official data contain:

- **40,000** labeled training rows
- **30,000** test rows
- **182** predictors
- **15%** positive-class prevalence in Train
- **29 temporal feature families × 6 monthly observations**
- **8** profile/activity fields

A structural audit revealed **10,000 latent profile groups**, each containing exactly **4 Train + 3 Test rows**. The pair `arpu + x_90_d_activity_rate` is sufficient to reconstruct those groups exactly.

That structure materially affects validation. Random row splits alone would not be an adequate robustness check.

## Approach

The final production system combined three complementary gradient-boosting components:

1. **XGBoost** on engineered temporal and cross-channel features
2. **CatBoost** on the same core representation with native categorical handling
3. **LightGBM** on an augmented representation containing derivative-direction features

Each model family was trained with three fixed random seeds:

```text
20260911
20260917
20260923
```

The seed-averaged components were blended equally:

```text
prediction =
    (xgboost_mean + catboost_mean + sign_lightgbm_mean) / 3
```

A leakage-safe nested calibration stage compared no calibration, Platt scaling, and beta calibration. Beta calibration was selected for the production system.

## Validation design

Two validation worlds were treated separately:

### Primary: construction-slot OOF

The four labeled rows inside each latent profile group occupy four exact construction slots. Each slot was held out once, giving four complete OOF folds while preserving the deployment-like fact that the same profile groups appear in Train and Test.

### Stress test: group-disjoint OOF

Entire latent profile groups were separated between train and validation folds. This deliberately removes same-group information and tests whether model ranking survives a harder generalization regime.

Small improvements were treated cautiously and were checked across folds, seeds, grouped bootstrap replicates, calibration variants, and the group-disjoint stress world.

## Results

### Final raw ensemble

| Validation world | ROC-AUC | Log Loss | Brier |
|---|---:|---:|---:|
| Construction-slot OOF | **0.910899** | **0.246023** | **0.073947** |
| Group-disjoint OOF | **0.911080** | **0.245812** | **0.073865** |

### Calibration

| Method | ROC-AUC | Log Loss | Brier |
|---|---:|---:|---:|
| None | 0.910899 | 0.246023 | 0.073947 |
| Platt | 0.910877 | 0.244113 | 0.073654 |
| **Beta** | **0.910837** | **0.244022** | **0.073638** |

The project later audited the working composite objective as:

```text
score = 0.6 + 0.4 * AUC - (0.6 / 0.595) * LogLoss
```

Under that formula, beta calibration improved the construction-slot score from approximately **0.71627** to **0.71826**.

### Public leaderboard

The final production submission based on this pipeline achieved a public Zindi Multi Score of:

```text
0.721714257
```

The public leaderboard score was treated as external evidence, not as a hyperparameter-tuning target.

## Feature engineering

The feature layer is entirely target-free. It includes:

- six-month mean, standard deviation, min/max, median and IQR
- recent-vs-old contrasts
- linear and log-scale temporal slopes
- first- and second-order differences
- volatility and sign-change descriptors
- zero-state transitions and recent activity runs
- exponentially weighted recent activity
- cross-channel inflow/outflow/activity summaries
- balance-to-activity and outflow-to-inflow log ratios
- derivative-direction counts and recent/old direction states

The final representation used **963 core engineered features** plus **145 derivative-direction features**.

## Why the validation structure mattered

The dataset is not ordinary i.i.d. tabular data:

- each latent profile group appears four times in Train and three times in Test;
- the same 10,000 groups occur in the same order in both sets;
- profile features are constant within group;
- dynamic features vary substantially within group;
- target persistence within group is negligible.

This meant the project needed both deployment-like same-group validation and a stricter group-disjoint stress test.

## Repository layout

```text
.
├── configs/
│   └── final_model.json
├── data/
│   └── README.md
├── docs/
│   ├── experiments.md
│   └── methodology.md
├── results/
│   ├── model_comparison.csv
│   └── validation_summary.csv
├── scripts/
│   └── train_oof.py
├── src/
│   └── financial_stress/
│       ├── __init__.py
│       ├── calibration.py
│       ├── features.py
│       ├── metrics.py
│       ├── modeling.py
│       └── validation.py
└── tests/
    └── test_metrics.py
```

## Reproducibility

Competition data are intentionally not committed.

Place the official files under:

```text
data/raw/Train.csv
data/raw/Test.csv
```

Then install the project and run:

```bash
pip install -e .
python scripts/train_oof.py --train data/raw/Train.csv --output results/oof_predictions.csv
```

The training script reconstructs the latent profile groups from the two exact fingerprint fields and evaluates the final raw ensemble with four construction-slot folds.

The strict competition pipeline used a deeper nested refit for calibration; see [`docs/methodology.md`](docs/methodology.md).

## Notes

- No external feature dataset is used.
- `ID` is treated as an opaque submission identifier, not a predictive feature.
- Raw challenge data are excluded from Git.
- Public leaderboard feedback was not used to choose validation splits or tune model parameters.
- Experimental dead ends are summarized in [`docs/experiments.md`](docs/experiments.md).

## Original project

The full private research archive contains the complete experiment history, negative results, manifests, checksums, and execution evidence. This public repository intentionally presents the stable technical core in a conventional software/research layout.
