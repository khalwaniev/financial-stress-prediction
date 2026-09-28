# Financial Stress Prediction

Leakage-aware tabular machine learning for the **Zindi Financial Stress Prediction Challenge (August 2026)**.

The task is probabilistic binary classification: predict whether a mobile-money user snapshot will experience liquidity stress in the next 30 days.

## Highlights

- 40,000 labeled rows, 30,000 test rows, 182 predictors
- exact latent structure: 10,000 profile groups with 4 Train + 3 Test rows each
- deployment-aware four-fold out-of-fold validation plus group-disjoint stress testing
- target-free temporal and cross-channel feature engineering
- XGBoost + CatBoost + LightGBM seed ensemble
- leakage-safe nested probability calibration
- grouped uncertainty analysis and systematic negative-result tracking
- final public Zindi Multi Score: **0.721714257**

## Data structure

The predictor matrix contains:

- **29 temporal feature families × 6 monthly observations**
- **8 profile/activity fields**
- **15%** positive-class prevalence in Train

A structural audit revealed an exact latent grouping. The pair

```text
arpu + x_90_d_activity_rate
```

reconstructs all **10,000 latent profile groups** exactly, with four labeled Train rows and three unlabeled Test rows per group.

That structure materially affects validation: random row splits alone are not a sufficient robustness check.

## Validation

Two validation worlds were used.

### Construction-slot OOF

Each of the four labeled positions within every latent group is held out once. This gives complete four-fold OOF predictions while preserving the deployment-like fact that the same profile groups appear in Train and Test.

### Group-disjoint stress

Entire latent groups are separated between training and validation folds. This tests whether model quality survives without repeated profiles.

The final raw ensemble remained stable across both worlds:

| Validation world | ROC-AUC | Log Loss | Brier |
|---|---:|---:|---:|
| Construction-slot OOF | **0.910899** | **0.246023** | **0.073947** |
| Group-disjoint OOF | **0.911080** | **0.245812** | **0.073865** |

## Model

The production system combines three gradient-boosting components:

1. **XGBoost** on engineered temporal and cross-channel features
2. **CatBoost** on the core representation with native categorical handling
3. **LightGBM** on an augmented derivative-direction representation

Each family is trained with three fixed random seeds:

```text
20260911
20260917
20260923
```

Predictions are averaged within each family and then blended equally:

```text
prediction =
    (xgboost_mean + catboost_mean + sign_lightgbm_mean) / 3
```

The final feature representation contains **963 core engineered features** plus **145 derivative-direction features**.

## Feature engineering

The feature layer is target-free and includes:

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

## Calibration

The competition rewards both ranking quality and probability quality, so calibration was evaluated explicitly.

| Method | ROC-AUC | Log Loss | Brier |
|---|---:|---:|---:|
| None | 0.910899 | 0.246023 | 0.073947 |
| Platt | 0.910877 | 0.244113 | 0.073654 |
| **Beta** | **0.910837** | **0.244022** | **0.073638** |

Calibration was fitted with nested model refits so the calibrator training probabilities were independent of each outer validation fold.

The late-stage metric audit used:

```text
score = 0.6 + 0.4 * AUC - (0.6 / 0.595) * LogLoss
```

Under that objective, beta calibration improved the construction-slot score from approximately **0.71627** to **0.71826**.

## Public leaderboard

The final production submission achieved:

```text
Zindi Multi Score: 0.721714257
```

Leaderboard feedback was treated as external evidence rather than a hyperparameter-tuning target.

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
│       ├── calibration.py
│       ├── features.py
│       ├── metrics.py
│       ├── modeling.py
│       └── validation.py
└── tests/
    └── test_metrics.py
```

## Reproduce the OOF pipeline

Competition data are not committed. Place the official files at:

```text
data/raw/Train.csv
data/raw/Test.csv
```

Install the package and run:

```bash
pip install -e .
python scripts/train_oof.py \
  --train data/raw/Train.csv \
  --output results/oof_predictions.csv
```

The script reconstructs the latent profile groups and evaluates the raw production ensemble with four construction-slot folds.

The competition run used a deeper nested calibration procedure; see [`docs/methodology.md`](docs/methodology.md).

## Experiment discipline

Late-stage model families became highly correlated, so experiments were evaluated for **conditional information**, not just standalone accuracy. Several approaches were rejected after weak or non-replicating gains, including RealMLP, xRFM, TabDPT, a temporal residual network, a latent PCA/copula residual model, and foundation-model variants.

See [`docs/experiments.md`](docs/experiments.md) for the compact failure ledger and [`docs/methodology.md`](docs/methodology.md) for validation and calibration details.

## Notes

- no external feature dataset is used
- `ID` is treated as an opaque submission identifier, not a predictive feature
- raw challenge data and large model artifacts are excluded from Git
- reported metrics come from stored competition experiment outputs
