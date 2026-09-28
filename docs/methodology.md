# Methodology

## 1. Data structure

The project began with a structural audit rather than immediate hyperparameter tuning.

Train contains 40,000 rows and Test contains 30,000 rows. Both contain the same 182 predictors. The temporal block contains 29 feature families observed over six monthly positions, plus eight non-month-indexed profile/activity fields.

The strongest structural finding was an exact latent grouping:

```text
arpu + x_90_d_activity_rate
```

identifies 10,000 groups. Every group contains exactly four Train rows and three Test rows.

The group structure is real, but target persistence inside a group is negligible. This makes group membership important for validation geometry without making group-label averages a useful prediction shortcut.

## 2. Validation

### Construction-slot OOF

The primary validation holds out one of the four labeled positions inside every latent group. Across four folds, every training row is predicted exactly once.

This was chosen because deployment Test data contain the same profile groups as Train.

### Group-disjoint stress

A second validation world separates entire groups between train and validation. This is intentionally harsher and checks that gains do not depend entirely on repeated profiles.

The final raw ensemble remained stable:

| World | AUC | Log Loss |
|---|---:|---:|
| Construction slot | 0.910899 | 0.246023 |
| Group disjoint | 0.911080 | 0.245812 |

## 3. Feature engineering

Feature engineering was target-free and row-local.

Per temporal family, the representation includes level summaries, recent/old contrasts, slopes, first and second differences, volatility, zero-state dynamics, sign changes, recent activity runs, and recency-weighted summaries.

Cross-channel features aggregate the seven mobile-money channels into inflow, outflow, net activity, total activity, volume, balance-to-activity ratios, and related six-month trends.

A compact second representation summarizes the direction of monthly changes. These derivative-direction features were used by the LightGBM component.

## 4. Model ensemble

Three model families were deliberately retained:

- XGBoost on the core engineered representation
- CatBoost on the core representation with native categorical variables
- LightGBM on the augmented derivative-direction representation

Each model was trained with three fixed seeds. Predictions were averaged within each family and then combined with equal weights across families.

The equal blend was fixed rather than repeatedly optimized against a small validation gain.

## 5. Calibration

The competition metric rewards both discrimination and probability quality, so calibration mattered.

Three options were evaluated:

- none
- Platt scaling
- beta calibration

The strict competition implementation did **not** fit a calibrator directly on the same OOF vector used to evaluate an outer fold. Instead, inner model refits generated calibration-training probabilities that were independent of the outer fold labels.

That deeper nested procedure was intentionally more expensive but avoided calibration leakage.

Pooled construction-slot results:

| Calibration | AUC | Log Loss | Brier |
|---|---:|---:|---:|
| none | 0.910899 | 0.246023 | 0.073947 |
| Platt | 0.910877 | 0.244113 | 0.073654 |
| beta | 0.910837 | 0.244022 | 0.073638 |

Beta calibration was selected.

## 6. Uncertainty and robustness

The project used grouped paired bootstrap replicates rather than treating tiny point-estimate differences as certain improvements.

For the raw final candidate versus the preceding raw ensemble, the exact-score bootstrap difference had:

```text
mean delta: +0.0004865
95% interval: [+0.0002215, +0.0007444]
P(delta > 0): 1.0
```

The project also compared ordinary construction OOF, Test-weighted sensitivity, group-disjoint stress, calibration variants, and hidden-partition simulations.

## 7. Metric audit

The challenge combines ROC-AUC and Log Loss. During the endgame, the working exact objective was audited as:

```text
S = 0.6 + 0.4*AUC - (0.6/0.595)*LogLoss
```

This changed how small AUC/Log-Loss tradeoffs were interpreted, but did not change the frozen validation geometry.

## 8. Reproducibility

The private research archive retained:

- raw-data hashes
- deterministic split manifests
- fixed seeds
- model configs
- prediction manifests
- calibration artifacts
- grouped bootstrap results
- failed-experiment reports

The public portfolio intentionally omits competition data and bulky model artifacts.
