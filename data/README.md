# Data

Competition data are not committed to this portfolio repository.

Expected local layout:

```text
data/
└── raw/
    ├── Train.csv
    └── Test.csv
```

The original competition files contained:

- Train: 40,000 × 184
- Test: 30,000 × 183
- 182 predictors
- target: `liquidity_stress_next_30d`
- identifier: `ID`

The code uses the original column names supplied by the competition.
