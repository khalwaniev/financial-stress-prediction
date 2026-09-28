# Experiment summary

The final model was not selected from a single model run. The project maintained a failure ledger so weak ideas were not repeatedly rediscovered.

## Promoted mechanism

### Temporal-sign ensemble

Adding derivative-direction features to a LightGBM component and blending it with XGBoost and CatBoost improved the protected inner development result.

On the 30,000-row inner construction OOF comparison:

| System | AUC | Log Loss | Brier |
|---|---:|---:|---:|
| Previous XGB + CatBoost ensemble | 0.906894 | 0.250720 | 0.075389 |
| XGB + CatBoost + sign-LightGBM | **0.907220** | **0.250343** | **0.075319** |

The full four-fold production run later reached raw construction OOF AUC 0.910899 and Log Loss 0.246023.

## Useful negative results

A large part of the work was deciding what **not** to keep.

### RealMLP

An initial development gain of roughly +0.0011 on the exact objective did not replicate. The replication-stage pooled delta became approximately -0.00035.

### xRFM

The tested configuration was negative relative to the incumbent.

### TabDPT

The tested tabular foundation-model path was materially negative.

### Pairwise XGBoost

Produced only a very small gain (~+0.00022) and was not promoted.

### Structured temporal residual network

A temporal residual mechanism produced only about +0.000036 exact-score improvement in Stage A and was rejected.

### Latent PCA/copula residual model

The latent residual component reduced the score and was rejected.

### TabPFN-3.5 zero-shot blend

Zero-shot TabPFN showed some ranking diversity, but a fixed 90/10 blend was not sufficiently stable across development folds to promote.

### Mitra-v2 full-weight adaptation

Full-weight adaptation executed successfully, but downstream meta-models found effectively no stable conditional information beyond the incumbent ensemble. The convex meta-model selected zero Mitra weight in both tested directions.

## General lesson

Late-stage gains were difficult because strong tree models became highly correlated. The most useful experiments were therefore those designed to test **conditional information** rather than simply adding another high-capacity learner.

That changed the research question from:

> Is this model strong by itself?

to:

> Does this model correct errors the incumbent does not already correct?

This distinction prevented several expensive but redundant models from being promoted.
