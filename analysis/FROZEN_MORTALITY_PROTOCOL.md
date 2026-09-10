# Frozen mortality-stratification protocol, version 1

This document preserves the completed, prespecified development experiment before
any landmark experiment. The implementation is `run_frozen_mortality_models.py`.
Generated results are retained locally under analysis/outputs/mortality_frozen_v1/ and are excluded from Git. Changes to later experiments must not replace this protocol.

## Endpoint and common cohorts

- Endpoint: `HOSP_MORT`, death = 1 and survival = 0.
- Primary common cohort for all Groups A-I: 2,261 training patients and 292
  validation patients, defined by the Phase 1 TFM patient feature tables.
- Join one-to-one on `HADM_ID`; require unique IDs, binary nonmissing outcomes,
  and no training/validation ID overlap. IDs are never predictors.
- Effective TFM formulation: memory = 3, consecutive observation pairs
  (`train_consecutive=false`), at least nine observations per eligible patient.
- Exclude `apache_outcome_prob`, `ICU_MORT`, and `label` from all predictors.
- `n_observations` and `n_intervals` are measurement-process metadata, not learned
  TFM features, and are excluded from all primary models.
- No access to or evaluation of the reserved test split. Validation is for
  development only; final evaluation requires a separately frozen protocol.

## Feature branches and preprocessing

All fitted preprocessing is learned exclusively from the fitting training fold.
After C selection, each complete pipeline is refitted on the full applicable
training cohort and applied unchanged to validation. No preprocessing is fitted
on validation. Combined models concatenate branches without another overall scaler.

### APACHE

Preserve raw `apache` in source and cohort CSVs, including -1. Only inside the
modeling branch, treat -1 or explicit missingness as unavailable and create a
0/1 `apache_unavailable` indicator. Convert unavailable score values to missing
in an internal copy, impute using the median available APACHE value in the fitting
training fold, and standardize the imputed score. Pass the unavailable indicator
without scaling. Fail on an unrecognized negative code, infinity, or a fitting
fold with no available APACHE scores.

The -1 interpretation is provisional: official eICU material documents the
missing-data convention, but the local repository does not establish the upstream
derivation of `apache`. This benchmark fits raw APACHE locally; it is not the
original APACHE IV mortality prediction equation.

### Conventional HR/MAP trajectories: 14 features

For each of `hr_normalized` and `map_normalized`, calculate:

1. Mean.
2. Population standard deviation (`ddof=0`).
3. Minimum.
4. Maximum.
5. First value.
6. Last value.
7. Linear slope against `time_scaled_v1`.

Use every available observation for each eligible patient, sorted by
`time_scaled_v1`, with equal observation weighting. Define slope as
`sum((t - mean(t)) * (x - mean(x))) / sum((t - mean(t)) ** 2)`.
Retain supplied normalized units; do not interpret slopes as per-hour changes.
Standardize the 14 summaries within the fitting fold. No thresholds, clipping,
nonlinear transforms, or outcome-driven feature selection are used. Invalid time
ordering or nonfinite inputs cause failure rather than silent exclusion.

No conventional norepinephrine summaries are included.

### Hidden TFM trajectory representation: 16 components

Use all 512 existing hidden summaries: mean and population SD of each of the 256
flow hidden activations. Their Phase 1 origin is the deterministic interval
midpoint (`u=0.5`), `flow_model.net[5]`, with observed three-step HR/MAP history,
no Gaussian jitter, no rollout, and no Brownian draw.

The exact fitting-fold branch is:

`512 summaries -> StandardScaler -> PCA(n_components=16, svd_solver="full", whiten=False) -> StandardScaler`

The final scaler standardizes the component scores. Use the same deterministic
hidden transformation across relevant groups for a given fitting fold.

### Diffusion-related/noise-amplitude representation: two features

- `noise_amplitude_abs_mean`
- `noise_amplitude_abs_std`

These are equal-interval mean and population SD of the absolute raw scalar
noise-network output at deterministic midpoints. Standardize both within the
fitting fold. Signed-amplitude summaries remain audit artifacts and are excluded.
No log transforms, squared amplitudes, or additional embeddings are included.
These features must not be called calibrated uncertainty.

## Groups A-I

Predictor counts exclude the intercept.

| Group | Branches | Predictor count |
| --- | --- | ---: |
| A | APACHE | 2 |
| B | Conventional HR/MAP | 14 |
| C | Hidden PCA | 16 |
| D | Absolute noise amplitude | 2 |
| E | Hidden PCA + absolute noise amplitude | 18 |
| F | APACHE + hidden PCA | 18 |
| G | APACHE + hidden PCA + absolute noise amplitude | 20 |
| H | APACHE + conventional HR/MAP | 16 |
| I | APACHE + conventional HR/MAP + hidden PCA + absolute noise amplitude | 34 |

## Classifier and training-only selection

- Ridge logistic regression: pure L2 penalty, `solver="lbfgs"`.
- `fit_intercept=True`, `class_weight=None`, `tol=1e-6`, `max_iter=5000`,
  `warm_start=False`.
- No oversampling, undersampling, SMOTE, or additional sample weights.
- C grid: `0.001, 0.01, 0.1, 1, 10, 100`.
- One fixed five-fold `StratifiedKFold`, `shuffle=True`, `random_state=42`,
  with identical fold indices across A-I in each analysis.
- Select C separately for each group by minimum mean held-out fold log loss.
  Ties within `1e-6` favor smaller C (stronger regularization).
- Resolve and document convergence failures before inspecting validation
  predictions; no validation-driven specification changes.
- Select and refit both primary and complete-case analyses before generating
  validation predictions.

Unweighted fitting targets the observed cohort probability distribution.
Balanced class weighting would change the objective and is not part of this
initial protocol.

## Validation metrics and paired bootstrap

Primary metric: log loss. Secondary metrics: AUROC, average precision (AP, not
trapezoidal PR area), binary Brier score, mean predicted risk, observed mortality,
calibration intercept, and calibration slope.

Calibration uses a joint unpenalized intercept/slope regression on prediction
logits; probabilities are clipped only to floating-point epsilon for the diagnostic
logit calculation. This is not recalibration. Report five equal-frequency bins,
resolving prediction ties by original cohort row order. Record unavailable or
failed calibration estimates explicitly.

Use 2,000 ordinary paired patient bootstrap attempts on validation, seed 42,
sharing sampled patient indices across all nine models. Discard and count
single-class resamples. Report percentile 95% intervals (2.5th and 97.5th
percentiles). These intervals condition on fitted models and do not include
training or hyperparameter-selection uncertainty.

Report every prespecified group and comparison, selected C, cohort/death counts,
and convergence. No threshold optimization, validation recalibration,
post-validation tuning, or alternative exploratory model is permitted.

## Comparison hierarchy and decision rule

Primary: **I versus H**, testing incremental predictive information from the TFM
representation beyond APACHE plus conventional HR/MAP summaries.

`delta_logloss = logloss(H) - logloss(I)`

Positive values favor I. A paired 95% interval entirely above zero constitutes
development evidence of improvement on the primary metric. An interval crossing
zero is inconclusive. Secondary metrics cannot replace the primary rule.

Prespecified secondary comparisons, in order:

1. G versus A (important secondary).
2. F versus A.
3. G versus F.
4. E versus C.
5. C versus B.

For each comparison, delta log loss is baseline minus augmented model log loss.
The implementation also reports prespecified comparisons' secondary discrimination
and Brier differences, with positive values favoring the augmented model.

## Complete-case APACHE sensitivity

Apply the same APACHE-available restriction to all Groups A-I, including groups
without APACHE predictors. The completed sensitivity cohort is 2,250 training
patients and 292 validation patients. Refit all preprocessing, PCA, CV selection,
and classifiers using only the restricted training cohort, with the same frozen
settings and comparison hierarchy. No APACHE imputation is needed and the
unavailable indicator is identically zero. Report separately; do not use the
sensitivity results to select a primary analysis.

## Completed primary comparison results

These are the recorded results, rounded to six decimal places, without a change
to the decision rule.

| Analysis | I versus H delta log loss | 95% paired-bootstrap CI |
| --- | ---: | --- |
| Primary | -0.001317 | [-0.014495, 0.012554] |
| Complete-case APACHE | -0.001695 | [-0.015028, 0.012371] |

The primary CI crosses zero and the prespecified improvement criterion was not
met. The complete-case CI also crosses zero. Neither result establishes clinical
usefulness or proves absence of incremental information.

## Interpretation limits

- This is whole-trajectory mortality risk stratification using available
  observation spans, not fixed-landmark early prediction.
- TFM is norepinephrine-conditioned; the conventional comparator is not fully
  information-matched. Differences cannot be attributed solely to representation
  learning from identical inputs.
- The frozen TFM representation was learned from the full training cohort;
  downstream CV selects regularization, not the entire TFM learning procedure.
- Validation contributed to TFM checkpoint selection and remains developmental.
- Diffusion-related/noise-amplitude features are not calibrated uncertainty.
- No test data was accessed for this completed development experiment. A later
  landmark experiment must be documented separately without rewriting this record.
