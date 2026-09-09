# Evaluation Audit — TFM-SDE on eICU Sepsis

## 1. Purpose

This document records how the evaluation metrics in the local Trajectory Flow Matching (TFM) repository are actually calculated and compares those calculations with the evaluation definitions described in the NeurIPS 2024 TFM paper.

The purpose is to avoid incorrectly comparing metrics that have similar names but different implementations.

### Experiment Context

- Dataset: eICU Sepsis
- Model: TFM-SDE
- State variables: Heart Rate (HR) and Mean Arterial Pressure (MAP)
- Conditioning variables: APACHE outcome probability and norepinephrine infusion
- Memory: 3 previous state vectors
- Model width: 256
- Test patients: 337
- Main reproduction seed: 42

### Important Principle

Metric names alone should not be used to establish equivalence between the paper and repository implementation. The complete calculation pathway must be checked before making a numerical comparison.

## 2. Repository `test_loss`

For the TFM-SDE model, testing is performed one patient at a time.

For each patient, the model generates a trajectory autoregressively. At every prediction interval, the predicted HR and MAP values are compared with the corresponding ground-truth HR and MAP values using the model's MSE loss function.

The repository then calculates:

1. MSE between predicted and true state at each trajectory interval.
2. The mean of these interval-level MSE values for that patient (`mse_all`).
3. PyTorch Lightning logs this value as `test_loss` with epoch-level aggregation across test batches.

Therefore, `test_loss` represents an aggregated patient-level trajectory prediction MSE rather than the metric stored in the JSON output as `Mean_MSE_test`.

### Fixed 200-Epoch Experiment

For the fixed 200-epoch, seed-42 experiment:

- `test_loss = 0.7302794456`
- `mse_loss_test = 0.7106946111`
- `Mean_MSE_test = 0.0008050483`

These values must not be treated as interchangeable because they are produced through different calculation pathways.

## 3. Repository `Mean_MSE_test`

The repository metric named `Mean_MSE` is calculated through the `variance_dist` evaluation pathway.

First, the repository computes consecutive trajectory differences:

- `pred_var = np.diff(pred, axis=0)`
- `true_var = np.diff(true, axis=0)`

For a trajectory with shape `[time_points, 2]`, where the two dimensions are HR and MAP, this converts the raw trajectory into changes between consecutive observations.

The distribution-distance function then calculates the mean of the predicted differences and the mean of the true differences:

- `mean(predicted differences)`
- `mean(true differences)`

It subsequently computes MSE between these two mean vectors.

Therefore, the repository field named `Mean_MSE_test` is not simply the ordinary MSE between every predicted trajectory state and its corresponding true state.

### Shape Verification

A temporary diagnostic was performed on an eICU Sepsis test patient.

Observed shapes:

- `pred_traj = (9, 2)`
- `full_traj = (9, 2)`

The two columns correspond to HR and MAP.

After `np.diff(..., axis=0)`, a `(9, 2)` trajectory becomes `(8, 2)`, representing consecutive changes in HR and MAP.

The temporary diagnostic print was removed after verification and did not modify the model algorithm or evaluation calculations.

## 4. Repository Distribution Metrics

The TFM-SDE configuration enables `variance_dist` in addition to `mse_loss` and `l1_loss`.

The `variance_dist` pathway first calculates consecutive trajectory changes using:

- `pred_var = np.diff(pred, axis=0)`
- `true_var = np.diff(true, axis=0)`

Distribution metrics are then calculated from these trajectory differences.

The reported metrics include:

- 1-Wasserstein
- 2-Wasserstein
- Linear MMD
- Polynomial MMD
- RBF MMD
- Mean-based distance metrics
- Median-based distance metrics

Therefore, the Wasserstein and MMD results produced through this pathway should be interpreted as comparisons involving consecutive HR/MAP trajectory changes rather than simply the raw HR/MAP states.

### Fixed 200-Epoch RBF MMD

For the fixed 200-epoch, seed-42 experiment:

- `RBF_MMD_test = 1.1566467921`

Across the three reproduction experiments:

| Experiment | RBF MMD |
|---|---:|
| 20 Epochs | 1.044869 |
| Early Stopped | 1.081169 |
| Fixed 200 Epochs | 1.156647 |

Lower RBF MMD indicates smaller discrepancy under this metric. In these experiments, extending training improved the pointwise error metrics but did not improve RBF MMD.

## 5. Paper-to-Repository Comparison Caution

The NeurIPS 2024 TFM paper and the repository use terminology that can appear similar, but metric names should not be assumed to represent identical calculations without tracing the implementation.

In particular:

- The repository's `Mean_MSE_test` is produced through the `variance_dist` pathway after consecutive trajectory differences are calculated.
- It therefore should not automatically be interpreted as the paper's trajectory-level Mean MSE solely because both use the term "Mean MSE".
- The repository's `test_loss` is generated from prediction-versus-ground-truth MSE during the autoregressive trajectory rollout and is conceptually closer to a trajectory prediction error.
- RBF MMD is also calculated after consecutive trajectory differences in the repository, which is relevant when comparing it with the paper's distributional evaluation.

### Reproduction Rule

Until metric equivalence is fully established, results should be reported using their exact repository names.

For example:

- `test_loss = 0.730279`
- `Mean_MSE_test = 0.000805`
- `RBF_MMD_test = 1.156647`

The analysis should not relabel one repository metric as a paper metric without documenting the calculation and aggregation procedure.

### Current Status

The fixed 200-epoch experiment is a successful reproduction run of the repository implementation on the eICU Sepsis dataset.

However, numerical agreement or disagreement with the published paper should only be claimed after confirming that preprocessing, metric definitions, aggregation, checkpoint selection, stochastic evaluation, and experimental settings are equivalent.

## 6. Audit Conclusions

The evaluation audit established the following:

1. TFM-SDE testing is performed one patient trajectory at a time.

2. Each trajectory contains two modeled state variables:
   - Heart Rate (HR)
   - Mean Arterial Pressure (MAP)

3. `test_loss` is derived from prediction-versus-ground-truth MSE during autoregressive trajectory rollout and is aggregated across test batches.

4. `Mean_MSE_test` is a separate metric generated through the `variance_dist` pathway and should not be assumed to be equivalent to `test_loss`.

5. The `variance_dist` pathway applies consecutive differences along the trajectory before calculating Wasserstein, MMD, mean-based, and median-based distance metrics.

6. The fixed 200-epoch experiment produced lower pointwise error metrics than the shorter experiments, while its RBF MMD was higher.

7. Therefore, longer training improved pointwise prediction error in the current experiments but did not produce a corresponding improvement in the evaluated trajectory-change distribution.

8. Direct numerical comparison with the published TFM paper requires confirmation of metric equivalence, preprocessing, aggregation, checkpoint selection, stochastic evaluation procedure, and experimental settings.

## Reproduction Principle

The authors' executable repository behavior and any later paper-consistent corrections should be maintained as separate experimental tracks.

The current results belong to the **repository reproduction track**.F

## 7. Public Repository Configuration vs. Paper Protocol

Git history was inspected to determine whether several training settings were introduced during the local reproduction work or were present in the authors' original public repository.

The original public repository commit:

- Commit: `6c733af20d1565d8dd99ee9aed2d7f99da4f05c9`
- Author: nZhangx
- Date: October 22, 2024
- Commit message: `feat: added files`

already contained the following configuration:

- `max_epochs: 200`
- `check_val_every_n_epoch: 10`
- `seed: 42`

Therefore, these settings were not introduced by the current reproduction work. They are part of the authors' publicly released executable configuration.

### Reproduction Implication

A distinction should be maintained between:

1. **Public Repository Reproduction**
   - Reproduce the behavior and configuration of the authors' released GitHub implementation.

2. **Paper-Protocol Reproduction**
   - Reproduce the experimental procedure described in the NeurIPS 2024 publication as closely as possible.

These two targets should not be silently combined when their documented settings differ.

Any differences between the public repository configuration and the published experimental protocol should be recorded explicitly and investigated before drawing conclusions about reproducibility.

### Important Configuration–Runner Discrepancy

Further inspection of the authors' original `src/main.py` revealed an important discrepancy between the configuration file and the executable training code.

Although the original `src/conf/config.yaml` contains:

- `check_val_every_n_epoch: 10`

the original October 22, 2024 `src/main.py` does not use this configuration value for the Trainer. Instead, it explicitly hard-codes:

- `check_val_every_n_epoch=50`

The original runner also enables:

- Early stopping: enabled
- Monitor: `val_loss`
- Patience: 3 validation checks
- Mode: `min`

Therefore, the effective training behavior of the authors' released public runner was validation every 50 epochs, despite the value of 10 stored in `config.yaml`.

With `max_epochs: 200`, the nominal validation schedule is therefore approximately:

- Epoch 50
- Epoch 100
- Epoch 150
- Epoch 200

This distinction is important because early-stopping patience counts validation checks rather than simply counting training epochs.

### Impact on Current Reproduction Experiments

The recent local experiments used the later configuration-driven implementation in which the Trainer reads `cfg.check_val_every_n_epoch`.

Consequently:

- The early-stopped local experiment used validation every 10 epochs and is not an exact reproduction of the original public runner's effective validation schedule.
- The fixed 200-epoch experiment also used validation every 10 epochs and disabled early stopping.
- Both experiments remain useful controlled reproduction experiments, but they should not be labeled as exact executions of the original released training schedule.

A future strict public-repository reproduction should preserve the original effective behavior separately from the current configurable implementation.