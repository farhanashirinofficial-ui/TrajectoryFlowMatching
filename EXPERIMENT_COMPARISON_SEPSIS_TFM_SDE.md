# TFM-SDE Sepsis Experiment Comparison

## Purpose

This document compares TFM-SDE experiments on the eICU Sepsis dataset to study how training duration and early stopping affect predictive error and distributional similarity.

---

## Experiment 1 — 20 Epochs

Training configuration:
- Dataset: eICU Sepsis
- Model: TFM-SDE
- Maximum epochs: 20
- Early stopping: Not applicable for this short run

Test results:

| Metric | Result |
|---|---:|
| 1-Wasserstein | 0.632970 |
| 2-Wasserstein | 0.859191 |
| Linear MMD | -0.347070 |
| Mean L1 | 0.025463 |
| Mean L2 | 0.028653 |
| Mean MSE | 0.001366 |
| Median L1 | 0.041844 |
| Median L2 | 0.049584 |
| Median MSE | 0.004772 |
| Poly MMD | 2.409821 |
| RBF MMD | 1.044869 |
| L1 Loss | 0.754648 |
| MSE Loss | 1.017303 |
| Test Loss | 1.045584 |

---

## Experiment 2 — Early-Stopped Run

Training configuration:
- Dataset: eICU Sepsis
- Model: TFM-SDE
- Maximum epochs: 200
- Early stopping: Enabled
- Training stopped at approximately epoch 40

Test results:

| Metric | Result |
|---|---:|
| 1-Wasserstein | 0.635547 |
| 2-Wasserstein | 0.863734 |
| Linear MMD | -0.346678 |
| Mean L1 | 0.023519 |
| Mean L2 | 0.026218 |
| Mean MSE | 0.001147 |
| Median L1 | 0.040857 |
| Median L2 | 0.048781 |
| Median MSE | 0.004655 |
| Poly MMD | 2.412720 |
| RBF MMD | 1.081169 |
| L1 Loss | 0.696976 |
| MSE Loss | 0.922836 |
| Test Loss | 0.947875 |

Note: These values are from the previously recorded early-stopped experiment. The original local JSON result file is not currently available.

---

## Experiment 3 — Fixed 200 Epochs

Training configuration:
- Dataset: eICU Sepsis
- Model: TFM-SDE
- Seed: 42
- Maximum epochs: 200
- Early stopping: Disabled
- Validation frequency: Every 10 epochs
- Training completed: 200 epochs
- Training was resumed from a recovery checkpoint after interruption.

Final training loss at epoch 200:
- Train loss: 0.228525
- Validation loss: 0.755977

Test results:

| Metric | Result |
|---|---:|
| 1-Wasserstein | 0.638475 |
| 2-Wasserstein | 0.866182 |
| Linear MMD | -0.347437 |
| Mean L1 | 0.018905 |
| Mean L2 | 0.020973 |
| Mean MSE | 0.000805 |
| Median L1 | 0.040138 |
| Median L2 | 0.048125 |
| Median MSE | 0.004545 |
| Poly MMD | 2.411269 |
| RBF MMD | 1.156647 |
| L1 Loss | 0.598373 |
| MSE Loss | 0.710695 |
| Test Loss | 0.730279 |

---

## Main Comparison

| Metric | 20 Epochs | Early Stopped | Fixed 200 |
|---|---:|---:|---:|
| Test Loss | 1.045584 | 0.947875 | **0.730279** |
| L1 Loss | 0.754648 | 0.696976 | **0.598373** |
| MSE Loss | 1.017303 | 0.922836 | **0.710695** |
| 1-Wasserstein | **0.632970** | 0.635547 | 0.638475 |
| 2-Wasserstein | **0.859191** | 0.863734 | 0.866182 |
| RBF MMD | **1.044869** | 1.081169 | 1.156647 |

---

## Preliminary Observation

Increasing the training duration improved the pointwise loss metrics. The fixed 200-epoch experiment achieved lower test loss, L1 loss, and MSE loss than the shorter experiments.

However, the distributional metrics did not show the same improvement. The Wasserstein distances remained similar and increased slightly with longer training, while RBF MMD also increased.

Therefore, longer training improved pointwise predictive error but did not produce a corresponding improvement in distributional similarity.

Further analysis and repeated runs are required before drawing a final conclusion.