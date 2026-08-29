# TFM-SDE Real Sepsis Reproduction — 20 Epoch Run

## Repository state
Commit used: aa4822c

## Dataset
Config: src/conf/data/eICU_sepsis.yaml
Data file: data/eICU_sepsis_tfm.pkl

Split sizes:
- Train patients: 2689
- Validation patients: 336
- Test patients: 337

Time: time_scaled_v1, range [0, 1]

State variables:
- hr_normalized
- map_normalized

Condition variables:
- apache_outcome_prob
- norepi_inf_scaled

Memory: 3 previous state vectors
Seed: 42

## Model
TFM-SDE
Hydra model config: model=tfm_sde

## Exact command

python src/main.py data=eICU_sepsis model=tfm_sde wandb_logging=false max_epochs=20 limit_train_batches=1.0 limit_val_batches=1.0 limit_test_batches=1.0 num_sanity_val_steps=2 hydra.run.dir=.

## Stability result
- 20 epochs completed
- 418 training batches per epoch
- 285 test batches
- Diagnostic test trajectories: 287
- Affected trajectories: 0
- first_nonfinite_reported: False
- stage_events: none
- Model parameters remained finite

## Final test metrics
- 1-Wasserstein: 0.6329698532864764
- 2-Wasserstein: 0.8591913218621569
- Linear MMD: -0.3470695585878403
- Mean L1: 0.025462715994779086
- Mean L2: 0.028652914782105292
- Mean MSE: 0.0013655663280292573
- Median L1: 0.0418435870040731
- Median L2: 0.049583707307082114
- Median MSE: 0.004771723669270849
- Poly MMD: 2.409821449246323
- RBF MMD: 1.0448690063075015
- l1_loss_test: 0.754647970199585
- mse_loss_test: 1.0173033475875854
- noise_test_loss: 1.0455834865570068
- test_loss: 1.0455834865570068

## Interpretation
This is a full-data 20-epoch stability/reproduction run on the real Sepsis cohort.

It is not the final 200-epoch paper reproduction result.

The author's TFM-SDE implementation remained numerically stable on the real clinical time scale, with no detected nonfinite trajectories during testing.