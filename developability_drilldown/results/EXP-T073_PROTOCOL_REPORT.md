# EXP-T073 — Protocol V2 Baseline (T030 architecture)

- git: `38b1e2a4be2208d71c364fb0277194e515b96d57`
- protocol: `DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2`
- architecture: T030 FULL separate H/L, REG_H||REG_L (unchanged)
- seed: 101 (single)
- lr_ref=0.0003, grid=[2.9999999999999997e-05, 8.999999999999999e-05, 0.0003, 0.0009]
- AdamW, max_epochs=200, patience=20, CosineAnnealingLR
- no nested CV; no full-Dev refit
- n_trainable=501633

## Selected LR

 scheme  fold  selected_lr  best_epoch  best_optimizer_step  best_val_mae  final_epoch  stopped_early                                                                                                                     checkpoint        init_hash  seed
primary     0      0.00090          12                   84      2.775854           32           True /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/primary_k0_lr0.0009_seed101.pt 1e0cea8ee76ae170   101
primary     1      0.00090          12                   72      3.201210           32           True /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/primary_k1_lr0.0009_seed101.pt 1e0cea8ee76ae170   101
primary     2      0.00090          12                   84      3.146419           32           True /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/primary_k2_lr0.0009_seed101.pt 1e0cea8ee76ae170   101
primary     3      0.00003          67                  469      2.589641           87           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/primary_k3_lr3e-05_seed101.pt 1e0cea8ee76ae170   101
primary     4      0.00009          41                  287      3.154061           61           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/primary_k4_lr9e-05_seed101.pt 1e0cea8ee76ae170   101
 shadow     0      0.00090           9                   63      2.615618           29           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/shadow_k0_lr0.0009_seed101.pt 1e0cea8ee76ae170   101
 shadow     1      0.00030           7                   49      3.301578           27           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/shadow_k1_lr0.0003_seed101.pt 1e0cea8ee76ae170   101
 shadow     2      0.00090          24                  168      3.232650           44           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/shadow_k2_lr0.0009_seed101.pt 1e0cea8ee76ae170   101
 shadow     3      0.00030          25                  175      3.349044           45           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/shadow_k3_lr0.0003_seed101.pt 1e0cea8ee76ae170   101
 shadow     4      0.00090          12                   84      2.915994           32           True  /workspace_developability_acquisition/developability_drilldown/results/EXP-T073_run/checkpoints/shadow_k4_lr0.0009_seed101.pt 1e0cea8ee76ae170   101

## OOF VAL (model-selection; not unbiased)

- VAL_P/S/mean/worst: 2.973623 / 3.081734 / 3.027679 / 3.081734

## OOF TEST (principal internal)

- TEST_P/S/mean/worst: 3.361936 / 3.144960 / 3.253448 / 3.361936

## External Test (4-way)

| Prediction | Public | Private | Overall |
|------------|--------|---------|---------|
| Primary mean | 3.541398 | 3.300341 | 3.420870 |
| Primary median | 3.490851 | 3.352425 | 3.421638 |
| Shadow mean | 3.636018 | 3.319072 | 3.477545 |
| Shadow median | 3.599165 | 3.311496 | 3.455330 |

## Contextual vs old T030

- historical T030 P/S: 3.317769 / 3.214610; Pub/Priv/Overall: 3.595529 / 3.255967 / 3.425748
- REPLAY P/S: 3.232677 / 3.201958; Pub/Priv/Overall: 3.595529 / 3.255966 / 3.425748
- NOTE: metrics are **not** protocol-equivalent (old used multi-seed + full-Dev refit).

## Readiness

Protocol V2 runs end-to-end and is usable as the architecture-series baseline,
with caveats: selected LR is fold-dependent (especially Primary), best epochs
are early (median 12; all stopped before 80), and Primary OOF TEST (3.362)
is weaker than REPLAY under the old protocol. External Overall for Primary mean
(~3.421) is close to historical T030 (~3.426) but metrics are not
protocol-equivalent. Architecture series not started.

## STOP

