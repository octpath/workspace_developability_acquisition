# EXP-T074 — Coarse Cosine Protocol V2 Platform (T030 architecture)

- git: `4b4958632bd3c47efdc2f5a037a5f894edf3c6b5`
- protocol: `DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2_COARSE_COSINE`
- architecture: T030 FULL separate H/L, REG_H||REG_L (unchanged)
- seed: 101 (single)
- lr_grid=[1e-05, 0.0001, 0.001, 0.01]
- AdamW; min_epochs=100; max_epochs=200; patience=20
- scheduler: cosine t=0..100 → eta_min=0.01×lr0, then hold; LR set at epoch start
- no nested CV; no full-Dev refit
- n_trainable=501633

## Selected runs

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010           7                   49          0.000991      2.716752          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/primary_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010           5                   30          0.000996      3.273595          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          16                  112          0.000946      3.065327          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001          19                  133          0.000092      2.634455          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0010 0.000010          52                  364          0.000489      3.208057          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/primary_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0010 0.000010           7                   49          0.000991      2.602089          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0010 0.000010          13                   91          0.000965      3.304160          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0010 0.000010          19                  133          0.000923      3.212158          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          11                   77          0.000976      3.376395          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0010 0.000010          15                  105          0.000953      2.999157          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T074_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101

## OOF VAL (selection-oriented)

- VAL_P/S/mean/worst: 2.979829 / 3.097439 / 3.038634 / 3.097439

## OOF TEST (principal internal)

- TEST_P/S/mean/worst: 3.286656 / 3.426023 / 3.356339 / 3.426023

## External Test (4-way)

| Prediction | Public | Private | Overall |
|------------|--------|---------|---------|
| Primary mean | 3.484075 | 3.274552 | 3.379313 |
| Primary median | 3.507680 | 3.379372 | 3.443526 |
| Shadow mean | 3.585692 | 3.353618 | 3.469655 |
| Shadow median | 3.637113 | 3.392790 | 3.514952 |

## vs EXP-T073 / old T030 (not protocol-equivalent)

- T073 OOF TEST P/S: 3.361936 / 3.144960; Pub/Priv/Overall (Primary mean): 3.541398 / 3.300341 / 3.420870
- hist T030 P/S: 3.317769 / 3.214610
- REPLAY P/S: 3.232677 / 3.201958

## Platform verdict

**ACCEPT_WITH_CAVEATS**

- Cosine gets a real horizon: all selected runs reach final_epoch=100.
- Coarse grid winners are interior (9×1e-3, 1×1e-4); no upper-bound monopoly; no NaN at 1e-2.
- Caveat: selected best_epoch still early (median 14); lr_at_best stays near initial LR, so trajectories do **not** clearly meet in one shared annealed band.
- External Primary mean Overall (3.379) improves vs T073 (3.421); metrics not protocol-equivalent to old T030.

Architecture series not started.

## STOP

