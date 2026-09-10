# EXP-T075 — Final DL Platform Baseline (T030 architecture)

- git: `cb5dda32be495a24274afe9768f85420d81ca314`
- platform: `DL_FOLDLOCAL_COSINE_V3`
- architecture: T030 FULL separate H/L (unchanged)
- seed: 101
- lr_grid=[1e-05, 0.0001, 0.001, 0.01]; patience=30; max_epochs=200; min_epochs=none
- cosine T_max=200; eta_min=0.01×lr0; no warmup/restart; no full-Dev
- n_trainable=501633

## Selected

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          37                  259          0.000092      2.728453           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/primary_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010          10                   60          0.000995      3.280419           40           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          54                  378          0.000838      3.131482           84           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001          19                  133          0.000098      2.627386           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0010 0.000010          33                  231          0.000939      3.091479           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/primary_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0001 0.000001          20                  140          0.000098      2.646367           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0001 0.000001          23                  161          0.000097      3.351653           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0001 0.000001          18                  126          0.000098      3.298451           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          12                   84          0.000993      3.305144           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0001 0.000001          21                  147          0.000098      3.008655           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T075_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101

## OOF VAL: 2.972246 / 3.120248 / 3.046247 / 3.120248
## OOF TEST: 3.512144 / 3.291950 / 3.402047 / 3.512144

| Prediction | Public | Private | Overall |
|------------|--------|---------|---------|
| Primary mean | 3.478785 | 3.287402 | 3.383093 |
| Primary median | 3.540297 | 3.306395 | 3.423346 |
| Shadow mean | 3.567248 | 3.341100 | 3.454174 |
| Shadow median | 3.579046 | 3.374153 | 3.476599 |

## Platform freeze

See `DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml` — status FROZEN after validation.

## STOP

