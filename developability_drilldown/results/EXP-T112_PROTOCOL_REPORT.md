# EXP-T112 — AbLang2 ARCH-3 joint unrestricted dual CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-3`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `a819807502d4fcc122febd0b2e6ad2d1a7d2e9cf32e3dbd1d55447bf2685017c`
- n_trainable: 399233
- VAL_P/S/mean/worst: 2.787511 / 2.862188 / 2.824850 / 2.862188

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          33                  231          0.000939      2.888489           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/primary_k0_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     1               0.0010 0.000010          39                  234          0.000914      3.318980           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/primary_k1_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     2               0.0010 0.000010          27                  189          0.000959      2.968919           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/primary_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     3               0.0001 0.000001          70                  490          0.000074      2.369174          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/primary_k3_lr1e-04_seed101.pt ec36df9f52a23f9e   101
primary     4               0.0010 0.000010          70                  490          0.000737      2.372232          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/primary_k4_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     0               0.0010 0.000010          31                  217          0.000946      2.487627           61           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/shadow_k0_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     1               0.0010 0.000010          60                  420          0.000802      2.704569           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/shadow_k1_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     2               0.0010 0.000010          71                  497          0.000730      2.837360          101           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/shadow_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     3               0.0010 0.000010          40                  280          0.000910      3.473178           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/shadow_k3_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     4               0.0010 0.000010          27                  189          0.000959      2.800819           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T112_run/checkpoints/shadow_k4_lr1e-03_seed101.pt ec36df9f52a23f9e   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

