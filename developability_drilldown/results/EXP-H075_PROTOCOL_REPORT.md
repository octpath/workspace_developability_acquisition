# EXP-H075 — Scratch ARCH-4 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-4`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `False`
- config_hash: `ee99d10962d0c1f032e286842c284862d1f69a05cbe5d084edee0a156612b7ea`
- n_trainable: 324097
- VAL_P/S/mean/worst: 0.459294 / 0.462139 / 0.460716 / 0.462139

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          37                  259          0.000923      0.476602           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/primary_k0_lr1e-03_seed101.pt c6778122963cd518   101
primary     1               0.0010 0.000010          44                  264          0.000891      0.506410           74           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/primary_k1_lr1e-03_seed101.pt c6778122963cd518   101
primary     2               0.0010 0.000010          23                  161          0.000971      0.407982           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/primary_k2_lr1e-03_seed101.pt c6778122963cd518   101
primary     3               0.0001 0.000001         103                  721          0.000049      0.447111          133           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/primary_k3_lr1e-04_seed101.pt c6778122963cd518   101
primary     4               0.0010 0.000010          11                   77          0.000994      0.456351           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/primary_k4_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          39                  273          0.000914      0.463472           69           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          40                  280          0.000910      0.452898           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          18                  126          0.000982      0.459454           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0010 0.000010          12                   84          0.000993      0.528655           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/shadow_k3_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     4               0.0001 0.000001          83                  581          0.000064      0.404096          113           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H075_run/checkpoints/shadow_k4_lr1e-04_seed101.pt c6778122963cd518   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

