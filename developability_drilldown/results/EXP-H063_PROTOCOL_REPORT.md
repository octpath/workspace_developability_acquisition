# EXP-H063 — ESM-2 ARCH-6 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `e24bfef786ca9eca3276e5414f907de6fbcb687b75de0b49469f51538d8e63d0`
- n_trainable: 551297
- VAL_P/S/mean/worst: 0.471888 / 0.433646 / 0.452767 / 0.471888

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          11                   77          0.000994      0.526726           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/primary_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     1               0.0010 0.000010          12                   72          0.000993      0.531352           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/primary_k1_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     2               0.0001 0.000001          85                  595          0.000063      0.343652          115           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/primary_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     3               0.0010 0.000010          32                  224          0.000942      0.478545           62           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/primary_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     4               0.0010 0.000010          17                  119          0.000984      0.475594           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/primary_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     0               0.0010 0.000010          60                  420          0.000802      0.377984           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/shadow_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     1               0.0010 0.000010          38                  266          0.000919      0.473663           68           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/shadow_k1_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     2               0.0001 0.000001          42                  294          0.000090      0.468461           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/shadow_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     3               0.0001 0.000001         102                  714          0.000050      0.420243          132           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/shadow_k3_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     4               0.0001 0.000001          53                  371          0.000084      0.430038           83           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H063_run/checkpoints/shadow_k4_lr1e-04_seed101.pt f4efb3ed83fd3486   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

