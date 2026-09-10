# EXP-H070 — Scratch ARCH-1 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-1`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `False`
- config_hash: `5ec3297a544942ffa60a5021a0c1677942000eb98ade5be99e0a7e72b6463af9`
- n_trainable: 324097
- VAL_P/S/mean/worst: 0.453531 / 0.459950 / 0.456741 / 0.459950

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          22                  154          0.000973      0.466469           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/primary_k0_lr1e-03_seed101.pt c6778122963cd518   101
primary     1               0.0100 0.000100          10                   60          0.009951      0.526213           40           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/primary_k1_lr1e-02_seed101.pt c6778122963cd518   101
primary     2               0.0010 0.000010          87                  609          0.000613      0.399864          117           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/primary_k2_lr1e-03_seed101.pt c6778122963cd518   101
primary     3               0.0001 0.000001          89                  623          0.000060      0.471093          119           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/primary_k3_lr1e-04_seed101.pt c6778122963cd518   101
primary     4               0.0010 0.000010          78                  546          0.000680      0.401342          108           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/primary_k4_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          27                  189          0.000959      0.452446           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          17                  119          0.000984      0.490231           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          40                  280          0.000910      0.470597           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0001 0.000001          64                  448          0.000078      0.515134           94           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/shadow_k3_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     4               0.0001 0.000001          94                  658          0.000056      0.369852          124           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H070_run/checkpoints/shadow_k4_lr1e-04_seed101.pt c6778122963cd518   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

