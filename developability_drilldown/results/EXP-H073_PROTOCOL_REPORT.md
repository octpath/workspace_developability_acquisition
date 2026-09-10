# EXP-H073 — Scratch ARCH-3 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-3`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `False`
- config_hash: `4a6a9fce3d32d72aa1973bb2434b550652ac620110b8a224b63132929e7b0c93`
- n_trainable: 324097
- VAL_P/S/mean/worst: 0.468682 / 0.438902 / 0.453792 / 0.468682

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          18                  126          0.000982      0.485789           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/primary_k0_lr1e-03_seed101.pt c6778122963cd518   101
primary     1               0.0100 0.000100           7                   42          0.009978      0.511083           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/primary_k1_lr1e-02_seed101.pt c6778122963cd518   101
primary     2               0.0010 0.000010          59                  413          0.000808      0.438994           89           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/primary_k2_lr1e-03_seed101.pt c6778122963cd518   101
primary     3               0.0001 0.000001         103                  721          0.000049      0.443742          133           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/primary_k3_lr1e-04_seed101.pt c6778122963cd518   101
primary     4               0.0001 0.000001          79                  553          0.000067      0.461942          109           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/primary_k4_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          63                  441          0.000783      0.385292           93           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          42                  294          0.000901      0.471215           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          23                  161          0.000971      0.460448           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0010 0.000010          35                  245          0.000931      0.482834           65           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/shadow_k3_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     4               0.0010 0.000010          19                  133          0.000980      0.395022           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H073_run/checkpoints/shadow_k4_lr1e-03_seed101.pt c6778122963cd518   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

