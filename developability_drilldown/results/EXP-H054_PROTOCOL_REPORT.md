# EXP-H054 — ESM-2 ARCH-H0 Heavy-only

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-H0`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `h_only` / `H_ONLY`
- geometry: `False`
- config_hash: `0741ab4e02c148417a368d33df7a633db7e3527c8d929ff158ceff468a488500`
- n_trainable: 485249
- VAL_P/S/mean/worst: 0.451982 / 0.438435 / 0.445209 / 0.451982

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010           9                   63          0.000996      0.487960           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/primary_k0_lr1e-03_seed101.pt 649131418599f207   101
primary     1               0.0010 0.000010          55                  330          0.000832      0.478481           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/primary_k1_lr1e-03_seed101.pt 649131418599f207   101
primary     2               0.0010 0.000010          25                  175          0.000965      0.362769           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/primary_k2_lr1e-03_seed101.pt 649131418599f207   101
primary     3               0.0001 0.000001          74                  518          0.000071      0.457889          104           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/primary_k3_lr1e-04_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010           6                   42          0.000998      0.470860           36           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0010 0.000010          18                  126          0.000982      0.400461           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 649131418599f207   101
 shadow     1               0.0010 0.000010          38                  266          0.000919      0.446497           68           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001          43                  301          0.000090      0.461629           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0001 0.000001         115                  805          0.000040      0.473349          145           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 649131418599f207   101
 shadow     4               0.0001 0.000001          60                  420          0.000080      0.410334           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H054_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 649131418599f207   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

