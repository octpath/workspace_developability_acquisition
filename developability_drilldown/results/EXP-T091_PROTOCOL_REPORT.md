# EXP-T091 — Scratch ARCH-1 separate dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-1`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `f22184ff811527c2674acd31531a0ec0899fe11ecca1d96e39532f84aaf612d4`
- n_trainable: 324097
- VAL_P/S/mean/worst: 3.032764 / 2.960919 / 2.996841 / 3.032764
- TEST_P/S/mean/worst: 3.365535 / 3.599481 / 3.482508 / 3.599481

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          95                  665          0.000055      3.014737          125           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/primary_k0_lr1e-04_seed101.pt c6778122963cd518   101
primary     1               0.0010 0.000010          37                  222          0.000923      3.218098           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/primary_k1_lr1e-03_seed101.pt c6778122963cd518   101
primary     2               0.0001 0.000001         103                  721          0.000049      3.278682          133           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/primary_k2_lr1e-04_seed101.pt c6778122963cd518   101
primary     3               0.0010 0.000010          18                  126          0.000982      2.591523           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/primary_k3_lr1e-03_seed101.pt c6778122963cd518   101
primary     4               0.0010 0.000010          30                  210          0.000950      3.055551           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/primary_k4_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          16                  112          0.000986      2.425497           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          23                  161          0.000971      3.048877           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          48                  336          0.000871      3.139084           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0010 0.000010          12                   84          0.000993      3.331126           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/shadow_k3_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     4               0.0010 0.000010          26                  182          0.000962      2.865175           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T091_run/checkpoints/shadow_k4_lr1e-03_seed101.pt c6778122963cd518   101

