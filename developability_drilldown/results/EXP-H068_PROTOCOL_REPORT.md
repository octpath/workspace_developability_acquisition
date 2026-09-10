# EXP-H068 — Scratch ARCH-H0 Heavy-only

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-H0`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `h_only` / `H_ONLY`
- geometry: `False`
- config_hash: `daf233c0e1ac60a8d840586ff3fca4b0aba3079bcf67ec0c778c947ba868525f`
- n_trainable: 324097
- VAL_P/S/mean/worst: 0.459137 / 0.438960 / 0.449048 / 0.459137

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          24                  168          0.000968      0.379353           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/primary_k0_lr1e-03_seed101.pt c6778122963cd518   101
primary     1               0.0001 0.000001          59                  354          0.000081      0.512907           89           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/primary_k1_lr1e-04_seed101.pt c6778122963cd518   101
primary     2               0.0001 0.000001          49                  343          0.000087      0.473068           79           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/primary_k2_lr1e-04_seed101.pt c6778122963cd518   101
primary     3               0.0001 0.000001          61                  427          0.000080      0.467926           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/primary_k3_lr1e-04_seed101.pt c6778122963cd518   101
primary     4               0.0001 0.000001          40                  280          0.000091      0.463243           70           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/primary_k4_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          47                  329          0.000876      0.433938           77           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          33                  231          0.000939      0.480816           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0001 0.000001          91                  637          0.000058      0.449046          121           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/shadow_k2_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     3               0.0010 0.000010          60                  420          0.000802      0.466919           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/shadow_k3_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     4               0.0010 0.000010          30                  210          0.000950      0.363365           60           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H068_run/checkpoints/shadow_k4_lr1e-03_seed101.pt c6778122963cd518   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

