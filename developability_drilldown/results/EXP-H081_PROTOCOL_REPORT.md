# EXP-H081 — Scratch ARCH-8 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-8`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `False`
- config_hash: `0ab4bc1ac835b625be8ad4aaedc04236fdc407fbc19fefce06470462d3c6c3a8`
- n_trainable: 390145
- VAL_P/S/mean/worst: 0.470365 / 0.441862 / 0.456113 / 0.470365

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          11                   77          0.000994      0.509045           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/primary_k0_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     1               0.0010 0.000010          12                   72          0.000993      0.515999           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     2               0.0010 0.000010          28                  196          0.000956      0.448640           58           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     3               0.0001 0.000001          94                  658          0.000056      0.449186          124           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/primary_k3_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     4               0.0010 0.000010          33                  231          0.000939      0.426321           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     0               0.0010 0.000010          40                  280          0.000910      0.405373           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     1               0.0010 0.000010          26                  182          0.000962      0.434953           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     2               0.0001 0.000001          46                  322          0.000088      0.506058           76           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     3               0.0001 0.000001         118                  826          0.000037      0.451773          148           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     4               0.0001 0.000001          75                  525          0.000070      0.411983          105           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H081_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

