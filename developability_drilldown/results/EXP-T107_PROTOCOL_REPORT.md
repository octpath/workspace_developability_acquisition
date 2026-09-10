# EXP-T107 — Scratch ARCH-6G geometry cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `True`
- config_hash: `988945786780b6e0e217a517af4cad217b797f987e2a79e815d1721ab82ebf9d`
- n_trainable: 406561
- VAL_P/S/mean/worst: 2.978038 / 3.044083 / 3.011060 / 3.044083

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          12                   84          0.000993      2.837760           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     1               0.0010 0.000010          23                  138          0.000971      3.338210           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     2               0.0001 0.000001          53                  371          0.000084      3.333444           83           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101
primary     3               0.0010 0.000010          16                  112          0.000986      2.730188           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     4               0.0010 0.000010          41                  287          0.000905      2.643715           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     0               0.0010 0.000010          18                  126          0.000982      2.566872           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     1               0.0001 0.000001          37                  259          0.000092      3.341236           67           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101
 shadow     2               0.0010 0.000010          12                   84          0.000993      2.959172           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     3               0.0010 0.000010           8                   56          0.000997      3.405641           38           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     4               0.0010 0.000010          14                   98          0.000990      2.951109           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T107_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

