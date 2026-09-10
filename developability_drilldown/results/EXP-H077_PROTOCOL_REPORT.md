# EXP-H077 — Scratch ARCH-6 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `False`
- config_hash: `71a1addf4d258c2fed97e3a7f148257479c14b1a321195f3df85c902f9eb795f`
- n_trainable: 390145
- VAL_P/S/mean/worst: 0.464784 / 0.442971 / 0.453877 / 0.464784

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          39                  273          0.000091      0.524521           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/primary_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     1               0.0010 0.000010           6                   36          0.000998      0.529183           36           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     2               0.0010 0.000010          53                  371          0.000844      0.406165           83           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     3               0.0001 0.000001          48                  336          0.000087      0.473416           78           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/primary_k3_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     4               0.0010 0.000010          36                  252          0.000927      0.386754           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     0               0.0010 0.000010          48                  336          0.000871      0.413408           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     1               0.0010 0.000010          30                  210          0.000950      0.443890           60           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     2               0.0001 0.000001          95                  665          0.000055      0.439766          125           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     3               0.0010 0.000010          20                  140          0.000978      0.496143           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     4               0.0010 0.000010          13                   91          0.000991      0.420910           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H077_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

