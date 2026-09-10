# EXP-H058 — ESM-2 ARCH-3 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-3`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `118b0a64acb005fbd0ee4178bf18dcfcd71197d67200a084eca935df3c47d128`
- n_trainable: 501633
- VAL_P/S/mean/worst: 0.460661 / 0.459621 / 0.460141 / 0.460661

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          18                  126          0.000982      0.501040           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/primary_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010          15                   90          0.000988      0.517233           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          40                  280          0.000910      0.374650           70           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0010 0.000010          36                  252          0.000927      0.441322           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/primary_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0010 0.000010          13                   91          0.000991      0.466031           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/primary_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0010 0.000010          44                  308          0.000891      0.422270           74           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0010 0.000010          38                  266          0.000919      0.482955           68           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0010 0.000010          16                  112          0.000986      0.469059           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          24                  168          0.000968      0.507873           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0001 0.000001          71                  497          0.000073      0.415608          101           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H058_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

