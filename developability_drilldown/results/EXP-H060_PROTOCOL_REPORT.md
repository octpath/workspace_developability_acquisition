# EXP-H060 — ESM-2 ARCH-4 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-4`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `c4c77d5d7dfc4a9d85a492462a416a5336175cbe93cfbcf92a257c2ca03d59c4`
- n_trainable: 501633
- VAL_P/S/mean/worst: 0.461805 / 0.442437 / 0.452121 / 0.461805

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          59                  413          0.000081      0.484091           89           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/primary_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010           9                   54          0.000996      0.529422           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          22                  154          0.000973      0.403336           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001         136                  952          0.000025      0.418468          166           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0001 0.000001          41                  287          0.000091      0.470900           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/primary_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0010 0.000010          41                  287          0.000905      0.410555           71           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0010 0.000010          20                  140          0.000978      0.472269           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0001 0.000001          77                  539          0.000069      0.467443          107           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          55                  385          0.000832      0.439108           85           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0001 0.000001          58                  406          0.000081      0.423911           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H060_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

