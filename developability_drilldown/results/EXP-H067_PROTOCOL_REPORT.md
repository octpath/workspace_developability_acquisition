# EXP-H067 — ESM-2 ARCH-8 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-8`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `85cf9b492aad64cdf2e4d6065117c68fbd0564732bc8bd5ef721fe52dd0b3a8e`
- n_trainable: 551297
- VAL_P/S/mean/worst: 0.451306 / 0.418359 / 0.434833 / 0.451306

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          11                   77          0.000994      0.522602           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/primary_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     1               0.0010 0.000010          14                   84          0.000990      0.528784           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/primary_k1_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     2               0.0010 0.000010          24                  168          0.000968      0.360154           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/primary_k2_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     3               0.0010 0.000010          58                  406          0.000814      0.466305           88           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/primary_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     4               0.0001 0.000001         125                  875          0.000032      0.374037          155           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/primary_k4_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     0               0.0010 0.000010          27                  189          0.000959      0.414897           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/shadow_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     1               0.0001 0.000001         130                  910          0.000029      0.381376          160           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/shadow_k1_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     2               0.0001 0.000001          43                  301          0.000090      0.467231           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/shadow_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     3               0.0010 0.000010          28                  196          0.000956      0.394156           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/shadow_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     4               0.0001 0.000001          53                  371          0.000084      0.435001           83           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H067_run/checkpoints/shadow_k4_lr1e-04_seed101.pt f4efb3ed83fd3486   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

