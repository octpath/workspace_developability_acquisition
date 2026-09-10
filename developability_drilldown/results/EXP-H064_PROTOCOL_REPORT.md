# EXP-H064 — ESM-2 ARCH-6G CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6G`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `True`
- config_hash: `c15775016dc656e09c62c9d33d1aabc885ff31689964c312b36f0d4d56f7e7a2`
- n_trainable: 567713
- VAL_P/S/mean/worst: 0.445406 / 0.423780 / 0.434593 / 0.445406

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          50                  350          0.000086      0.512253           80           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/primary_k0_lr1e-04_seed101.pt 9b1e9e6632131a81   101
primary     1               0.0010 0.000010          14                   84          0.000990      0.521321           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/primary_k1_lr1e-03_seed101.pt 9b1e9e6632131a81   101
primary     2               0.0010 0.000010          40                  280          0.000910      0.349635           70           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/primary_k2_lr1e-03_seed101.pt 9b1e9e6632131a81   101
primary     3               0.0010 0.000010          34                  238          0.000935      0.423358           64           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/primary_k3_lr1e-03_seed101.pt 9b1e9e6632131a81   101
primary     4               0.0010 0.000010         139                  973          0.000227      0.416001          169           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/primary_k4_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     0               0.0010 0.000010          31                  217          0.000946      0.337420           61           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     1               0.0010 0.000010          41                  287          0.000905      0.438240           71           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     2               0.0010 0.000010          34                  238          0.000935      0.462359           64           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     3               0.0010 0.000010          58                  406          0.000814      0.435420           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     4               0.0010 0.000010          16                  112          0.000986      0.447794           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H064_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 9b1e9e6632131a81   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

