# EXP-H065 — ESM-2 ARCH-6G MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6G`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `True`
- config_hash: `ad459bb000e51af80dbe8819f35b89e01d642abf686dce8af4fe13d76d958845`
- n_trainable: 551329
- VAL_P/S/mean/worst: 0.472007 / 0.435754 / 0.453880 / 0.472007

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          40                  280          0.000091      0.520206           70           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/primary_k0_lr1e-04_seed101.pt 466029b1a7d89082   101
primary     1               0.0010 0.000010          12                   72          0.000993      0.531350           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/primary_k1_lr1e-03_seed101.pt 466029b1a7d89082   101
primary     2               0.0010 0.000010          28                  196          0.000956      0.334935           58           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/primary_k2_lr1e-03_seed101.pt 466029b1a7d89082   101
primary     3               0.0010 0.000010          18                  126          0.000982      0.494339           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/primary_k3_lr1e-03_seed101.pt 466029b1a7d89082   101
primary     4               0.0010 0.000010          17                  119          0.000984      0.475844           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/primary_k4_lr1e-03_seed101.pt 466029b1a7d89082   101
 shadow     0               0.0010 0.000010          58                  406          0.000814      0.392872           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 466029b1a7d89082   101
 shadow     1               0.0010 0.000010          38                  266          0.000919      0.480244           68           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 466029b1a7d89082   101
 shadow     2               0.0001 0.000001          42                  294          0.000090      0.468459           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 466029b1a7d89082   101
 shadow     3               0.0001 0.000001         117                  819          0.000038      0.407696          147           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 466029b1a7d89082   101
 shadow     4               0.0010 0.000010          17                  119          0.000984      0.431714           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H065_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 466029b1a7d89082   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

