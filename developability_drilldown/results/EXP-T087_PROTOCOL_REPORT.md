# EXP-T087 — AbLingua ARCH-7 REG-only cross MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-7`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `5909f576c2435aa9940ce2eff97e179e4aeb536284860480b3834cf99ab56cb0`
- n_trainable: 551297
- VAL_P/S/mean/worst: 2.974722 / 3.040068 / 3.007395 / 3.040068
- TEST_P/S/mean/worst: 3.235371 / 3.276434 / 3.255903 / 3.276434

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          46                  322          0.000881      2.572762           76           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/primary_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     1               0.0010 0.000010           4                   24          0.000999      3.451898           34           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/primary_k1_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     2               0.0001 0.000001          24                  168          0.000097      3.150491           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/primary_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     3               0.0010 0.000010          11                   77          0.000994      2.545531           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/primary_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     4               0.0010 0.000010          23                  161          0.000971      3.150578           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/primary_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     0               0.0010 0.000010          36                  252          0.000927      2.714808           66           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/shadow_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     1               0.0001 0.000001          22                  154          0.000097      3.259464           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/shadow_k1_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     2               0.0010 0.000010          53                  371          0.000844      3.019719           83           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/shadow_k2_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     3               0.0010 0.000010          31                  217          0.000946      3.243297           61           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/shadow_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     4               0.0010 0.000010          24                  168          0.000968      2.966866           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T087_run/checkpoints/shadow_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101

