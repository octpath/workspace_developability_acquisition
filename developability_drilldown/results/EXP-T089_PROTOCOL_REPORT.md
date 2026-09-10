# EXP-T089 — AbLingua ARCH-8 within-chain extra attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-8`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `5176d65d0953191a6fc06d38e2ab476f3b8a3d5c4d158f2193cdf1453086f40f`
- n_trainable: 551297
- VAL_P/S/mean/worst: 3.037090 / 3.022799 / 3.029944 / 3.037090
- TEST_P/S/mean/worst: 3.332799 / 3.259449 / 3.296124 / 3.332799

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          24                  168          0.000968      2.720460           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/primary_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     1               0.0010 0.000010          21                  126          0.000976      3.552356           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/primary_k1_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     2               0.0001 0.000001          26                  182          0.000096      3.137855           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/primary_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     3               0.0001 0.000001          23                  161          0.000097      2.587151           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/primary_k3_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     4               0.0010 0.000010          33                  231          0.000939      3.181421           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/primary_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     0               0.0010 0.000010          14                   98          0.000990      2.683352           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/shadow_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     1               0.0001 0.000001          30                  210          0.000095      3.246830           60           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/shadow_k1_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     2               0.0001 0.000001          42                  294          0.000090      3.069517           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/shadow_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     3               0.0010 0.000010          15                  105          0.000988      3.243356           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/shadow_k3_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     4               0.0010 0.000010          29                  203          0.000953      2.874654           59           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T089_run/checkpoints/shadow_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101

