# EXP-T104 — Scratch ARCH-8 within-chain extra attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-8`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `2a9e8b7a795230b5d7dcb51303f0835c7430dfbeb16ac70c9f00c5dd42e1f329`
- n_trainable: 390145
- VAL_P/S/mean/worst: 2.980188 / 3.026486 / 3.003337 / 3.026486
- TEST_P/S/mean/worst: 3.312663 / 3.393869 / 3.353266 / 3.393869

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          69                  483          0.000074      3.153417           99           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/primary_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     1               0.0010 0.000010          21                  126          0.000976      3.122912           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     2               0.0001 0.000001          54                  378          0.000084      3.515056           84           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/primary_k2_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     3               0.0010 0.000010          14                   98          0.000990      2.634215           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/primary_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     4               0.0010 0.000010          31                  217          0.000946      2.465464           61           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     0               0.0001 0.000001          56                  392          0.000083      2.573002           86           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     1               0.0010 0.000010          10                   70          0.000995      3.372311           40           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     2               0.0010 0.000010          50                  350          0.000860      3.228063           80           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     3               0.0010 0.000010          14                   98          0.000990      3.123806           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     4               0.0010 0.000010          27                  189          0.000959      2.846376           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T104_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101

