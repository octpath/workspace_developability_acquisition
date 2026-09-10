# EXP-T100 — Scratch ARCH-6 ungated cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-6`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `71f0a50058d505a5bd7d1bba8063608dbd961a171c6d3b3cd55920f7796fa6c4`
- n_trainable: 390145
- VAL_P/S/mean/worst: 2.993783 / 3.014238 / 3.004010 / 3.014238
- TEST_P/S/mean/worst: 3.365584 / 3.489141 / 3.427363 / 3.489141

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          46                  322          0.000088      3.005631           76           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/primary_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     1               0.0010 0.000010          12                   72          0.000993      2.997504           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     2               0.0010 0.000010          17                  119          0.000984      3.445713           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     3               0.0010 0.000010           9                   63          0.000996      2.695687           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/primary_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     4               0.0010 0.000010          30                  210          0.000950      2.823896           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     0               0.0010 0.000010          15                  105          0.000988      2.640680           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     1               0.0010 0.000010          19                  133          0.000980      3.107897           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     2               0.0010 0.000010          12                   84          0.000993      2.908974           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     3               0.0100 0.000100          10                   70          0.009951      3.495386           40           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     4               0.0001 0.000001          79                  553          0.000067      2.914888          109           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T100_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101

