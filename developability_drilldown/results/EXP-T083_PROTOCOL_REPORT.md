# EXP-T083 — AbLingua ARCH-5 gated cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-5`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `8772f744623122f0f41c2d75e6426f6a88eb85b6461ef61e76e7a8371a314166`
- n_trainable: 551299
- VAL_P/S/mean/worst: 3.023216 / 3.060923 / 3.042069 / 3.060923
- TEST_P/S/mean/worst: 3.382470 / 3.233085 / 3.307777 / 3.382470

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          31                  217          0.000946      2.693464           61           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/primary_k0_lr1e-03_seed101.pt befeb0de331c5d6d   101
primary     1               0.0001 0.000001          11                   66          0.000099      3.546159           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/primary_k1_lr1e-04_seed101.pt befeb0de331c5d6d   101
primary     2               0.0010 0.000010          14                   98          0.000990      2.991591           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/primary_k2_lr1e-03_seed101.pt befeb0de331c5d6d   101
primary     3               0.0010 0.000010           8                   56          0.000997      2.554554           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/primary_k3_lr1e-03_seed101.pt befeb0de331c5d6d   101
primary     4               0.0001 0.000001          12                   84          0.000099      3.324272           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/primary_k4_lr1e-04_seed101.pt befeb0de331c5d6d   101
 shadow     0               0.0001 0.000001          23                  161          0.000097      2.757661           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/shadow_k0_lr1e-04_seed101.pt befeb0de331c5d6d   101
 shadow     1               0.0010 0.000010          19                  133          0.000980      3.294649           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/shadow_k1_lr1e-03_seed101.pt befeb0de331c5d6d   101
 shadow     2               0.0001 0.000001         107                  749          0.000046      2.897392          137           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/shadow_k2_lr1e-04_seed101.pt befeb0de331c5d6d   101
 shadow     3               0.0010 0.000010          13                   91          0.000991      3.407195           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/shadow_k3_lr1e-03_seed101.pt befeb0de331c5d6d   101
 shadow     4               0.0010 0.000010          17                  119          0.000984      2.946372           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T083_run/checkpoints/shadow_k4_lr1e-03_seed101.pt befeb0de331c5d6d   101

