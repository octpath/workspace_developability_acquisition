# EXP-T080 — AbLingua ARCH-1 separate dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-1`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `bedec9a7b086579a95a627cd8afe795631fbca2f3bbdab99d57f7110014c55c2`
- n_trainable: 485249
- VAL_P/S/mean/worst: 3.016081 / 3.035997 / 3.026039 / 3.035997
- TEST_P/S/mean/worst: 3.260250 / 3.306672 / 3.283461 / 3.306672

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          15                  105          0.000988      2.754428           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/primary_k0_lr1e-03_seed101.pt 649131418599f207   101
primary     1               0.0001 0.000001          12                   72          0.000099      3.531714           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/primary_k1_lr1e-04_seed101.pt 649131418599f207   101
primary     2               0.0001 0.000001          33                  231          0.000094      3.033859           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/primary_k2_lr1e-04_seed101.pt 649131418599f207   101
primary     3               0.0010 0.000010           8                   56          0.000997      2.513193           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/primary_k3_lr1e-03_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010          14                   98          0.000990      3.239272           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0001 0.000001          20                  140          0.000098      2.732957           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 649131418599f207   101
 shadow     1               0.0001 0.000001          34                  238          0.000093      3.261866           64           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001          69                  483          0.000074      2.894639           99           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0010 0.000010          31                  217          0.000946      3.249485           61           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 649131418599f207   101
 shadow     4               0.0010 0.000010           7                   49          0.000998      3.043834           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T080_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 649131418599f207   101

