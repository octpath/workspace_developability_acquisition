# EXP-T094 — Scratch ARCH-3 joint unrestricted dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-3`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `aee2d26d2b9089b1535ffd41d371b3826356bc478be2202d9868da4636e97273`
- n_trainable: 324097
- VAL_P/S/mean/worst: 3.103264 / 3.114210 / 3.108737 / 3.114210
- TEST_P/S/mean/worst: 3.187985 / 3.407729 / 3.297857 / 3.407729

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001         111                  777          0.000043      2.994660          141           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/primary_k0_lr1e-04_seed101.pt c6778122963cd518   101
primary     1               0.0010 0.000010          25                  150          0.000965      3.213549           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/primary_k1_lr1e-03_seed101.pt c6778122963cd518   101
primary     2               0.0010 0.000010          14                   98          0.000990      3.557326           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/primary_k2_lr1e-03_seed101.pt c6778122963cd518   101
primary     3               0.0010 0.000010          14                   98          0.000990      2.728022           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/primary_k3_lr1e-03_seed101.pt c6778122963cd518   101
primary     4               0.0010 0.000010          19                  133          0.000980      3.022712           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/primary_k4_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          21                  147          0.000976      2.852026           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0010 0.000010          33                  231          0.000939      3.171567           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/shadow_k1_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          13                   91          0.000991      2.932649           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0010 0.000010          13                   91          0.000991      3.450016           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/shadow_k3_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     4               0.0100 0.000100           7                   49          0.009978      3.162492           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T094_run/checkpoints/shadow_k4_lr1e-02_seed101.pt c6778122963cd518   101

