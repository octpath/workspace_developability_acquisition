# EXP-T096 — Scratch ARCH-4 joint chain-specific dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-4`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `961c4154722808d9905bdcdbb96a95ea434eef9735a313aa0cae46d457eee39f`
- n_trainable: 324097
- VAL_P/S/mean/worst: 3.040937 / 3.103488 / 3.072213 / 3.103488
- TEST_P/S/mean/worst: 3.252413 / 3.263832 / 3.258122 / 3.263832

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          18                  126          0.000982      2.976778           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/primary_k0_lr1e-03_seed101.pt c6778122963cd518   101
primary     1               0.0100 0.000100           4                   24          0.009995      3.331678           34           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/primary_k1_lr1e-02_seed101.pt c6778122963cd518   101
primary     2               0.0001 0.000001          98                  686          0.000053      3.251847          128           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/primary_k2_lr1e-04_seed101.pt c6778122963cd518   101
primary     3               0.0010 0.000010          15                  105          0.000988      2.624375           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/primary_k3_lr1e-03_seed101.pt c6778122963cd518   101
primary     4               0.0010 0.000010          25                  175          0.000965      3.012926           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/primary_k4_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     0               0.0010 0.000010          17                  119          0.000984      2.740365           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/shadow_k0_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     1               0.0001 0.000001          75                  525          0.000070      3.337271          105           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/shadow_k1_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     2               0.0010 0.000010          16                  112          0.000986      2.813183           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/shadow_k2_lr1e-03_seed101.pt c6778122963cd518   101
 shadow     3               0.0001 0.000001          84                  588          0.000064      3.469062          114           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/shadow_k3_lr1e-04_seed101.pt c6778122963cd518   101
 shadow     4               0.0010 0.000010          34                  238          0.000935      3.157485           64           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T096_run/checkpoints/shadow_k4_lr1e-03_seed101.pt c6778122963cd518   101

