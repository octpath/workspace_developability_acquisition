# EXP-T097 — Scratch ARCH-5 gated cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-5`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `33b60d993f8cb2b82171e9d4962824307ce940db8aa819445540b1f29b604b40`
- n_trainable: 406531
- VAL_P/S/mean/worst: 2.923771 / 3.070546 / 2.997159 / 3.070546
- TEST_P/S/mean/worst: 3.378371 / 3.361811 / 3.370091 / 3.378371

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          14                   98          0.000990      2.545870           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/primary_k0_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101
primary     1               0.0010 0.000010          51                  306          0.000855      3.232760           81           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/primary_k1_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101
primary     2               0.0001 0.000001          97                  679          0.000054      3.196663          127           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/primary_k2_lr1e-04_seed101.pt 3d7ee2e4b77eb9f4   101
primary     3               0.0010 0.000010          22                  154          0.000973      2.702732           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/primary_k3_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101
primary     4               0.0010 0.000010          30                  210          0.000950      2.942985           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/primary_k4_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101
 shadow     0               0.0001 0.000001          77                  539          0.000069      2.677484          107           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 3d7ee2e4b77eb9f4   101
 shadow     1               0.0001 0.000001          48                  336          0.000087      3.409920           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 3d7ee2e4b77eb9f4   101
 shadow     2               0.0010 0.000010          20                  140          0.000978      3.070606           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101
 shadow     3               0.0001 0.000001          83                  581          0.000064      3.383442          113           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 3d7ee2e4b77eb9f4   101
 shadow     4               0.0010 0.000010          58                  406          0.000814      2.813783           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T097_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 3d7ee2e4b77eb9f4   101

