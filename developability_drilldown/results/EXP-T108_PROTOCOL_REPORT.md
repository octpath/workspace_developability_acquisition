# EXP-T108 — Scratch ARCH-6G geometry cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `True`
- config_hash: `6598aed53d098d17411f74b7846ae40a6b9bcb9a52927b8c04ac466d4bda6dac`
- n_trainable: 390177
- VAL_P/S/mean/worst: 2.936663 / 3.002687 / 2.969675 / 3.002687

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          25                  175          0.000965      2.829505           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     1               0.0010 0.000010          12                   72          0.000993      2.996507           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     2               0.0001 0.000001          44                  308          0.000089      3.311961           74           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8425f2f579fed963   101
primary     3               0.0010 0.000010           9                   63          0.000996      2.695729           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     4               0.0010 0.000010          16                  112          0.000986      2.851091           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     0               0.0010 0.000010          19                  133          0.000980      2.711211           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     1               0.0010 0.000010          13                   91          0.000991      3.191570           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     2               0.0010 0.000010          12                   84          0.000993      2.908710           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     3               0.0001 0.000001          46                  322          0.000088      3.438483           76           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 8425f2f579fed963   101
 shadow     4               0.0010 0.000010          18                  126          0.000982      2.758953           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T108_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8425f2f579fed963   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

