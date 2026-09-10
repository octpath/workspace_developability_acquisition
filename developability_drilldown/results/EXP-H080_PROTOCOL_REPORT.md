# EXP-H080 — Scratch ARCH-8 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-8`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `False`
- config_hash: `ea98e0e01513881cec7197c980ce2781fa9d3e106a38888a526bc527f1a89df2`
- n_trainable: 406529
- VAL_P/S/mean/worst: 0.464791 / 0.415229 / 0.440010 / 0.464791

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          61                  427          0.000796      0.423238           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8462d848103517b5   101
primary     1               0.0001 0.000001          43                  258          0.000090      0.509105           73           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/primary_k1_lr1e-04_seed101.pt 8462d848103517b5   101
primary     2               0.0010 0.000010          27                  189          0.000959      0.462290           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8462d848103517b5   101
primary     3               0.0010 0.000010          18                  126          0.000982      0.473998           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8462d848103517b5   101
primary     4               0.0010 0.000010          13                   91          0.000991      0.455238           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     0               0.0010 0.000010          25                  175          0.000965      0.402645           55           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     1               0.0010 0.000010          24                  168          0.000968      0.463177           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     2               0.0010 0.000010          24                  168          0.000968      0.425140           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     3               0.0010 0.000010          42                  294          0.000901      0.458753           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     4               0.0010 0.000010          70                  490          0.000737      0.325463          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H080_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8462d848103517b5   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

