# EXP-T123 — AbLang2 ARCH-8 within-chain extra-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-8`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `ef54dac3dba25ae36220d449203f43e516bfd3efb1b337febccdec9f583e4d35`
- n_trainable: 448897
- VAL_P/S/mean/worst: 2.819573 / 2.845489 / 2.832531 / 2.845489

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          13                   91          0.000991      2.878822           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8d6e8476a7702db3   101
primary     1               0.0100 0.000100           4                   24          0.009995      3.582210           34           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/primary_k1_lr1e-02_seed101.pt 8d6e8476a7702db3   101
primary     2               0.0001 0.000001          83                  581          0.000064      2.962445          113           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     3               0.0001 0.000001          55                  385          0.000083      2.300806           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     4               0.0010 0.000010          48                  336          0.000871      2.347898           78           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     0               0.0010 0.000010          53                  371          0.000844      2.564440           83           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     1               0.0010 0.000010          23                  161          0.000971      2.778085           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     2               0.0010 0.000010          53                  371          0.000844      2.518328           83           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     3               0.0100 0.000100          17                  119          0.009844      3.547374           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 8d6e8476a7702db3   101
 shadow     4               0.0010 0.000010          24                  168          0.000968      2.806065           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T123_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8d6e8476a7702db3   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

