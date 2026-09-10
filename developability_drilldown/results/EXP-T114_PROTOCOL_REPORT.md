# EXP-T114 — AbLang2 ARCH-4 joint chain-specific dual CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-4`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `556377794d4351336892fe8cfdf66b10e33f33a2f40341acc87c1b42e0191b91`
- n_trainable: 399233
- VAL_P/S/mean/worst: 2.845561 / 2.913854 / 2.879707 / 2.913854

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          19                  133          0.000980      2.775299           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/primary_k0_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     1               0.0010 0.000010          25                  150          0.000965      3.231295           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/primary_k1_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     2               0.0010 0.000010          81                  567          0.000658      2.996094          111           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/primary_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     3               0.0001 0.000001          70                  490          0.000074      2.497579          100           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/primary_k3_lr1e-04_seed101.pt ec36df9f52a23f9e   101
primary     4               0.0010 0.000010          79                  553          0.000673      2.717679          109           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/primary_k4_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     0               0.0010 0.000010          29                  203          0.000953      2.535515           59           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/shadow_k0_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     1               0.0010 0.000010          24                  168          0.000968      2.789437           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/shadow_k1_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     2               0.0010 0.000010          32                  224          0.000942      2.982489           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/shadow_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     3               0.0100 0.000100           8                   56          0.009970      3.365571           38           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/shadow_k3_lr1e-02_seed101.pt ec36df9f52a23f9e   101
 shadow     4               0.0001 0.000001          70                  490          0.000074      2.893963          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T114_run/checkpoints/shadow_k4_lr1e-04_seed101.pt ec36df9f52a23f9e   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

