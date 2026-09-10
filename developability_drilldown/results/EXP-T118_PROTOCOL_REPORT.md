# EXP-T118 — AbLang2 ARCH-6G geometry cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `True`
- config_hash: `613745eb42d15e94dc682e02482f504c38543634be2cde436d1f2c14b5a55671`
- n_trainable: 465313
- VAL_P/S/mean/worst: 2.828037 / 2.975666 / 2.901851 / 2.975666

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          63                  441          0.000078      2.851162           93           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/primary_k0_lr1e-04_seed101.pt 0efcf64722ac5a6a   101
primary     1               0.0010 0.000010          24                  144          0.000968      3.494402           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0efcf64722ac5a6a   101
primary     2               0.0010 0.000010          30                  210          0.000950      2.802821           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/primary_k2_lr1e-03_seed101.pt 0efcf64722ac5a6a   101
primary     3               0.0001 0.000001          65                  455          0.000077      2.308528           95           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/primary_k3_lr1e-04_seed101.pt 0efcf64722ac5a6a   101
primary     4               0.0010 0.000010          42                  294          0.000901      2.661724           72           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/primary_k4_lr1e-03_seed101.pt 0efcf64722ac5a6a   101
 shadow     0               0.0010 0.000010          16                  112          0.000986      2.516267           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 0efcf64722ac5a6a   101
 shadow     1               0.0001 0.000001          50                  350          0.000086      2.861204           80           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 0efcf64722ac5a6a   101
 shadow     2               0.0010 0.000010          84                  588          0.000636      3.031557          114           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 0efcf64722ac5a6a   101
 shadow     3               0.0100 0.000100          23                  161          0.009707      3.548520           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 0efcf64722ac5a6a   101
 shadow     4               0.0010 0.000010          19                  133          0.000980      2.917236           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T118_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 0efcf64722ac5a6a   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

