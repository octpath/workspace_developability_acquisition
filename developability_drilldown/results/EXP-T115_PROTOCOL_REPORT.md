# EXP-T115 — AbLang2 ARCH-4 joint chain-specific dual MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-4`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `a687bc9218a004667d36e3dff9a8419d8d973de7c7c6925bda73a73cba5be865`
- n_trainable: 382849
- VAL_P/S/mean/worst: 2.806355 / 2.886529 / 2.846442 / 2.886529

## Selected LR

 scheme  fold  selected_initial_lr      eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0              0.00100 1.000000e-05          23                  161          0.000971      2.647405           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/primary_k0_lr1e-03_seed101.pt 13358cb1e3782e93   101
primary     1              0.00100 1.000000e-05          45                  270          0.000886      3.401896           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/primary_k1_lr1e-03_seed101.pt 13358cb1e3782e93   101
primary     2              0.00010 1.000000e-06         108                  756          0.000045      2.765203          138           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/primary_k2_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     3              0.00010 1.000000e-06          72                  504          0.000072      2.224942          102           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/primary_k3_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     4              0.00100 1.000000e-05          30                  210          0.000950      2.978685           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/primary_k4_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     0              0.00010 1.000000e-06          65                  455          0.000077      2.642655           95           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 13358cb1e3782e93   101
 shadow     1              0.00010 1.000000e-06         129                  903          0.000029      2.800252          159           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 13358cb1e3782e93   101
 shadow     2              0.00100 1.000000e-05          24                  168          0.000968      2.777203           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     3              0.00001 1.000000e-07           4                   28          0.000010      3.544868           34           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/shadow_k3_lr1e-05_seed101.pt 13358cb1e3782e93   101
 shadow     4              0.00100 1.000000e-05          32                  224          0.000942      2.654717           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T115_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 13358cb1e3782e93   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

