# EXP-T110 — AbLang2 ARCH-1 separate dual MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-1`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `6de920e8134e0fd9c7cfb752daf2a2b3a7340796552d05bf8770ac6068be13d0`
- n_trainable: 382849
- VAL_P/S/mean/worst: 2.861236 / 2.773900 / 2.817568 / 2.861236

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          68                  476          0.000075      2.622815           98           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/primary_k0_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     1               0.0010 0.000010          37                  222          0.000923      3.588308           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/primary_k1_lr1e-03_seed101.pt 13358cb1e3782e93   101
primary     2               0.0001 0.000001          77                  539          0.000069      2.797609          107           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/primary_k2_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     3               0.0001 0.000001          78                  546          0.000068      2.330093          108           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/primary_k3_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     4               0.0010 0.000010          37                  259          0.000923      2.952085           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/primary_k4_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     0               0.0010 0.000010          82                  574          0.000651      2.380531          112           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     1               0.0001 0.000001         101                  707          0.000051      2.761713          131           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 13358cb1e3782e93   101
 shadow     2               0.0010 0.000010          28                  196          0.000956      2.603307           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     3               0.0100 0.000100          21                  147          0.009758      3.445685           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 13358cb1e3782e93   101
 shadow     4               0.0001 0.000001         102                  714          0.000050      2.669565          132           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T110_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 13358cb1e3782e93   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

