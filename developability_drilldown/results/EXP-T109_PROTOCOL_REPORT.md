# EXP-T109 — AbLang2 ARCH-1 separate dual CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-1`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `248277a94fcb233f85945e36a16281ab27ca83103f5b500be1f0acd7e270b11a`
- n_trainable: 399233
- VAL_P/S/mean/worst: 2.930362 / 2.938914 / 2.934638 / 2.938914

## Selected LR

 scheme  fold  selected_initial_lr      eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0              0.00010 1.000000e-06          68                  476          0.000075      2.773676           98           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/primary_k0_lr1e-04_seed101.pt ec36df9f52a23f9e   101
primary     1              0.01000 1.000000e-04           7                   42          0.009978      3.457436           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/primary_k1_lr1e-02_seed101.pt ec36df9f52a23f9e   101
primary     2              0.00100 1.000000e-05          18                  126          0.000982      3.128137           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/primary_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
primary     3              0.00010 1.000000e-06          61                  427          0.000080      2.497185           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/primary_k3_lr1e-04_seed101.pt ec36df9f52a23f9e   101
primary     4              0.00100 1.000000e-05          44                  308          0.000891      2.783802           74           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/primary_k4_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     0              0.00010 1.000000e-06         134                  938          0.000026      2.543425          164           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/shadow_k0_lr1e-04_seed101.pt ec36df9f52a23f9e   101
 shadow     1              0.00100 1.000000e-05          28                  196          0.000956      2.883237           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/shadow_k1_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     2              0.00100 1.000000e-05          41                  287          0.000905      2.913808           71           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/shadow_k2_lr1e-03_seed101.pt ec36df9f52a23f9e   101
 shadow     3              0.00001 1.000000e-07          17                  119          0.000010      3.545212           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/shadow_k3_lr1e-05_seed101.pt ec36df9f52a23f9e   101
 shadow     4              0.00100 1.000000e-05          43                  301          0.000896      2.802301           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T109_run/checkpoints/shadow_k4_lr1e-03_seed101.pt ec36df9f52a23f9e   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

