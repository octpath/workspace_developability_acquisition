# EXP-T098 — Scratch ARCH-5 gated cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-5`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `40f82f2118d810deada34a8ecadc8aa2368b715212ebd504b8f7ac21fc4c4293`
- n_trainable: 390147
- VAL_P/S/mean/worst: 3.041302 / 3.125757 / 3.083530 / 3.125757
- TEST_P/S/mean/worst: 3.208532 / 3.403675 / 3.306103 / 3.403675

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          84                  588          0.000064      3.077759          114           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/primary_k0_lr1e-04_seed101.pt efc5c97f142d97d9   101
primary     1               0.0001 0.000001         119                  714          0.000037      3.159572          149           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/primary_k1_lr1e-04_seed101.pt efc5c97f142d97d9   101
primary     2               0.0010 0.000010          17                  119          0.000984      3.326902           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/primary_k2_lr1e-03_seed101.pt efc5c97f142d97d9   101
primary     3               0.0010 0.000010          13                   91          0.000991      2.739632           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/primary_k3_lr1e-03_seed101.pt efc5c97f142d97d9   101
primary     4               0.0010 0.000010          17                  119          0.000984      2.897811           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/primary_k4_lr1e-03_seed101.pt efc5c97f142d97d9   101
 shadow     0               0.0001 0.000001          59                  413          0.000081      2.686951           89           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/shadow_k0_lr1e-04_seed101.pt efc5c97f142d97d9   101
 shadow     1               0.0010 0.000010          20                  140          0.000978      3.298845           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/shadow_k1_lr1e-03_seed101.pt efc5c97f142d97d9   101
 shadow     2               0.0010 0.000010          14                   98          0.000990      3.103594           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/shadow_k2_lr1e-03_seed101.pt efc5c97f142d97d9   101
 shadow     3               0.0010 0.000010          12                   84          0.000993      3.519067           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/shadow_k3_lr1e-03_seed101.pt efc5c97f142d97d9   101
 shadow     4               0.0010 0.000010          24                  168          0.000968      3.021749           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T098_run/checkpoints/shadow_k4_lr1e-03_seed101.pt efc5c97f142d97d9   101

