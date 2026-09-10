# EXP-T081 — AbLingua ARCH-3 joint unrestricted dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-3`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `eb5c0a3512650aa9f7afb1a11a52de90622bcb39f94b32d36ccfadcfb8906e28`
- n_trainable: 485249
- VAL_P/S/mean/worst: 2.989605 / 2.960474 / 2.975039 / 2.989605
- TEST_P/S/mean/worst: 3.431368 / 3.479819 / 3.455593 / 3.479819

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          16                  112          0.000986      2.827587           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/primary_k0_lr1e-03_seed101.pt 649131418599f207   101
primary     1               0.0010 0.000010          22                  132          0.000973      3.547468           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/primary_k1_lr1e-03_seed101.pt 649131418599f207   101
primary     2               0.0001 0.000001          24                  168          0.000097      2.977532           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/primary_k2_lr1e-04_seed101.pt 649131418599f207   101
primary     3               0.0001 0.000001          20                  140          0.000098      2.606079           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/primary_k3_lr1e-04_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010          57                  399          0.000821      2.976990           87           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0001 0.000001          21                  147          0.000098      2.641856           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 649131418599f207   101
 shadow     1               0.0001 0.000001          18                  126          0.000098      3.373175           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001          97                  679          0.000054      2.819951          127           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0001 0.000001          22                  154          0.000097      3.262118           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 649131418599f207   101
 shadow     4               0.0010 0.000010          26                  182          0.000962      2.705800           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T081_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 649131418599f207   101

