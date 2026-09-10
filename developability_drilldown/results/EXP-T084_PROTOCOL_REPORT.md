# EXP-T084 — AbLingua ARCH-6 ungated cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-6`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `concat`
- config_hash: `bc24cb6ea2a5aae31d37ba45ea1e21edcc5e22a26e53f8243f9454807681e4c5`
- n_trainable: 567681
- VAL_P/S/mean/worst: 2.976885 / 3.078464 / 3.027675 / 3.078464
- TEST_P/S/mean/worst: 3.322090 / 3.264489 / 3.293290 / 3.322090

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          13                   91          0.000991      2.672941           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/primary_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     1               0.0010 0.000010           7                   42          0.000998      3.342656           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/primary_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     2               0.0001 0.000001          26                  182          0.000096      3.138069           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/primary_k2_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     3               0.0001 0.000001          18                  126          0.000098      2.560938           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/primary_k3_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     4               0.0001 0.000001          69                  483          0.000074      3.167889           99           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/primary_k4_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     0               0.0001 0.000001          20                  140          0.000098      2.596700           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     1               0.0010 0.000010           7                   49          0.000998      3.192779           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     2               0.0010 0.000010          23                  161          0.000971      3.208435           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     3               0.0001 0.000001          15                  105          0.000099      3.345254           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     4               0.0010 0.000010           6                   42          0.000998      3.055870           36           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T084_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 570e590061e5af2a   101

