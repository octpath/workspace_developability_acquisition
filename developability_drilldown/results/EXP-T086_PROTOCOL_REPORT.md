# EXP-T086 — AbLingua ARCH-7 REG-only cross CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-7`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `concat`
- config_hash: `8ae1472c7ca8920ed2190f6d0f282e258d7926043bca2f1b682d0bbd11566a7e`
- n_trainable: 567681
- VAL_P/S/mean/worst: 2.921529 / 3.087614 / 3.004571 / 3.087614
- TEST_P/S/mean/worst: 3.373515 / 3.338105 / 3.355810 / 3.373515

## Selected LR

 scheme  fold  selected_initial_lr      eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0              0.00100 1.000000e-05           7                   49          0.000998      2.637962           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/primary_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     1              0.00010 1.000000e-06          13                   78          0.000099      3.361507           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/primary_k1_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     2              0.00100 1.000000e-05          77                  539          0.000687      3.010015          107           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/primary_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     3              0.00100 1.000000e-05           9                   63          0.000996      2.651040           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/primary_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     4              0.00100 1.000000e-05          55                  385          0.000832      2.942234           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/primary_k4_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     0              0.00100 1.000000e-05          14                   98          0.000990      2.609030           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     1              0.00100 1.000000e-05          11                   77          0.000994      3.225557           41           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     2              0.00001 1.000000e-07          97                  679          0.000005      3.304348          127           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/shadow_k2_lr1e-05_seed101.pt 570e590061e5af2a   101
 shadow     3              0.00100 1.000000e-05          18                  126          0.000982      3.282625           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     4              0.00010 1.000000e-06          23                  161          0.000097      3.025370           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T086_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 570e590061e5af2a   101

