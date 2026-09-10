# EXP-T088 — AbLingua ARCH-8 within-chain extra attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-8`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `concat`
- config_hash: `51daf190d0f6e40df33e964acf1f3a7312c5e633d1340b1b9195eb530b5e49b9`
- n_trainable: 567681
- VAL_P/S/mean/worst: 2.944350 / 3.109555 / 3.026952 / 3.109555
- TEST_P/S/mean/worst: 3.445308 / 3.331284 / 3.388296 / 3.445308

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010           9                   63          0.000996      2.806404           39           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/primary_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     1               0.0010 0.000010           7                   42          0.000998      3.347363           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/primary_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     2               0.0010 0.000010          52                  364          0.000849      2.977175           82           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/primary_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     3               0.0001 0.000001          18                  126          0.000098      2.596218           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/primary_k3_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     4               0.0010 0.000010          13                   91          0.000991      2.986307           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/primary_k4_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     0               0.0010 0.000010          11                   77          0.000994      2.572752           41           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     1               0.0010 0.000010          13                   91          0.000991      3.326847           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     2               0.0001 0.000001          18                  126          0.000098      3.320647           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     3               0.0010 0.000010           7                   49          0.000998      3.254010           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     4               0.0001 0.000001          20                  140          0.000098      3.085779           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T088_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 570e590061e5af2a   101

