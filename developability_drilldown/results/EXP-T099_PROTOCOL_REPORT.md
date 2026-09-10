# EXP-T099 — Scratch ARCH-6 ungated cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-6`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `301150113540406cfea1a70e6e678114436eca1f79bc181311f276588424c2a0`
- n_trainable: 406529
- VAL_P/S/mean/worst: 3.032267 / 3.055491 / 3.043879 / 3.055491
- TEST_P/S/mean/worst: 3.422744 / 3.222762 / 3.322753 / 3.422744

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          45                  315          0.000089      2.984967           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/primary_k0_lr1e-04_seed101.pt 8462d848103517b5   101
primary     1               0.0010 0.000010          23                  138          0.000971      3.333258           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8462d848103517b5   101
primary     2               0.0001 0.000001          47                  329          0.000088      3.335171           77           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8462d848103517b5   101
primary     3               0.0010 0.000010          16                  112          0.000986      2.729483           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8462d848103517b5   101
primary     4               0.0010 0.000010          36                  252          0.000927      2.770526           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     0               0.0001 0.000001          77                  539          0.000069      2.521837          107           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8462d848103517b5   101
 shadow     1               0.0010 0.000010          13                   91          0.000991      3.328856           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     2               0.0010 0.000010          12                   84          0.000993      2.959081           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     3               0.0010 0.000010          23                  161          0.000971      3.299307           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     4               0.0010 0.000010          14                   98          0.000990      3.177430           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T099_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8462d848103517b5   101

