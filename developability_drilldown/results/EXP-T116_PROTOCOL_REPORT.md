# EXP-T116 — AbLang2 ARCH-6 ungated residue cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `977aed25d906c93a0421a5809075c12f97ccfb105feec5afbb655ece498a974f`
- n_trainable: 465281
- VAL_P/S/mean/worst: 2.832277 / 2.940437 / 2.886357 / 2.940437

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          54                  378          0.000084      2.900362           84           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/primary_k0_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
primary     1               0.0010 0.000010          24                  144          0.000968      3.508642           54           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     2               0.0010 0.000010          17                  119          0.000984      2.780056           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     3               0.0001 0.000001          65                  455          0.000077      2.308577           95           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/primary_k3_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
primary     4               0.0010 0.000010          18                  126          0.000982      2.640482           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     0               0.0010 0.000010          12                   84          0.000993      2.444562           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     1               0.0001 0.000001          55                  385          0.000083      2.898248           85           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
 shadow     2               0.0010 0.000010          56                  392          0.000826      2.901959           86           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     3               0.0100 0.000100           8                   56          0.009970      3.537040           38           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 5ab9c53eb35c587c   101
 shadow     4               0.0010 0.000010          31                  217          0.000946      2.917228           61           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T116_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

