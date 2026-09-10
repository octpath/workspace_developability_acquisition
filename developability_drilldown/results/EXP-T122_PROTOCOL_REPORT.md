# EXP-T122 — AbLang2 ARCH-8 within-chain extra-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-8`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `ee804ec91483f3af4e7f207c7706e5f4f6fb9db6aa7008f6e5fa59c8e707d4a8`
- n_trainable: 465281
- VAL_P/S/mean/worst: 2.817561 / 2.827331 / 2.822446 / 2.827331

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          23                  161          0.000971      2.700953           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/primary_k0_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     1               0.0010 0.000010          36                  216          0.000927      3.447919           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     2               0.0010 0.000010          18                  126          0.000982      3.164871           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     3               0.0001 0.000001          71                  497          0.000073      2.259075          101           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/primary_k3_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
primary     4               0.0010 0.000010          42                  294          0.000901      2.498930           72           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     0               0.0010 0.000010          28                  196          0.000956      2.433974           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     1               0.0010 0.000010          26                  182          0.000962      2.729424           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     2               0.0010 0.000010          68                  476          0.000750      2.756830           98           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     3               0.0100 0.000100           9                   63          0.009961      3.498364           39           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 5ab9c53eb35c587c   101
 shadow     4               0.0010 0.000010          20                  140          0.000978      2.709387           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T122_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

