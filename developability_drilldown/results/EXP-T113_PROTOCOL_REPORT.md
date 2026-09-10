# EXP-T113 — AbLang2 ARCH-3 joint unrestricted dual MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-3`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `21dc225a94af89b8253dcf9375695463e9e53d86ae1419cd7b709f6e3e26a4b8`
- n_trainable: 382849
- VAL_P/S/mean/worst: 2.748090 / 2.965465 / 2.856778 / 2.965465

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          37                  259          0.000923      2.779233           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/primary_k0_lr1e-03_seed101.pt 13358cb1e3782e93   101
primary     1               0.0010 0.000010          36                  216          0.000927      3.324992           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/primary_k1_lr1e-03_seed101.pt 13358cb1e3782e93   101
primary     2               0.0001 0.000001         132                  924          0.000027      2.769919          162           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/primary_k2_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     3               0.0001 0.000001          68                  476          0.000075      2.211286           98           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/primary_k3_lr1e-04_seed101.pt 13358cb1e3782e93   101
primary     4               0.0010 0.000010          26                  182          0.000962      2.636018           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/primary_k4_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     0               0.0010 0.000010          28                  196          0.000956      2.620316           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     1               0.0010 0.000010          32                  224          0.000942      2.882578           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     2               0.0010 0.000010          33                  231          0.000939      2.797659           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 13358cb1e3782e93   101
 shadow     3               0.0100 0.000100          24                  168          0.009680      3.505277           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 13358cb1e3782e93   101
 shadow     4               0.0001 0.000001         102                  714          0.000050      3.015412          132           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T113_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 13358cb1e3782e93   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

