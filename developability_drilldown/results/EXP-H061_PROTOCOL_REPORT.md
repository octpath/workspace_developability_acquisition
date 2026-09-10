# EXP-H061 — ESM-2 ARCH-4 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-4`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `de7048a46e6a1cd72a5d88d5daee910aab143d18d79c2a0ac73e7f290b1f7e64`
- n_trainable: 485249
- VAL_P/S/mean/worst: 0.456796 / 0.438236 / 0.447516 / 0.456796

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          61                  427          0.000080      0.498081           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/primary_k0_lr1e-04_seed101.pt 649131418599f207   101
primary     1               0.0010 0.000010           4                   24          0.000999      0.531265           34           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/primary_k1_lr1e-03_seed101.pt 649131418599f207   101
primary     2               0.0010 0.000010          25                  175          0.000965      0.342129           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/primary_k2_lr1e-03_seed101.pt 649131418599f207   101
primary     3               0.0010 0.000010          27                  189          0.000959      0.436207           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/primary_k3_lr1e-03_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010          10                   70          0.000995      0.472683           40           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0010 0.000010          70                  490          0.000737      0.400012          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 649131418599f207   101
 shadow     1               0.0010 0.000010          22                  154          0.000973      0.477475           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001          68                  476          0.000075      0.460979           98           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0010 0.000010          28                  196          0.000956      0.428053           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 649131418599f207   101
 shadow     4               0.0001 0.000001          69                  483          0.000074      0.426173           99           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H061_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 649131418599f207   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

