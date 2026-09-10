# EXP-H062 — ESM-2 ARCH-6 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `d6c1b9d284c8700efe7a6415bbf10ff8f2e21aed0175a556173b6105e4812f92`
- n_trainable: 567681
- VAL_P/S/mean/worst: 0.445760 / 0.445605 / 0.445683 / 0.445760

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          55                  385          0.000083      0.516303           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/primary_k0_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     1               0.0010 0.000010          14                   84          0.000990      0.521361           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/primary_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     2               0.0010 0.000010          35                  245          0.000931      0.300716           65           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/primary_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     3               0.0010 0.000010          25                  175          0.000965      0.467282           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/primary_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     4               0.0010 0.000010          50                  350          0.000860      0.418573           80           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/primary_k4_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     0               0.0010 0.000010          43                  301          0.000896      0.406915           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     1               0.0010 0.000010          26                  182          0.000962      0.473411           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     2               0.0001 0.000001          46                  322          0.000088      0.466770           76           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     3               0.0010 0.000010          47                  329          0.000876      0.426511           77           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     4               0.0010 0.000010          18                  126          0.000982      0.456222           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H062_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 570e590061e5af2a   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

