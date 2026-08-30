# 01 — Split Objective and Method

Gate B7.3 constructs Public/Private masks using an explicit **model-blind**
lexicographic distribution-matching objective, then validates frozen finalists
with separated model banks.

## Hard constraints (L0)

- Public = Private = 81
- Atomic `sequence_group` integrity
- HIC HIGH public count ∈ {3,4} (Test HIGH total = 7)
- Confirmed band totals: LOW=149, MEDIUM=6, HIGH=7
- Prefer MEDIUM exactly 3/3

## Lexicographic objective

1. **L1 (minimax counts)**: max(TmApp_q8_L1, HIC_q8_L1, HIC_MED_imbalance, joint_grid_L1)
2. **L2 (continuous)**: max(TmApp_W_norm, HIC_W_norm, joint_energy_norm)
3. **L3 (biology/sequence)**: max(mean|SMD|, germline_TV_mean, group_mass_TV)
4. **L4**: aggregate tie-break

## Formulas

- Quantile L1: `Σ_b |n_pub(b) − n_priv(b)|` on equal-frequency bins of full Test
- Wasserstein norm: `W1 / Test_SD`
- Joint energy norm: energy distance on z-scored (TmApp,HIC) / scale=0.031815
  (scale = median energy of 200 random feasible splits)
- |SMD| on predeclared continuous list (incl. paired nearest-train identity)
- TV = half L1 of category probability vectors

Protocol hash: `7742fe3a84021fac32f8c73dc263bba963787c942846beea0492e8fbed874853`

See `config/B7_3_PROTOCOL.json` for the frozen specification.
