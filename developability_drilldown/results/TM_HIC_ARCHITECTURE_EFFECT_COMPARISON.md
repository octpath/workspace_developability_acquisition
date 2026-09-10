# TmApp vs HIC architecture-effect comparison

Compare **within-target deltas**, not raw MAE scales (Tm °C vs HIC minutes).

See `TM_HIC_ARCHITECTURE_EFFECTS.csv` and `TM_HIC_PAIRED_BOOTSTRAP.csv` for numeric contrasts.

## Questions

1. Separate vs joint — similar across targets?
2. MEAN vs CONCAT preference transfer?
3. Does Scratch benefit more from inductive bias on both?
4. Does residue cross-attention help both?
5. Does geometry help both?
6. Does HIC prefer Heavy-only more strongly?
7. Target-specific architecture preferences?

## Effects snapshot

| Contrast | Target | A | B | Δ TEST_mean |
|---|---|---|---|---:|
| geom_vs_arch6_ablingua_concat | TmApp | EXP-T105 | EXP-T084 | 0.0328 |
| geom_vs_arch6_ablingua_mean | TmApp | EXP-T106 | EXP-T085 | -0.1577 |
| geom_vs_arch6_scratch_concat | TmApp | EXP-T107 | EXP-T099 | -0.0049 |
| geom_vs_arch6_scratch_mean | TmApp | EXP-T108 | EXP-T100 | -0.1383 |
| geom_vs_arch6_ablang2_concat | TmApp | EXP-T118 | EXP-T116 | -0.0365 |
| geom_vs_arch6_ablang2_mean | TmApp | EXP-T119 | EXP-T117 | 0.0346 |
| hic_esm2_geom_vs_arch6_concat | HIC | EXP-H064 | EXP-H062 | 0.0051 |
| hic_esm2_geom_vs_arch6_mean | HIC | EXP-H065 | EXP-H063 | -0.0043 |
| hic_scratch_geom_vs_arch6_concat | HIC | EXP-H078 | EXP-H076 | -0.0461 |
| hic_scratch_geom_vs_arch6_mean | HIC | EXP-H079 | EXP-H077 | 0.0156 |
| hic_esm2_vs_scratch_ARCH-H0_h_only | HIC | EXP-H054 | EXP-H068 | -0.0285 |
| hic_esm2_vs_scratch_ARCH-1_concat | HIC | EXP-H055 | EXP-H069 | -0.0091 |
| hic_esm2_vs_scratch_ARCH-1_mean | HIC | EXP-H056 | EXP-H070 | -0.0138 |
| hic_esm2_vs_scratch_ARCH-2_None | HIC | EXP-H057 | EXP-H071 | 0.0163 |
| hic_esm2_vs_scratch_ARCH-3_concat | HIC | EXP-H058 | EXP-H072 | 0.0161 |
| hic_esm2_vs_scratch_ARCH-3_mean | HIC | EXP-H059 | EXP-H073 | 0.0008 |
| hic_esm2_vs_scratch_ARCH-4_concat | HIC | EXP-H060 | EXP-H074 | 0.0153 |
| hic_esm2_vs_scratch_ARCH-4_mean | HIC | EXP-H061 | EXP-H075 | -0.0028 |
| hic_esm2_vs_scratch_ARCH-6_concat | HIC | EXP-H062 | EXP-H076 | -0.0119 |
| hic_esm2_vs_scratch_ARCH-6_mean | HIC | EXP-H063 | EXP-H077 | 0.0041 |
| hic_esm2_vs_scratch_ARCH-6G_concat | HIC | EXP-H064 | EXP-H078 | 0.0392 |
| hic_esm2_vs_scratch_ARCH-6G_mean | HIC | EXP-H065 | EXP-H079 | -0.0158 |
| hic_esm2_vs_scratch_ARCH-8_concat | HIC | EXP-H066 | EXP-H080 | 0.0056 |
| hic_esm2_vs_scratch_ARCH-8_mean | HIC | EXP-H067 | EXP-H081 | -0.0225 |
