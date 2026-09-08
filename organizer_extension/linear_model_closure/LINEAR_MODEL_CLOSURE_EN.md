# Linear Model Closure Report (English)

## Top answers

1. **What did previous Top-3 mean?** **A. Top-3 among the small restricted Ridge/Lasso FEATURE_RECIPE_FREEZE inventory** in `endgame_model_benchmark` — **not** all-time organizer linear models.
2. **Was HIC naming misleading?** **YES.** Aliases like ARO+TITR / ARO+CONT actually included ESM-2 Heavy + SEQ_ALL + AROMATIC-TOPO + …
3. **Did best HIC feature-level CV model contain a PLM?** **YES** (ESM-2 Heavy). Public strongest: YES; Private strongest: YES (Public/Private winners are prediction blends).
4. **Historical linear models audited?** 116 registry rows (42 FEATURE_LINEAR; 50 PREDICTION_LINEAR_BLEND; 24 OTHER)
5. **Historical HIC ≈0.42 recovered?** **YES** — e.g. Round1 PRIMARY blend Public≈0.420 / Private≈0.424. Rows with Private<0.43: 2. Not ranked in FEATURE_LEVEL CV Top-10.
6. **TmApp canonical CV winner:** `TM_PARENT_ABLINGUA_CDR3__RIDGE` (raw=5689.0, effective_after_PCA=633.0)
7. **TmApp Public winner:** `TmApp__ablang2__HL__RidgeOpt__HIST` — AbLang2 HL concat, StandardScaler + PCA48, RidgeOpt alpha=78.35145884661983
8. **TmApp Private winner:** `TmApp__META_diversity__convex_mae__HIST` — IDENTICAL_PREDICTION_FAMILY of 8 META_diversity labels (SHA256=`8b7f14e326c369ab…`)
9. **HIC canonical CV winner:** `HIC_HYDRO_TITRATION__LASSO`
10. **HIC Public winner:** `HIC__SIMPLE_blend_seq_surf__HIST`
11. **HIC Private winner:** `HIC__SIMPLE_blend_esm2_surf__HIST`
12. **Rank shake:** Upper leaderboard shakes between CV and Private/Public; see `RANK_SHAKE_ANALYSIS.csv`. MAE gaps among near-ties can be tiny.
13. **Ridge vs Lasso:** Ridge wins CV-worst more often on matched recipes; Lasso helps high-dim HIC; TmApp authority remains Ridge + selective AbLingua PCA.
14. **Advanced-model freeze:** see `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json` (TmApp=['TM_PARENT_ABLINGUA_CDR3__RIDGE', 'TM_PARENT_ABLINGUA_GLOBAL__RIDGE', 'TM_BASE_BIOEMU_MPNN__RIDGE']; HIC=['HIC_HYDRO_TITRATION__LASSO', 'HIC_ARO_CONTINUOUS_SURFACE__LASSO', 'HIC_ESM2_SEQ_AROMATIC__LASSO'])
15. **Participant bundle consistent?** YES
16. **LINEAR_MODEL_CLOSED = YES**

## Metadata repairs (no score changes)
- Effective dimension for AbLingua Ridge recipes corrected (PCA32 per AbLingua block).
- TmApp Public winner provenance expanded from Stage-2 artifacts (PCA48, not SVR boilerplate).
- TmApp Private 8-way tie classified as IDENTICAL_PREDICTION_FAMILY.

English mirror: this file. Japanese: `LINEAR_MODEL_CLOSURE_JA.md`.
