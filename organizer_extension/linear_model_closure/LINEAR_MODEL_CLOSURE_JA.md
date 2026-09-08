# Linear Model Closure Report
## Top answers
1. **What did previous Top-3 mean?** **A. Top-3 among the small restricted Ridge/Lasso FEATURE_RECIPE_FREEZE inventory** in `endgame_model_benchmark` — **not** all-time organizer linear models.
2. **Was HIC naming misleading?** **YES.** Aliases like `ARO+TITR` / `ARO+CONT` / `ARO+HYDRO+TITR` actually included **ESM-2 Heavy-chain embedding + SEQ_ALL sequence descriptors + AROMATIC-TOPO + …**. They were never aromatic-only.
3. **Did best HIC feature-level CV model contain a PLM?** **YES** (ESM-2 Heavy). Composition: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv). Separately — strongest HIC Public model PLM? YES; strongest HIC Private model PLM? YES (Public/Private winners here are prediction blends of SVR bases, not feature-level Ridge/Lasso).
4. **Historical linear models audited in registry?** 116 rows (42 FEATURE_LINEAR; 50 PREDICTION_LINEAR_BLEND; 24 OTHER)
5. **Were historical ~0.42 HIC results recovered?** **YES** — e.g. Round1 PRIMARY / `HIC__SIMPLE_blend_seq_surf_adv` Public≈0.420 / Private≈0.424 (equal-mean **SVR** prediction blend). Recovered rows with Private<0.43: 2. These are **not** feature-level Ridge/Lasso and are **not** ranked in main FEATURE_LEVEL CV Top-10.
6. **TmApp canonical CV winner:** `TM_PARENT_ABLINGUA_CDR3__RIDGE` — AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding with RIDGE
7. **TmApp Public winner:** `TmApp__ablang2__HL__RidgeOpt__HIST`
8. **TmApp Private winner:** `TmApp__META_diversity__convex_mae__HIST`
9. **HIC canonical CV winner:** `HIC_HYDRO_TITRATION__LASSO` — ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv) with LASSO
10. **HIC Public winner:** `HIC__SIMPLE_blend_seq_surf__HIST`
11. **HIC Private winner:** `HIC__SIMPLE_blend_esm2_surf__HIST`
12. **Rank shake:** median |private_rank−cv_rank| among shake table ≈ 25.0; upper board shakes between CV and Private (especially HIC aromatic-only Ridge private strength vs CV).
13. **Ridge vs Lasso:** On matched feature recipes, Ridge wins CV-worst more often (12 vs 7). Lasso helps high-dimensional HIC concatenations; TmApp authority recipes remain Ridge (+ selective PCA on AbLingua).
14. **Advanced-model Top-3 freeze:** see `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`
15. **Participant bundle consistent?** needed blocks=['AROMATIC_TOPO', 'AbLang2_HL_paired', 'AbLingua_CDR3', 'AbLingua_HL_mean', 'BIOEMU_NEW_PAIRWISE', 'CONTINUOUS_SURFACE', 'ESM2_H', 'HYDRO_FIELD', 'M1_PROTEINMPNN', 'SEQ_ALL', 'SEQ_BASIC', 'TITRATION_SHAPE']; missing=none
16. **LINEAR_MODEL_CLOSED = YES**

## Integrity
- Canonical Top-3 audit written with full block expansion
- Main CV Top-10 = FEATURE_LINEAR + canonical Simple TVT only
- Historical blends retained separately / overall Public–Private tables
- Public/Private not used for advanced freeze
