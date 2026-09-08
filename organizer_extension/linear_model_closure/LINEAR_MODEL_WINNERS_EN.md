# Linear Model Winners — Detailed Dossiers (English)
Abbreviations such as PARENT / BASE / ARO / CONT are expanded in full. Public/Private winners are **postmortem only** and were not used for model selection.
Metadata repairs (effective dimension, Public provenance, Private tie) applied without changing scores or rankings.

==================================================
A. TmApp — canonical CV winner
==================================================
1. **Full model name:** AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding with RIDGE
2. **Target:** TmApp
3. **Why winner:** lowest cv_worst_mae among FEATURE_LINEAR canonical Simple TVT only.
4. **Scores:** CV Primary=2.732072376654289; CV Shadow=2.784957206877823; CV mean=2.758514791766056; CV worst=2.784957206877823; Public=3.1185249613955217; Private=3.289917689937472
5. **Complete feature composition:** AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding
6. **Feature sources:** virtual_participant/round1_finalization/cache/round1_embeddings.npz::ablang2__HL_paired | built via stage1 make_xy(SEQ_BASIC) on competition sequences | organizer_extension/.../BIOEMU_ISOLATED_REASSESS_FEATURES.csv (ca_rmsd*) | organizer_extension/.../proteinmpnn/M1_FEATURES.csv | ablingua600m/embeddings/ablingua600m_HL_mean_concat.parquet | ablingua600m/embeddings_guided/ablingua600m_CDR3.parquet

Per-block dimensions (from actual feature matrices + canonical PCA pipeline):

| block | raw_dim | transform | effective_dim |
|---|---:|---|---:|
| AbLang2_HL_paired | 480 | raw (no PCA) | 480 |
| SEQ_BASIC | 78 | raw (no PCA) | 78 |
| BIOEMU_NEW_PAIRWISE | 10 | raw (no PCA) | 10 |
| M1_PROTEINMPNN | 1 | raw (no PCA) | 1 |
| AbLingua_HL_mean | 2560 | fold-local PCA | 32 |
| AbLingua_CDR3 | 2560 | fold-local PCA | 32 |
| __TOTAL__ | 5689 | concat after per-block transforms | 633 |

   Raw concatenated dimension: 5689.0; **Effective Ridge input dimension after PCA: 633.0** (not equal to raw; AbLingua GLOBAL and CDR3 each reduced with fold-local PCA32).
7. **Regressor:** RIDGE — sklearn Ridge
8. **Preprocessing:** recipe_raw + PCA32_per_abl_block + Ridge; dimensionality reduction: PCA32 per AbLingua block. Median impute on recipe blocks; fold-local PCA32 independently on each AbLingua block; StandardScaler on the concatenated train+val matrix before Ridge (canonical_simple_tvt_v1).
9. **Hyperparameters:** final_alpha=100.0; Lasso nonzero=nan / total=nan
10. **CV protocol:** canonical_simple_tvt_v1
11. **Final Test-training protocol:** FULL_DEV median Primary alpha
12. **Test prediction provenance:** CANONICAL_FULL_DEV_REFIT (canonical_replay_status=REPLAYED_CANONICAL)
13. **Public/Private scoring:** N_public=81, N_private=81 on official solution mask
14. **Interpretation:** Combines AbLang2 paired PLM embedding, sequence descriptors, BioEmu pairwise ensemble geometry, ProteinMPNN compatibility, and two AbLingua embeddings (global + CDR3-guided) under Ridge after PCA on AbLingua only.
15. **Caveat:** Selected using canonical CV only (cv_worst_mae).

**model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`

==================================================
B. TmApp — Public winner
==================================================
1. **Full model name:** Stage-2 AbLang2 heavy+light (HL concat of whole-chain means) under Optuna-tuned Ridge (RidgeOpt), alpha=78.35145884661983, fold-local StandardScaler + PCA48
2. **Target:** TmApp
3. **Why winner:** lowest public_mae among all linear/linear-blend rows in the master registry (postmortem). This is **not** the endgame AbLang2_HL_paired recipe.
4. **Scores:** CV Primary=2.9375619787158387; CV Shadow=3.030556011199951; CV mean=2.984058994957895; CV worst=3.030556011199951; Public=3.0853249349711853; Private=3.159989439410928
5. **Complete feature composition:** AbLang2 protein-language-model embeddings only — Heavy-chain whole-chain mean (480-d) concatenated with Light-chain whole-chain mean (480-d) = 960-d raw vector. No SEQ_BASIC, BioEmu, ProteinMPNN, or AbLingua blocks.
6. **Feature sources / provenance:**
   - Representation: `HL` = Heavy-chain whole-chain mean embedding concatenated with Light-chain whole-chain mean embedding (NOT AbLang2 HL_paired / paired seqcoding)
   - Pooling: `whole_chain_mean`
   - Embedding loader: `Stage-2 AbLang2 cache via run_stage2 load_ablang2 (H mean, L mean → HL concat); distinct from endgame AbLang2_HL_paired block`
   - Authoritative Stage-2 row: `virtual_participant/stage2_plm/stage2_primary_results.csv`
   - Frozen Test predictions: `/workspace_developability_acquisition/virtual_participant/round1_postmortem/predictions_exploratory/TmApp__ablang2__HL__RidgeOpt.csv`
   - Prediction SHA256: `8f2e45b69c09293aae52b4e9e78e4f1232df100db44bcd80efa9a354b847bf28`
7. **Regressor:** RidgeOpt — alpha=78.35145884661983 (Optuna, Optuna; notes field reports trials=40; pca_dim was a search hyperparameter)
8. **Preprocessing (model-specific):** StandardScaler; fitted inside each training fold (with SimpleImputer median). **PCA used: YES** — Optuna-selected PCA dimension = 48 (raw 960 → 48). This is **not** “no PCA”, and it is **not** an SVR base-model pipeline.
9. **Hyperparameters:** final_alpha=78.35145884661983; Lasso n/a
10. **CV protocol:** Stage-2 Primary folds + Shadow confirmation; NOT proven identical to canonical_simple_tvt_v1
11. **Final Test-training protocol:** 履歴資料から確定できない: Stage-2 script documents fold-local CV/Optuna and OOF; authoritative Test predictions are the frozen exploratory artifact above. Exact full-DEV refit code path that produced that Test file is not pinned in a single authoritative script comment beyond Round1 postmortem inventory usage.
12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)
13. **Public/Private scoring:** N_public=81, N_private=81
14. **Interpretation:** A Stage-2 PLM-only Ridge model on AbLang2 HL-concat embeddings with fold-local scaling and PCA48; historically strong on Public relative to endgame stacks.
15. **Caveat:** Public designation is **postmortem** and was **not** used for selection or advanced-model freeze.

**model_id:** `TmApp__ablang2__HL__RidgeOpt__HIST`

==================================================
C. TmApp — Private winner
==================================================
1. **Full model name:** prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)
2. **Target:** TmApp
3. **Why winner:** lowest private_mae among registry rows. **Private 8-way tie audit:** classification=`IDENTICAL_PREDICTION_FAMILY` — eight META_diversity labels share **bit-identical** Test prediction vectors (shared SHA256=`8b7f14e326c369ab54702b94b1242483b793b61a765c226c6639125c6c1f5f2d`). This is an IDENTICAL_PREDICTION_FAMILY, not eight distinct prediction vectors that merely share the same Private MAE. Representative kept by stable model_id ordering: `TmApp__META_diversity__convex_mae__HIST` (Public was not used to break the tie).
4. **Scores:** CV Primary=2.813262920374202; CV Shadow=nan; CV mean=nan; CV worst=nan; Public=3.345595945037222; Private=3.15007534338225
5. **Complete feature composition:** prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)
6. **Feature sources:** prediction-level blend over frozen Stage base OOF/Test predictions; see `TMAPP_PRIVATE_TIE_AUDIT.csv` for hashes.
7. **Regressor:** convex_mae — {} (label differs across the eight tied IDs; Test predictions are identical)
8. **Preprocessing:** historical Stage-5 nested meta over base-model predictions (bases include SVR / RidgeOpt / ElasticNetOpt — this winner is a **prediction blend**, not a feature-level Ridge/Lasso matrix model)
9. **Hyperparameters:** meta-learner label-specific; for this representative see Stage-5 inventory
10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)
11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission
12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)
13. **Public/Private scoring:** N_public=81, N_private=81
14. **Interpretation:** Prediction-level diversity meta over SEQ_BASIC SVR, AbLang2 HL_paired RidgeOpt (no PCA), ESMFold STRUCT_RASA ElasticNet, and ADV_INTERACTIONS SVR. Eight named variants collapsed to one prediction family on Test.
15. **Caveat:** Private designation is **postmortem** and was **not** used for selection.

**model_id:** `TmApp__META_diversity__convex_mae__HIST`

==================================================
D. HIC — canonical CV winner
==================================================
1. **Full model name:** ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv) with LASSO
2. **Target:** HIC
3. **Why winner:** lowest cv_worst_mae among FEATURE_LINEAR canonical Simple TVT only.
4. **Scores:** CV Primary=0.4872225781431885; CV Shadow=0.4833719610538652; CV mean=0.4852972695985268; CV worst=0.4872225781431885; Public=0.4702379457310565; Private=0.4641712708181718
5. **Complete feature composition:** ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
6. **Feature sources:** virtual_participant/round1_finalization/cache/round1_embeddings.npz::esm2__H | built via stage1 make_xy(SEQ_ALL) on competition sequences | structure_gap_closure/cache/aromatic_features_esmfold.csv | HYDRO-FIELD/features_esmfold.parquet | TITRATION-SHAPE/features_esmfold.parquet
   Raw dimension: 1448.0; Effective: 1448.0
7. **Regressor:** LASSO — sklearn Lasso (no PCA)
8. **Preprocessing:** impute+StandardScaler+Lasso(no_PCA); dimensionality reduction: none
9. **Hyperparameters:** final_alpha=0.1; Lasso nonzero=24.0 / total=1448.0
10. **CV protocol:** canonical_simple_tvt_v1
11. **Final Test-training protocol:** FULL_DEV median Primary alpha
12. **Test prediction provenance:** CANONICAL_FULL_DEV_REFIT (canonical_replay_status=REPLAYED_CANONICAL)
13. **Public/Private scoring:** N_public=81, N_private=81
14. **Interpretation:** result_class=FEATURE_LINEAR; any PLM=YES; ESM-2 Heavy=YES; sequence=YES; aromatic=YES; continuous surface=NO; hydro_field=YES; titration=YES.
15. **Caveat:** Selected using canonical CV only.

**model_id:** `HIC_HYDRO_TITRATION__LASSO`

==================================================
E. HIC — Public winner
==================================================
1. **Full model name:** Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)
2. **Target:** HIC
3. **Why winner:** lowest Public/Private MAE among registry rows (postmortem).
4. **Scores:** CV Primary=0.4257721922584362; CV Shadow=0.4328749669956691; CV mean=0.4293235796270526; CV worst=0.4328749669956691; Public=0.4148392100594437; Private=0.4318145768709808
5. **Complete feature composition:** Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)
6. **Feature sources:** virtual_participant stage OOF / Round1 inventory; base_model_ids when present
   Raw dimension: nan; Effective: nan
7. **Regressor:** equal_mean_blend — {}
8. **Preprocessing:** historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix); dimensionality reduction: varies by base model
9. **Hyperparameters:** final_alpha=nan; Lasso nonzero=nan / total=nan
10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)
11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission
12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)
13. **Public/Private scoring:** N_public=81, N_private=81
14. **Interpretation:** result_class=PREDICTION_LINEAR_BLEND; any PLM=YES; ESM-2 Heavy=YES; sequence=YES; aromatic=NO; continuous surface=YES; hydro_field=NO; titration=NO.
15. **Caveat:** Public/Private designation is postmortem and was not used for selection.

**model_id:** `HIC__SIMPLE_blend_seq_surf__HIST`

==================================================
F. HIC — Private winner
==================================================
1. **Full model name:** Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)
2. **Target:** HIC
3. **Why winner:** lowest Public/Private MAE among registry rows (postmortem).
4. **Scores:** CV Primary=0.4324716368936545; CV Shadow=nan; CV mean=nan; CV worst=nan; Public=0.4237480344059232; Private=0.4180226409189143
5. **Complete feature composition:** Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)
6. **Feature sources:** virtual_participant stage OOF / Round1 inventory; base_model_ids when present
   Raw dimension: nan; Effective: nan
7. **Regressor:** equal_mean_blend — {}
8. **Preprocessing:** historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix); dimensionality reduction: varies by base model
9. **Hyperparameters:** final_alpha=nan; Lasso nonzero=nan / total=nan
10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)
11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission
12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)
13. **Public/Private scoring:** N_public=81, N_private=81
14. **Interpretation:** result_class=PREDICTION_LINEAR_BLEND; any PLM=YES; ESM-2 Heavy=YES; sequence=NO; aromatic=NO; continuous surface=YES; hydro_field=NO; titration=NO.
15. **Caveat:** Public/Private designation is postmortem and was not used for selection.

**model_id:** `HIC__SIMPLE_blend_esm2_surf__HIST`

## Appendix — TmApp GLOBAL AbLingua Ridge effective dimension

| block | raw_dim | transform | effective_dim |
|---|---:|---|---:|
| AbLang2_HL_paired | 480 | raw (no PCA) | 480 |
| SEQ_BASIC | 78 | raw (no PCA) | 78 |
| BIOEMU_NEW_PAIRWISE | 10 | raw (no PCA) | 10 |
| M1_PROTEINMPNN | 1 | raw (no PCA) | 1 |
| AbLingua_HL_mean | 2560 | fold-local PCA | 32 |
| __TOTAL__ | 3129 | concat after per-block transforms | 601 |
