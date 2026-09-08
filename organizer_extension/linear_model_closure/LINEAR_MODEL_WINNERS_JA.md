# Linear Model Winners — Detailed Dossiers

Abbreviations such as PARENT / BASE / ARO / CONT are expanded in full. Public/Private winners are **postmortem only** and were not used for model selection.


==================================================
A. TmApp — canonical CV winner
==================================================

1. **Full model name:** AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding with RIDGE

2. **Target:** TmApp

3. **Why winner:** lowest cv_worst_mae among the ranking pool for this section (FEATURE_LINEAR canonical Simple TVT only).

4. **Scores:** CV Primary=2.732072376654289; CV Shadow=2.784957206877823; CV mean=2.758514791766056; CV worst=2.784957206877823; Public=3.1185249613955217; Private=3.289917689937472

5. **Complete feature composition:** AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding

6. **Feature sources:** virtual_participant/round1_finalization/cache/round1_embeddings.npz::ablang2__HL_paired | built via stage1 make_xy(SEQ_BASIC) on competition sequences | organizer_extension/.../BIOEMU_ISOLATED_REASSESS_FEATURES.csv (ca_rmsd*) | organizer_extension/.../proteinmpnn/M1_FEATURES.csv | ablingua600m/embeddings/ablingua600m_HL_mean_concat.parquet | ablingua600m/embeddings_guided/ablingua600m_CDR3.parquet

   Raw dimension: 5689.0; Effective: 5689.0

7. **Regressor:** RIDGE — sklearn Ridge

8. **Preprocessing:** recipe_raw + PCA32_per_abl_block + Ridge; dimensionality reduction: PCA32 per AbLingua block

9. **Hyperparameters:** final_alpha=100.0; Lasso nonzero=nan / total=nan

10. **CV protocol:** canonical_simple_tvt_v1

11. **Final Test-training protocol:** FULL_DEV median Primary alpha

12. **Test prediction provenance:** CANONICAL_FULL_DEV_REFIT (canonical_replay_status=REPLAYED_CANONICAL)

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=FEATURE_LINEAR; ESM-2 Heavy PLM=NO; AbLang2 PLM=YES; any PLM=YES; sequence descriptors=YES; aromatic structural=NO; continuous surface=NO; hydrophobic-field=NO; titration=NO. Information combined: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding

15. **Caveat:** Selected using canonical CV only (cv_worst_mae).


**model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`


==================================================
B. TmApp — Public winner
==================================================

1. **Full model name:** Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only (no SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua blocks) under Optuna-tuned Ridge (alpha≈78.35145884661983)

2. **Target:** TmApp

3. **Why winner:** lowest public_mae among the ranking pool for this section (all linear/linear-blend rows with this metric).

4. **Scores:** CV Primary=2.9375619787158387; CV Shadow=3.030556011199951; CV mean=2.984058994957895; CV worst=3.030556011199951; Public=3.0853249349711853; Private=3.159989439410928

5. **Complete feature composition:** Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only (no SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua blocks) under Optuna-tuned Ridge (alpha≈78.35145884661983)

6. **Feature sources:** virtual_participant stage OOF / Round1 inventory; base_model_ids when present

   Raw dimension: nan; Effective: nan

7. **Regressor:** RidgeOpt — {"alpha": 78.35145884661983}

8. **Preprocessing:** historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix); dimensionality reduction: varies by base model

9. **Hyperparameters:** final_alpha=nan; Lasso nonzero=nan / total=nan

10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)

11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission

12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=HISTORICAL_LINEAR_OTHER; ESM-2 Heavy PLM=NO; AbLang2 PLM=YES; any PLM=YES; sequence descriptors=NO; aromatic structural=NO; continuous surface=NO; hydrophobic-field=NO; titration=NO. Information combined: Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only (no SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua blocks) under Optuna-tuned Ridge (alpha≈78.35145884661983)

15. **Caveat:** This Public/Private designation is **postmortem**. It was **not** used to choose recipes, regressors, or advanced-model freeze. If this winner differs from the canonical CV winner, that difference must not drive selection.


**model_id:** `TmApp__ablang2__HL__RidgeOpt__HIST`


==================================================
C. TmApp — Private winner
==================================================

1. **Full model name:** prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)

2. **Target:** TmApp

3. **Why winner:** lowest private_mae among the ranking pool for this section (all linear/linear-blend rows with this metric). Exact-score tie among 8 rows at private_mae=3.15007534338225; representative winner listed first by stable sort: `TmApp__META_diversity__convex_mae__HIST`, `TmApp__META_diversity__nnls__HIST`, `TmApp__META_diversity__nnls_norm__HIST`, `TmApp__META_diversity__quantile_0.0__HIST`, `TmApp__META_diversity__quantile_0.001__HIST`, `TmApp__META_diversity__quantile_0.01__HIST`, `TmApp__META_diversity__quantile_0.1__HIST`, `TmApp__META_diversity__ridge_100.0__HIST`.

4. **Scores:** CV Primary=2.813262920374202; CV Shadow=nan; CV mean=nan; CV worst=nan; Public=3.345595945037222; Private=3.15007534338225

5. **Complete feature composition:** prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)

6. **Feature sources:** virtual_participant stage OOF / Round1 inventory; base_model_ids when present

   Raw dimension: nan; Effective: nan

7. **Regressor:** convex_mae — {}

8. **Preprocessing:** historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix); dimensionality reduction: varies by base model

9. **Hyperparameters:** final_alpha=nan; Lasso nonzero=nan / total=nan

10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)

11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission

12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=PREDICTION_LINEAR_BLEND; ESM-2 Heavy PLM=NO; AbLang2 PLM=YES; any PLM=YES; sequence descriptors=YES; aromatic structural=NO; continuous surface=NO; hydrophobic-field=NO; titration=NO. Information combined: prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)

15. **Caveat:** This Public/Private designation is **postmortem**. It was **not** used to choose recipes, regressors, or advanced-model freeze. If this winner differs from the canonical CV winner, that difference must not drive selection.


**model_id:** `TmApp__META_diversity__convex_mae__HIST`


==================================================
D. HIC — canonical CV winner
==================================================

1. **Full model name:** ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv) with LASSO

2. **Target:** HIC

3. **Why winner:** lowest cv_worst_mae among the ranking pool for this section (FEATURE_LINEAR canonical Simple TVT only).

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

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=FEATURE_LINEAR; ESM-2 Heavy PLM=YES; AbLang2 PLM=NO; any PLM=YES; sequence descriptors=YES; aromatic structural=YES; continuous surface=NO; hydrophobic-field=YES; titration=YES. Information combined: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)

15. **Caveat:** Selected using canonical CV only (cv_worst_mae).


**model_id:** `HIC_HYDRO_TITRATION__LASSO`


==================================================
E. HIC — Public winner
==================================================

1. **Full model name:** Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

2. **Target:** HIC

3. **Why winner:** lowest public_mae among the ranking pool for this section (all linear/linear-blend rows with this metric).

4. **Scores:** CV Primary=0.4257721922584362; CV Shadow=0.4328749669956691; CV mean=0.42932357962705264; CV worst=0.4328749669956691; Public=0.4148392100594437; Private=0.4318145768709808

5. **Complete feature composition:** Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

6. **Feature sources:** virtual_participant stage OOF / Round1 inventory; base_model_ids when present

   Raw dimension: nan; Effective: nan

7. **Regressor:** equal_mean_blend — {}

8. **Preprocessing:** historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix); dimensionality reduction: varies by base model

9. **Hyperparameters:** final_alpha=nan; Lasso nonzero=nan / total=nan

10. **CV protocol:** historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)

11. **Final Test-training protocol:** historical Round1 full-DEV refit / frozen submission

12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION (canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE)

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=PREDICTION_LINEAR_BLEND; ESM-2 Heavy PLM=YES; AbLang2 PLM=NO; any PLM=YES; sequence descriptors=YES; aromatic structural=NO; continuous surface=YES; hydrophobic-field=NO; titration=NO. Information combined: Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

15. **Caveat:** This Public/Private designation is **postmortem**. It was **not** used to choose recipes, regressors, or advanced-model freeze. If this winner differs from the canonical CV winner, that difference must not drive selection.


**model_id:** `HIC__SIMPLE_blend_seq_surf__HIST`


==================================================
F. HIC — Private winner
==================================================

1. **Full model name:** Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

2. **Target:** HIC

3. **Why winner:** lowest private_mae among the ranking pool for this section (all linear/linear-blend rows with this metric).

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

13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask

14. **Interpretation:** result_class=PREDICTION_LINEAR_BLEND; ESM-2 Heavy PLM=YES; AbLang2 PLM=NO; any PLM=YES; sequence descriptors=NO; aromatic structural=NO; continuous surface=YES; hydrophobic-field=NO; titration=NO. Information combined: Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

15. **Caveat:** This Public/Private designation is **postmortem**. It was **not** used to choose recipes, regressors, or advanced-model freeze. If this winner differs from the canonical CV winner, that difference must not drive selection.


**model_id:** `HIC__SIMPLE_blend_esm2_surf__HIST`
