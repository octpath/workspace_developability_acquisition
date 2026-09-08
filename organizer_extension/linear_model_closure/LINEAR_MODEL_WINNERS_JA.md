# 線形モデル勝者 — 詳細ドシエ（日本語）

略語（PARENT / BASE / ARO / CONT など）は展開して記載する。Public / Private の「最良」は事後解析（postmortem）のみであり、モデル選定には用いていない。
有効次元・Public 由来・Private 同点の監査を反映済み。スコア・順位は変更していない。

英語版: `LINEAR_MODEL_WINNERS_EN.md`

==================================================
A. TmApp — canonical CV 最良モデル
==================================================

1. **モデルの完全な構成:** AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding with RIDGE

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** FEATURE_LINEAR かつ canonical Simple TVT のみの母集団で、`cv_worst_mae`（Primary と Shadow の悪い方）が最小。

4. **スコア:** CV Primary=2.732072376654289; CV Shadow=2.784957206877823; CV mean=2.758514791766056; CV worst=2.784957206877823; Public=3.1185249613955217; Private=3.289917689937472

5. **使用特徴量の完全な構成:** AbLang2 の重鎖・軽鎖ペア埋め込み（paired heavy+light）＋ stage-1 の SEQ_BASIC 配列記述子（長さ・組成・CDR 長要約）＋ BioEmu の孤立 VH/VL NEW_PAIRWISE アンサンブル Cα RMSD 記述子＋ ProteinMPNN の配列–構造適合ネイティブスコア記述子＋ AbLingua-600M の masked-mean 大域（GLOBAL）重鎖+軽鎖連結埋め込み＋ AbLingua-600M の CDR3 誘導残基プーリング埋め込み

6. **各特徴量の由来:** virtual_participant/round1_finalization/cache/round1_embeddings.npz::ablang2__HL_paired | built via stage1 make_xy(SEQ_BASIC) on competition sequences | organizer_extension/.../BIOEMU_ISOLATED_REASSESS_FEATURES.csv (ca_rmsd*) | organizer_extension/.../proteinmpnn/M1_FEATURES.csv | ablingua600m/embeddings/ablingua600m_HL_mean_concat.parquet | ablingua600m/embeddings_guided/ablingua600m_CDR3.parquet

実行列から測ったブロック別次元（canonical 評価器の PCA パイプラインに基づく）:

| 特徴ブロック | 生次元 | 変換 | 有効次元 |
|---|---:|---|---:|
| AbLang2_HL_paired | 480 | raw (no PCA) | 480 |
| SEQ_BASIC | 78 | raw (no PCA) | 78 |
| BIOEMU_NEW_PAIRWISE | 10 | raw (no PCA) | 10 |
| M1_PROTEINMPNN | 1 | raw (no PCA) | 1 |
| AbLingua_HL_mean | 2560 | fold-local PCA | 32 |
| AbLingua_CDR3 | 2560 | fold-local PCA | 32 |
| __TOTAL__ | 5689 | concat after per-block transforms | 633 |

生の連結次元: 5689.0。**PCA 後の Ridge 入力有効次元: 633.0**（生次元と同じではない。AbLingua GLOBAL と CDR3 にそれぞれ fold 内 PCA32）。

7. **回帰モデル:** Ridge（sklearn Ridge）

8. **前処理:** レシピ側は中央値補完のうえ生特徴を連結。AbLingua ブロックのみそれぞれ fold 内 PCA32。連結後の train+val 行列に StandardScaler をかけて Ridge（`canonical_simple_tvt_v1` / `run_recipe_plus_abl_blocks`）。

9. **ハイパーパラメータ:** final_alpha=100.0（full DEV では Primary 5-fold の中央値アルファ方針）

10. **CVプロトコル:** Primary 折りと Shadow 折りの Simple TVT 回転（`canonical_simple_tvt_v1`）

11. **Test予測時の最終学習方法:** DEV 全体での再学習。アルファは Primary fold の中央値（Test ラベル非使用）

12. **Test予測値の由来:** CANONICAL_FULL_DEV_REFIT（canonical_replay_status=REPLAYED_CANONICAL）

13. **Public / Private 評価:** N_public=81, N_private=81（公式 solution マスク）

14. **モデルの解釈:** 配列 PLM（AbLang2・AbLingua）、古典配列記述子、アンサンブル幾何（BioEmu）、逆折りたたみ適合（ProteinMPNN）を線形に統合。高次元の AbLingua のみ PCA で圧縮。

15. **注意点:** 選定は canonical CV のみ。Public/Private は事後評価。

**model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`


==================================================
B. TmApp — Public 最良モデル
==================================================

1. **モデルの完全な構成:** Stage-2 の AbLang2 重鎖+軽鎖（HL：各鎖 whole-chain mean の連結）を、Optuna 調整 Ridge（RidgeOpt, alpha=78.35145884661983）で回帰。fold 内 StandardScaler ＋ PCA48。

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** マスタ登録上の線形／線形ブレンド行の中で public_mae が最小（事後解析）。endgame の AbLang2_HL_paired レシピとは別物。

4. **スコア:** CV Primary=2.9375619787158387; CV Shadow=3.030556011199951; CV mean=2.984058994957895; CV worst=3.030556011199951; Public=3.0853249349711853; Private=3.159989439410928

5. **使用特徴量の完全な構成:** AbLang2 のタンパク質言語モデル（PLM）埋め込みのみ。重鎖 whole-chain mean（480次元）と軽鎖 whole-chain mean（480次元）を連結した 960 次元。SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua は含まない。**AbLang2 HL_paired（ペア配列コーディング）ではない。**

6. **各特徴量の由来:**
   - 表現: `HL` — Heavy-chain whole-chain mean embedding concatenated with Light-chain whole-chain mean embedding (NOT AbLang2 HL_paired / paired seqcoding)
   - プーリング: `whole_chain_mean`
   - 埋め込み取得: Stage-2 `run_stage2.py` の AbLang2 キャッシュ経路
   - 権威テーブル: `virtual_participant/stage2_plm/stage2_primary_results.csv`
   - 凍結 Test 予測: `/workspace_developability_acquisition/virtual_participant/round1_postmortem/predictions_exploratory/TmApp__ablang2__HL__RidgeOpt.csv`
   - 予測 SHA256: `8f2e45b69c09293aae52b4e9e78e4f1232df100db44bcd80efa9a354b847bf28`

7. **回帰モデル:** RidgeOpt（Optuna 調整 Ridge）。alpha=78.35145884661983。探索試行数は notes 上 40。

8. **前処理（本モデル固有）:** 各学習 fold 内で SimpleImputer（中央値）→ StandardScaler → PCA。**PCA 使用: あり**（Optuna が選んだ次元 48。生 960 → 48）。SVR 基底モデルのパイプラインではない。「PCA なし」でもない。

9. **ハイパーパラメータ:** alpha=78.35145884661983。pca_dim も探索対象。

10. **CVプロトコル:** Stage-2 Primary 折り＋ Shadow 確認。`canonical_simple_tvt_v1` との同一性は証明されていない。

11. **Test予測時の最終学習方法:** 履歴資料から確定できない: Stage-2 script documents fold-local CV/Optuna and OOF; authoritative Test predictions are the frozen exploratory artifact above. Exact full-DEV refit code path that produced that Test file is not pinned in a single authoritative script comment beyond Round1 postmortem inventory usage.

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION（canonical_replay_status=HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE）

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM 単独の線形モデルが、後段の複雑なスタックより Public で強かった事例。選定根拠には使わない。

15. **注意点:** Public 最良は事後ラベル。advanced-model 凍結やレシピ選定には未使用。

**model_id:** `TmApp__ablang2__HL__RidgeOpt__HIST`


==================================================
C. TmApp — Private 最良モデル
==================================================

1. **モデルの完全な構成:** prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt); AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA; ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt; Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** private_mae 最小。ただし **8 ラベル同点監査**の結果、分類は `IDENTICAL_PREDICTION_FAMILY`。8 個の META_diversity ラベルは **ビット単位で同一の Test 予測ベクトル**を共有する（共有 SHA256=`8b7f14e326c369ab54702b94b1242483b793b61a765c226c6639125c6c1f5f2d`）。Private MAE が偶然一致した別予測ではなく、**同一予測ファミリーの重複ラベル**である。代表は安定ソートで `TmApp__META_diversity__convex_mae__HIST`（Public ではタイブレークしない）。

4. **スコア:** CV Primary=2.813262920374202; CV Shadow=nan; CV mean=nan; CV worst=nan; Public=3.345595945037222; Private=3.15007534338225

5. **使用特徴量の完全な構成:** 特徴行列ではなく、基底モデル予測のメタ統合: SEQ_BASIC 上の SVROpt、AbLang2 HL_paired 上の RidgeOpt（PCA なし）、ESMFold STRUCT_RASA 上の ElasticNetOpt、Stage-3 ADV_INTERACTIONS 上の SVROpt。

6. **各特徴量の由来:** Stage-5 ネスト OOF / 凍結 Test 予測。詳細ハッシュは `TMAPP_PRIVATE_TIE_AUDIT.csv`。

7. **回帰モデル:** 予測値レベルのメタ学習器（代表ラベル: convex_mae）。8 ラベルは名前が違うが Test 予測は同一。

8. **前処理:** Stage-5 の予測ブレンド／メタ学習。特徴レベル Ridge/Lasso の endgame パイプラインではない。

9. **ハイパーパラメータ:** メタ学習器ラベル依存（代表は Stage-5 inventory 参照）

10. **CVプロトコル:** 履歴 Stage-5 OOF（canonical Simple TVT とは未証明）

11. **Test予測時の最終学習方法:** 履歴 Round1 / Stage-5 の凍結提出・予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** 多様性セット上のメタ統合が Private で強い一方、8 名称は実質 1 予測。

15. **注意点:** Private 最良は事後。モデル選定・凍結には未使用。

**model_id:** `TmApp__META_diversity__convex_mae__HIST`


==================================================
D. HIC — canonical CV 最良モデル
==================================================

1. **モデルの完全な構成:** ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv) with LASSO

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** FEATURE_LINEAR canonical Simple TVT で cv_worst_mae 最小。

4. **スコア:** CV Primary=0.4872225781431885; CV Shadow=0.4833719610538652; CV mean=0.4852972695985268; CV worst=0.4872225781431885; Public=0.4702379457310565; Private=0.4641712708181718

5. **使用特徴量の完全な構成:** ESM-2 重鎖 PLM 埋め込み＋ SEQ_ALL 配列記述子＋ AROMATIC-TOPO 露出芳香族構造トポロジー特徴量（ESMFold Fv）＋ HYDRO_FIELD 連続疎水性場表面特徴量＋ TITRATION_SHAPE 静電滴定形状特徴量

6. **各特徴量の由来:** virtual_participant/round1_finalization/cache/round1_embeddings.npz::esm2__H | built via stage1 make_xy(SEQ_ALL) on competition sequences | structure_gap_closure/cache/aromatic_features_esmfold.csv | HYDRO-FIELD/features_esmfold.parquet | TITRATION-SHAPE/features_esmfold.parquet
   生次元: 1448.0; 有効次元: 1448.0（Lasso・PCA なしのため一致）

7. **回帰モデル:** Lasso（PCA なし）

8. **前処理:** 中央値補完＋ StandardScaler ＋ Lasso（no PCA）

9. **ハイパーパラメータ:** final_alpha=0.1; 非ゼロ係数=24.0 / 全特徴=1448.0

10. **CVプロトコル:** canonical_simple_tvt_v1

11. **Test予測時の最終学習方法:** FULL_DEV median Primary alpha

12. **Test予測値の由来:** CANONICAL_FULL_DEV_REFIT

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM（ESM-2 Heavy）あり。配列＋芳香族トポロジー＋疎水性場＋滴定形状を線形スパース回帰で統合。

15. **注意点:** CV のみで選定。

**model_id:** `HIC_HYDRO_TITRATION__LASSO`


==================================================
E. HIC — Public 最良モデル
==================================================

1. **モデルの完全な構成:** Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** public_mae 最小（事後）。特徴レベル Ridge/Lasso ではない。

4. **スコア:** CV Primary=0.4257721922584362; CV Shadow=0.4328749669956691; CV mean=0.4293235796270526; CV worst=0.4328749669956691; Public=0.4148392100594437; Private=0.4318145768709808

5. **使用特徴量の完全な構成:** 予測値の等平均ブレンド — (1) ESM-2 重鎖＋SEQ_ALL 融合の SVROpt 予測、(2) ESMFold STRUCT_SURFACE_CHEM の SVROpt 予測

6. **各特徴量の由来:** Round1 / Stage-5 予測アーティファクト（`base_model_ids` 展開）

7. **回帰モデル:** 等平均予測ブレンド（特徴行列 Ridge/Lasso ではない）

8. **前処理:** 各基底は履歴 Stage パイプライン（多くは SVR）。ブレンド重みは等平均。

9. **ハイパーパラメータ:** ブレンド重み固定（等平均）

10. **CVプロトコル:** 履歴 Stage OOF（canonical と未証明）

11. **Test予測時の最終学習方法:** 履歴 full-DEV refit / 凍結予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM を含む SVR 基底の予測平均。≈0.42 帯の歴史的強さの一端。

15. **注意点:** 事後最良。選定未使用。

**model_id:** `HIC__SIMPLE_blend_seq_surf__HIST`


==================================================
F. HIC — Private 最良モデル
==================================================

1. **モデルの完全な構成:** Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt); ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** private_mae 最小（事後）

4. **スコア:** CV Primary=0.4324716368936545; CV Shadow=nan; CV mean=nan; CV worst=nan; Public=0.4237480344059232; Private=0.4180226409189143

5. **使用特徴量の完全な構成:** 予測値の等平均ブレンド — ESM-2 重鎖 SVROpt 予測 ＋ ESMFold STRUCT_SURFACE_CHEM SVROpt 予測

6. **各特徴量の由来:** Round1 / Stage-5 予測アーティファクト

7. **回帰モデル:** 等平均予測ブレンド

8. **前処理:** 履歴 Stage パイプライン上の基底予測の平均

9. **ハイパーパラメータ:** 等平均

10. **CVプロトコル:** 履歴（Shadow 欠損の行あり）

11. **Test予測時の最終学習方法:** 履歴凍結予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM（ESM-2）ありの 2 モデル平均が Private で最安。feature-level 最良（CV）とは別系統。

15. **注意点:** 事後最良。選定未使用。

**model_id:** `HIC__SIMPLE_blend_esm2_surf__HIST`


## 付録 — TmApp GLOBAL AbLingua Ridge の有効次元

| 特徴ブロック | 生次元 | 変換 | 有効次元 |
|---|---:|---|---:|
| AbLang2_HL_paired | 480 | raw (no PCA) | 480 |
| SEQ_BASIC | 78 | raw (no PCA) | 78 |
| BIOEMU_NEW_PAIRWISE | 10 | raw (no PCA) | 10 |
| M1_PROTEINMPNN | 1 | raw (no PCA) | 1 |
| AbLingua_HL_mean | 2560 | fold-local PCA | 32 |
| __TOTAL__ | 3129 | concat after per-block transforms | 601 |
