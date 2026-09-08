# 線形モデル・クロージャ報告（日本語）

英語版: `LINEAR_MODEL_CLOSURE_EN.md`

## 冒頭回答

1. **以前の「Top-3」は何を意味していたか**  
   endgame Ridge/Lasso の **狭い FEATURE_RECIPE_FREEZE 在庫内**の Top-3。履歴の全 organizer 線形モデルの総合ランキングではない。

2. **HIC recipe名がなぜ誤解を招いたか**  
   `ARO+TITR` などは芳香族＋滴定だけに見えるが、実際は **ESM-2 重鎖 PLM ＋ SEQ_ALL ＋ AROMATIC-TOPO ＋ …** を含む。

3. **HICの最良feature-levelモデルにPLMは入っていたか**  
   **はい（ESM-2 Heavy）**。構成: ESM-2 重鎖のタンパク質言語モデル埋め込み ＋ SEQ_ALL 配列記述子 ＋ AROMATIC-TOPO 露出芳香族構造トポロジー特徴量 ＋ HYDRO_FIELD 連続疎水性場表面特徴量 ＋ TITRATION_SHAPE 静電滴定形状特徴量。  
   別途 — Public 最良の PLM: はい；Private 最良の PLM: はい（いずれも予測ブレンド）。

4. **何件のhistorical linear modelを監査したか**  
   レジストリ 116 行（FEATURE_LINEAR 42；PREDICTION_LINEAR_BLEND 50；OTHER 24）。

5. **historical HIC ≈0.42を回収できたか**  
   **できた。** 例: Round1 PRIMARY 等平均 SVR ブレンド（Public≈0.420 / Private≈0.424）。Private<0.43 の行は 2。FEATURE_LEVEL CV Top-10 には混ぜない。

6. **TmApp canonical CV 最良モデル**  
   `TM_PARENT_ABLINGUA_CDR3__RIDGE`（生次元=5689.0、PCA後有効次元=633.0）

7. **TmApp Public 最良モデル**  
   `TmApp__ablang2__HL__RidgeOpt__HIST` — AbLang2 HL 連結、StandardScaler＋PCA48、RidgeOpt alpha=78.35145884661983

8. **TmApp Private 最良モデル**  
   `TmApp__META_diversity__convex_mae__HIST` — 8 ラベルは **同一予測ファミリー**（SHA256=`8b7f14e326c369ab…`）

9. **HIC canonical CV 最良モデル**  
   `HIC_HYDRO_TITRATION__LASSO`

10. **HIC Public 最良モデル**  
    `HIC__SIMPLE_blend_seq_surf__HIST`

11. **HIC Private 最良モデル**  
    `HIC__SIMPLE_blend_esm2_surf__HIST`

12. **CV/Public/Private間のrank shake**  
    上位は指標で入れ替わる。詳細は `RANK_SHAKE_ANALYSIS.csv`。MAE差が微小なときの順位差は過大解釈しない。

13. **Ridge vs Lasso の結論**  
    同一特徴では Ridge が CV-worst で多い。HIC 高次元連結では Lasso が有利。TmApp 権威レシピは Ridge＋AbLingua 選択的 PCA。

14. **advanced model用に凍結したfeature recipe**  
    `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`  
    TmApp: ['TM_PARENT_ABLINGUA_CDR3__RIDGE', 'TM_PARENT_ABLINGUA_GLOBAL__RIDGE', 'TM_BASE_BIOEMU_MPNN__RIDGE']  
    HIC: ['HIC_HYDRO_TITRATION__LASSO', 'HIC_ARO_CONTINUOUS_SURFACE__LASSO', 'HIC_ESM2_SEQ_AROMATIC__LASSO']

15. **participant bundleとの整合性**  
    整合（必要ブロック欠落なし）

16. **LINEAR_MODEL_CLOSED = YES**

## 今回のメタデータ修復（スコア不変）

- AbLingua Ridge の有効次元を PCA32 後の入力幅に修正
- TmApp Public 勝者の由来を Stage-2 根拠に基づき具体化（PCA48・SVR ではない）
- TmApp Private 8 同点を同一予測ファミリーと分類

## 整合性

- `REPORT_LANGUAGE_CONSISTENCY_AUDIT.md` を参照
