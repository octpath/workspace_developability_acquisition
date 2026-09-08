# AbLingua PARENT スコア不一致 — 監査報告

## 冒頭回答（必須12問）

1. **なぜ 2.811/2.991 と 2.747/2.823 が違うか？**  
   **Sprint A（`02_simple_tvt_eval.py`）の `load_seq_basic` が `make_xy` の RangeIndex を文字列化してから `reindex(ADI-*)` したため、SEQ_BASIC 78 列が全 NaN（162×78）。**  
   Sprint B（`04`）は `seq_b.index = ids` で正しく整列。同じ「CURRENT_RECIPE + AbLingua GLOBAL」でも **ベースレシピの SEQ 成分が実質欠落 vs 正常** でスコアが乖離した。

2. **親特徴行列は同一だったか？** **NO。** AbLingua GLOBAL は同一（sha一致）。recipe は SEQ_BASIC が A=全NaN / B=正常。

3. **DEV ID は同一か？** **YES**（N=162、同一 hash）。

4. **Primary/Shadow fold は同一か？** **YES**（`cv_primary.csv` / `cv_shadow.csv`、TEST=k / VAL=(k+1)%5）。

5. **AbLingua GLOBAL は同一か？** **YES**（`embeddings/ablingua600m_HL_mean_concat.parquet`）。RESIDUE_GLOBAL は未使用。

6. **PCA は同一か？** **意図は同一**（recipe に PCA なし、AbLingua のみ fold-local PCA32）。実装差は副因。主因は SEQ NaN。

7. **Ridge/alpha は同一か？** グリッド `{0.1,1,10,100}` は同一。SEQ 行列が違うため選択・予測が分岐。

8. **権威ある結果は？** **AUTHORITATIVE_PARENT = B**（正しい SEQ 整列 = `load_bases()` と同じ `index=ids`）。

9. **修正後 PARENT Primary/Shadow MAE？** **2.746600 / 2.822725**

10. **修正後 CDR3 ΔP/ΔS？** Guided スプリントは既に権威 PARENT を使用。**+0.0145 / +0.0378**（変更なし）。

11. **CDR3 は GUIDED_TRY のまま？** **YES**

12. **以前の AbLingua 結論は変わるか？** **YES。**  
    - 「ADD to current recipe が両側正」は **撤回**（SEQ 修正後 Δ **−0.044 / +0.023** → MIXED）。  
    - REPLACE は引き続き悪化。  
    - SEQ+AbLang2+AbLingua も MIXED。

---

## 経路対照

| | Sprint A (`c121f7bf`) | Sprint B (`27ce3f36`) |
|--|----------------------|----------------------|
| Script | `scripts/02_simple_tvt_eval.py` → `run_incr(recipe, HL)` | `scripts/04_guided_pooling_tvt.py` → `run_parent_plus(recipe,[HL],[])` |
| Result row | `ADD_to_current_recipe` plus MAE | `PARENT` |
| Reported | 2.811 / 2.991 | 2.747 / 2.823 |
| SEQ_BASIC | **全 NaN（bug）** | **正常** |
| AbLingua GLOBAL | HL_mean_concat | 同一ファイル |

## 成分比較

| component | A | B | same? |
|-----------|---|---|-------|
| AbLang2 | 480（値一致、列名のみ異） | 480 | YES values |
| SEQ_BASIC | 78 **all-NaN** | 78 finite | **NO** |
| BIOEMU_NEW_PAIRWISE | 10 | 10 | YES |
| M1_PROTEINMPNN | 1 | 1 | YES |
| AbLingua GLOBAL | 2560 | 2560 | YES |
| RESIDUE_GLOBAL | — | 未使用（診断のみ） | — |

## 予測差（再現）

| split | MAE_A (plus) | MAE_B (PARENT) | max\|pred_A−pred_B\| | corr |
|-------|--------------|----------------|----------------------|------|
| Primary | 2.8107 | 2.7466 | 2.64 | 0.959 |
| Shadow | 2.9908 | 2.8227 | 4.82 | 0.931 |

## SEQ 修正後の再計算（権威）

| model | Primary | Shadow | Δ vs recipe_only |
|-------|---------|--------|------------------|
| recipe_only (CURRENT_RECIPE) | **2.7027** | **2.8459** | — |
| PARENT = recipe + AbLingua GLOBAL | **2.7466** | **2.8227** | **−0.044 / +0.023** |
| PARENT + CDR3 (guided) | 2.7321 | 2.7850 | vs PARENT +0.015 / +0.038 |

**ADD は Primary を悪化**するため、以前の ABL_TRY（ADD）は無効。

## Guided pooling 再実行

**不要。** Guided スプリントは既に正しい SEQ 整列の PARENT を使用。CDR3 判定は維持。

## 回帰テスト

`scripts/test_parent_regression.py`:
- SEQ_BASIC index == competition ids
- nan 率ガード
- `assert_same_parent_predictions` → PARENT MAE == 2.746600 / 2.822725

## 修正ファイル

- `02_simple_tvt_eval.py` `load_seq_basic` 修正
- `ABLINGUA_SIMPLE_TVT_RESULTS.csv` 旧行に SUPERSEDED 注記 + CORRECTED 行追加
- `ABLINGUA_FINAL_REPORT_JA.md` 更新
- 本レポート
