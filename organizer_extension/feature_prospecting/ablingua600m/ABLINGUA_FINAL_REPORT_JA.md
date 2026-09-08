# AbLingua-600M TmApp Endgame — 最終報告

> **ERRATUM（PARENT 監査後）**  
> Sprint A の `load_seq_basic` が SEQ_BASIC を全 NaN にしていた。  
> 権威ある `CURRENT_RECIPE + AbLingua GLOBAL` = **Primary 2.7466 / Shadow 2.8227**（guided sprint B と一致）。  
> ADD の「両側改善」は **撤回**（修正後 Δ −0.044 / +0.023 → MIXED）。  
> 詳細: `ABLINGUA_PARENT_RECONCILIATION_REPORT_JA.md`

## 冒頭回答（必須16問）

1. **AbLingua-600M extraction succeeded for 324/324?**  
   **YES**（DEV+Test、ラベル未使用）。

2. **Official tokenization token-length range?**  
   **min=103 / median=114 / max=140**（VH・VL合算）。いずれも `max_position_embeddings=256` 未満 → 切り捨てなし。

3. **CLS pooling supported?**  
   **CLS_UNSUPPORTED**。公式 `BioTokenizer.tokenize` / `Simple_Collator` は `[CLS]` を挿入しない（vocab に ID=2 はあるが公式 embedding パスでは未使用）。発明しない。

4. **H_mean Primary / Shadow MAE**（Ridge PCA32）: **3.347 / 3.422**

5. **L_mean Primary / Shadow MAE**（Ridge PCA32）: **3.363 / 3.469**

6. **HL_mean_concat Primary / Shadow MAE**（Ridge PCA32）: **3.186 / 3.162**

7. **HL_cls_concat?**  
   **N/A（CLS_UNSUPPORTED）**

8. **Which pooling stronger?**  
   **MASKED_MEAN only**（比較対象なし）。H/L 診断より **HL_mean_concat（PCA32）** が最良。

9. **Best AbLingua vs AbLang2（同一 Simple TVT）?**  
   AbLang2 `HL_paired` Ridge PCA32: **3.101 / 3.030**  
   AbLingua `HL_mean_concat` Ridge PCA32: **3.186 / 3.162**  
   → **AbLang2 が明確に優位**（Δ ≈ −0.085 / −0.132）。raw・固定 SVR でも同様。

10. **AbLingua + SEQ_BASIC は Primary∩Shadow 改善?**  
    SEQ 修正後も SEQ 単体への増分は正（再計算 ΔP=+0.154 / ΔS=+0.081）。  
    ただし **AbLang2+SEQ には届かない**。

11. **現行コンペ recipe で AbLang2 を AbLingua に置換?**  
    **NO**（強く悪化）。修正後 REPLACE Δ ≈ **−0.973 / −0.920**。

12. **現行 recipe に AbLingua を ADD?**  
    **NO（両側正ではない）**。SEQ 修正後 FREE Δ **−0.044 / +0.023**（MIXED）。  
    PARENT MAE **2.747 / 2.823**（recipe_only **2.703 / 2.846**）。  
    ~~旧誤報: +0.035/+0.064・MAE 2.811/2.991（SEQ 全 NaN）~~

13. **AbLang2 + AbLingua 直接融合?**  
    **NO（MIXED）**。修正後 FREE Δ **−0.033 / +0.035**。  
    ~~旧誤報: 両側正~~

14. **BASE_FIXED_ALPHA 生存?**  
    - ADD: **非生存**（Primary 負）  
    - REPLACE / standalone vs AbLang2: **非生存**

15. **Best endgame TmApp recipe（本スプリント単独）**  
    AbLingua GLOBAL の ADD は採用しない。recipe_only の方が Primary 良い。  
    Guided 後の候補は別レポート（PARENT+CDR3）を参照。

16. **Final verdict:** **ABL_DROP（competition ADD）** / standalone は従来通り AbLang2 劣後。  
    ~~旧 ABL_TRY（ADD）は SEQ bug による誤判定~~

---

## 抽出メタ

| 項目 | 値 |
|------|-----|
| Model | `IDEA-AI4S/AbLingua` rev `4d1272df…` |
| Tokenizer | `baysicx/AbLingua` 3-gram（HF に tokenizer 無し） |
| layers / hidden / max_pos | 30 / 1280 / 256 |
| dtype / batch / device | FP32 / 8 / RTX 3090 |
| wall time | **~14.3 s**（324 Abs × H+L） |
| pooling | MASKED_MEAN only |

## 候補サマリ表

| candidate | Primary MAE | Shadow MAE | ΔP / ΔS vs parent | fixed-α | verdict |
|-----------|-------------|------------|-------------------|---------|---------|
| HL_mean standalone PCA32 | 3.186 | 3.162 | −0.085 / −0.132 vs AbLang2 | — | ABL_DROP |
| HL + SEQ_BASIC (incr) | 3.186 | 3.162 | +0.262 / +0.286 vs SEQ | +0.279 / +0.321 | ABL_TRY vs SEQ; lose vs AbLang2 |
| REPLACE AbLang2 in recipe | 3.734 | 3.792 | −0.889 / −0.737 | same | ABL_DROP |
| ADD to current recipe | 2.811 | 2.991 | +0.035 / +0.064 | +0.035 / +0.055 | **ABL_TRY** |
| SEQ+AbLang2+AbLingua | 2.839 | 2.832 | +0.013 / +0.278 | +0.013 / +0.075 | **ABL_TRY** |

## Bootstrap（B=10000、記録用）

- ADD_recipe: Primary CI95 [−0.055, 0.120] p≤0≈0.22；Shadow [−0.071, 0.199] p≈0.18  
- A2+ABL: Primary CI crosses 0；Shadow 強い正  
→ 有意性は主張しない。分類は点推定の Primary∩Shadow 正に基づく **ABL_TRY**。

## ファイル

- `embeddings/ablingua600m_{H,L,HL}_mean*.parquet`
- `ABLINGUA_EXTRACTION_METADATA.json`
- `ABLINGUA_SIMPLE_TVT_SPEC.md` / `ABLINGUA_FEATURE_FREEZE.json`（ラベル前凍結）
- `ABLINGUA_SIMPLE_TVT_RESULTS.csv`
- `scripts/01_extract_embeddings.py` / `02_simple_tvt_eval.py`
- `vendor/AbLingua/`（公式 tokenizer）
