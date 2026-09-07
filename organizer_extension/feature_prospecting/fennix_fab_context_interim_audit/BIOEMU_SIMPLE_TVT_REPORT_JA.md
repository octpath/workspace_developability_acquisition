# BioEmu — Simple TVT 直接融合レポート

**日付:** 2026-09-08  
**プロトコル:** Simple TVT（FREE_ALPHA / BASE_FIXED_ALPHA）。BASE に PCA はかけない。高次元 STRUCTURE のみ train 内 PCA32。  
**FeNNix 本番:** 未変更。

---

## 冒頭 8 問

1. **この Exact Simple TVT 直接融合で BioEmu を以前評価したか？**  
   **いいえ。** `SIMPLE_TVT_NOT_YET_RUN`。従来は standalone / residual / frozen-OOF late-fusion のみ（`BIOEMU_ISOLATED_CONFIRMED_NO_INCREMENT`）。

2. **NEW_CONTACT は TmApp を改善するか？**  
   **はい（弱い〜中程度）。** Primary Δ=+0.014 / Shadow Δ=+0.028。`survives_fixed_alpha=True`。

3. **NEW_PAIRWISE は TmApp を改善するか？**  
   **はい。** Primary Δ=+0.034 / Shadow Δ=+0.033。`DIRECT_STRONG_CANDIDATE`、fixed-alpha 生存。

4. **COMBINED frozen BioEmu ブロックは TmApp を改善するか？**  
   **はい（弱い）。** Primary Δ=+0.004 / Shadow Δ=+0.064。方向は両方正だが Primary は小さい。V12 旧ブロックは **両方とも悪化**（Δ≈−0.05/−0.043）。

5. **HIC 探索的評価で改善はあるか？**  
   CONTACT/COMBINED は Primary のみ微正・Shadow 負（`DIRECT_MIXED`）。PAIRWISE は両方非正。**強い機序主張は不可**（`EXPLORATORY_HIC_BIOEMU`）。

6. **Primary / Shadow 方向は一致するか？**  
   TmApp: CONTACT / PAIRWISE / FLEX / COMBINED は一致改善。SHAPE と V12 は一致しないまたは両方悪化。

7. **BASE_FIXED_ALPHA で生き残るか？**  
   TmApp の CONTACT / PAIRWISE / FLEX / COMBINED は **すべて生存**（Δ が FREE と同値または同様に正）。

8. **従来の standalone/late-fusion 結論と違うか？**  
   **大きく違う。** Late-fusion/residual では `NO_INCREMENT` だったが、直接特徴融合では NEW_* 系が再現可能な正の Δ を示す。これは融合アーキテクチャ依存であり、BioEmu 記述子に情報が無いことの証明にはならない。

---

## 数値要約（TmApp）

| family | Primary Δ | Shadow Δ | fixed 生存 | free_verdict |
|--------|-----------|----------|------------|--------------|
| BIOEMU_NEW_PAIRWISE | +0.034 | +0.033 | Yes | DIRECT_STRONG_CANDIDATE |
| BIOEMU_NEW_FLEX | +0.027 | +0.015 | Yes | DIRECT_WEAK_CANDIDATE |
| BIOEMU_NEW_CONTACT | +0.014 | +0.028 | Yes | DIRECT_WEAK_CANDIDATE |
| BIOEMU_NEW_COMBINED | +0.004 | +0.064 | Yes | DIRECT_WEAK_CANDIDATE |
| BIOEMU_NEW_SHAPE | −0.020 | +0.023 | No | DIRECT_MIXED |
| BIOEMU_V12 | −0.055 | −0.043 | No | DIRECT_NO_IMPROVEMENT |

BASE = `ABLANG2_HL_PAIRED+SEQ_BASIC`（dim=558）。特徴パス: `bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv`。

## 解釈の切り分け

| 列 | BioEmu NEW_* (TmApp) |
|----|----------------------|
| SCIENTIFIC_VERDICT | PAIRWISE=`SCI_ROBUST_POSITIVE`；CONTACT/FLEX=`SCI_ROBUST_POSITIVE`（Δは中〜小） |
| COMPETITION_VERDICT | PAIRWISE=`COMP_PRIORITY`；他 NEW 正例=`COMP_TRY` |
| 機序主張 | 直接融合で有用な可能性。孤立 VH/VL の限界は残る。Fab 文脈 BioEmu は未評価。 |
