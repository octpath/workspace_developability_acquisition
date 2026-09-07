# Simple TVT CV — 日本語報告

**注意:** `STRICT_NESTED_INCREMENT` の代替ではない。直接特徴融合の直交チェック。FeNNix 本番未変更。

## 冒頭 7 問

1. **直接特徴融合は TmApp を改善したか？**  
   Gap Closure TmApp family では **いいえ**（一貫改善なし）。Interim FeNNix の **CONSTANT** のみ Primary∧Shadow で暫定改善（`PROVISIONAL_SIMPLE_CV`；boot CI は 0 を含む）。

2. **直接特徴融合は HIC を改善したか？**  
   **はい（弱い）** — `CONTINUOUS_SURFACE`（ΔP=+0.017 / ΔS=+0.002）、`HIC_SURFACE_ALL`（+0.023 / +0.002）。いずれも Primary boot CI が 0 をまたぎ、Shadow `p_improve`≈0.55 程度で不安定。

3. **Strict nested で失敗しつつ Simple TVT で効いた Gap Closure は？**  
   **`HIC_SURFACE_ALL`**（`STRICT_NEGATIVE_SIMPLE_POSITIVE`）。連続表面は Strict 未評価のため `SIMPLE_ONLY_POSITIVE`。

4. **Interim FeNNix で Simple TVT 改善は？**  
   **CONSTANT のみ**一貫改善（暫定）。他は MIXED または NO_IMPROVEMENT。

5. **Primary / Shadow 一貫性は？**  
   一貫改善: CONTINUOUS_SURFACE / HIC_SURFACE_ALL / CONSTANT。混合: BURIED_UNSAT と大半の FeNNix。それ以外は両方とも非改善。

6. **STRICT_NESTED と結論は一致するか？**  
   TmApp Gap Closure 合算・界面系は **BOTH_NEGATIVE で一致**。HIC 表面と FeNNix CONSTANT は **プロトコル依存**（late-fusion 陰性 / 直接融合陽性）。

7. **プロトコル選択は科学的結論を実質変えるか？**  
   **一部で変える。** Incumbent late-fusion では「増分なし」でも、同一 Ridge への直接 concat では HIC 表面・暫定 CONSTANT に弱い改善が見える。ただし効果は小さく、Shadow でほぼ消える／CI が 0 を含むため、強い反証にはならない。

## BASE 定義

- TmApp: `ABLANG2_HL_PAIRED+SEQ_BASIC` dim=558
- HIC `CONTINUOUS_SURFACE`: BASE=`ESM2_H+SEQ_ALL+AROMATIC_TOPO` dim=1414
- HIC `HIC_SURFACE_ALL`: BASE=`ESM2_H+SEQ_ALL` dim=1395（aromatic は STRUCTURE 側）

## 要約表

| scope | family | Primary Δ | Shadow Δ | verdict |
|-------|--------|-----------|----------|---------|
| GapClosure | CONTINUOUS_SURFACE | +0.0166 | +0.0018 | SIMPLE_CV_CONSISTENT_IMPROVEMENT |
| GapClosure | HIC_SURFACE_ALL | +0.0228 | +0.0018 | SIMPLE_CV_CONSISTENT_IMPROVEMENT |
| GapClosure | BURIED_UNSAT | −0.0072 | +0.0499 | SIMPLE_CV_MIXED |
| GapClosure | CORE_DEFECT | −0.0832 | −0.0698 | SIMPLE_CV_NO_IMPROVEMENT |
| GapClosure | FAB_INTERFACE | −0.0254 | −0.0263 | SIMPLE_CV_NO_IMPROVEMENT |
| GapClosure | GAP_ALL | −0.1567 | −0.1192 | SIMPLE_CV_NO_IMPROVEMENT |
| GapClosure | PACKING_CAVITY | −0.0779 | −0.1347 | SIMPLE_CV_NO_IMPROVEMENT |
| InterimFeNNix | CONSTANT | +0.1160 | +0.0512 | SIMPLE_CV_CONSISTENT_IMPROVEMENT |
| InterimFeNNix | COMBINED_PREDECLARED | +0.0896 | −0.1590 | SIMPLE_CV_MIXED |
| InterimFeNNix | DELTA_ENV | +0.0288 | −0.2397 | SIMPLE_CV_MIXED |
| InterimFeNNix | DELTA_GEOM | +0.0254 | −0.0366 | SIMPLE_CV_MIXED |
| InterimFeNNix | INTERFACE | +0.0620 | −0.0309 | SIMPLE_CV_MIXED |
| InterimFeNNix | FULL_FAB_NORMALIZED | −0.0083 | −0.0070 | SIMPLE_CV_NO_IMPROVEMENT |

## vs Strict Nested

| family | interpretation |
|--------|----------------|
| FAB_INTERFACE / CORE_DEFECT / GAP_ALL / FULL_FAB_NORMALIZED | BOTH_NEGATIVE |
| HIC_SURFACE_ALL / CONSTANT | STRICT_NEGATIVE_SIMPLE_POSITIVE |
| CONTINUOUS_SURFACE | SIMPLE_ONLY_POSITIVE（Strict 未走行） |
| その他 | MIXED |

詳細: `SIMPLE_VS_STRICT_COMPARISON.csv` / 仕様: `SIMPLE_TVT_CV_SPEC.md`。

Interim FeNNix はすべて `PROVISIONAL_SIMPLE_CV`（完了サブセット N=100）。フルコホート完了後に同一プロトコルで再実行する。
