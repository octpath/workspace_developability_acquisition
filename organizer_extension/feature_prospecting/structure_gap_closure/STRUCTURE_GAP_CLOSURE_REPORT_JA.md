# Structure Gap Closure — 日本語報告

## 冒頭 10 問への回答

1. **以前の surface-patch graph は真の連続分子表面パッチか？**  
   **いいえ。** `RESIDUE_GRAPH_ONLY_OR_INCOMPLETE`（残基 CA 距離グラフ + `rasa_proxy`）。三角化 SAS/SES メッシュも真の Å² パッチ面積もなし。詳細は `SURFACE_PATCH_IMPLEMENTATION_AUDIT.md`。

2. **連続分子表面疎水パッチは HIC で AROMATIC-TOPO / incumbent を超えたか？**  
   **いいえ。** FreeSASA Lee–Richards + 外部表面サンプリングによる PRIMARY_SURFACE_DEF（KD/FP/BM 全スケール）を Fv 主・Fab 副で評価。ESM2-H / ESM2-H+AROMATIC / incumbent に対する ΔMAE は Primary・Shadow とも **正（悪化）** → `NO_SIGNAL`。

3. **Fab cavity / packing は TmApp 信号を足したか？**  
   **incumbent 増分としてはいいえ。** Standalone では Primary のみ median より僅かに良いが Shadow は悪化。対 AbLang2 / incumbent / residual はすべて悪化 → `NO_SIGNAL`（かつ prep 感度は主に `PREP_DEPENDENT`）。

4. **Buried unsatisfied polar / H-bond は TmApp を足したか？**  
   **incumbent 増分としてはいいえ。** Standalone は Primary+Shadow で median より改善（弱い standalone）だが、成熟 incumbent / AbLang2 への積み上げは悪化 → 増分基準では `NO_SIGNAL`。化学定義は prepared Fab 323 主。

5. **Fab domain-interface quality は TmApp を足したか？**  
   **いいえ。** BSA proxy・接触・H-bond/塩橋・肘角など。増分すべて悪化。古典 SC は実装せず `SHAPE_COMPLEMENTARITY_TECHNICAL_BLOCK`。

6. **陽性信号は raw vs prepared で頑健か？**  
   **陽性の incumbent 増分は無し。** 感度監査では PACKING/UNSAT/SURFACE の多くが `PREP_DEPENDENT`、INTERFACE は比較的 `PREP_ROBUST` 寄り。

7. **Fv 由来信号は生成器横断で頑健か？**  
   表面特徴の生成器 Spearman 中央値 ≈ **0.42**（ABB2/Boltz2 vs ESMFold）→ 中程度。予測方向の incumbent 増分は元々無し。

8. **Primary かつ Shadow で成熟 incumbent を改善した新 family は？**  
   **なし。**

9. **凍結幾何エンコーダ（GearNet 等）は？**  
   **`DEFERRED_TECHNICAL`（TECHNICAL_BLOCK）。** A–E 完了後の任意 Task F。TorchDrug/CUDA リスクと FeNNix 並行を避け未実施。

10. **恒久クローズできる構造情報 family は？**  
    - 連続表面疎水パッチ（本 PRIMARY_SURFACE_DEF）を HIC 増分源として追う価値は低い（AROMATIC-TOPO/incumbent 既に強い）。  
    - Fab cavity/packing・interface・buried-unsat の **単純 Ridge 増分**も現状クローズ候補（standalone 弱信号は残すが incumbent 非増分）。  
    - Shape complementarity 古典実装と GearNet は技術ブロック/延期のまま。

---

## 資源・並行

- FeNNix 本計算は **非干渉**（affinity 0–15 維持）。本タスクは **16–23（≤8コア）**。
- OpenMM / 追加 FeNNix / MD なし。

## Task A

分類: **`RESIDUE_GRAPH_ONLY_OR_INCOMPLETE`** → Task B 実施。

## 特徴凍結

`STRUCTURE_GAP_CLOSURE_SPEC.md` / `.json`（target-blind）。

## 評価サマリ（増分 ΔMAE; 負=改善）

全 family × 参照で **Primary かつ Shadow の同時改善なし**。詳細は `results/GAP_CLOSURE_SCORECARD.csv`。

| Family | Target | 増分 vs incumbent | 備考 |
|--------|--------|-------------------|------|
| CONT_SURFACE (KD/FP/BM/Fv) | HIC | NO_SIGNAL | 全スケール悪化 |
| DELTA_PATCH_CONTEXT | HIC | NO_SIGNAL | |
| PACKING_CAVITY | TmApp | NO_SIGNAL | prep 依存大 |
| BURIED_UNSAT | TmApp | NO_SIGNAL | standalone のみ弱 |
| FAB_INTERFACE | TmApp | NO_SIGNAL | SC = TECHNICAL_BLOCK |
| GearNet | — | TECHNICAL_BLOCK | DEFERRED |

## 成果物

- `SURFACE_PATCH_IMPLEMENTATION_AUDIT.md`
- `TMAPP_PACKING_CAVITY_FEATURES.csv`（raw Fab 324）
- `TMAPP_BURIED_UNSAT_FEATURES.csv`（prepared 323）
- `TMAPP_INTERFACE_FEATURES.csv`
- `HIC_CONTINUOUS_SURFACE_FEATURES.csv`
- `PREP_SENSITIVITY_AUDIT.csv`
- `GAP_CLOSURE_SCORECARD.csv`
- `GENERATOR_ROBUSTNESS_SURFACE.csv`

## 状態ラベル

`STRUCTURE_GAP_CLOSURE_COMPLETE_NO_INCUMBENT_INCREMENT`


## 追補（評価補強・FeNNix 非干渉）

- `FEATURE_FREEZE.json` を記録（特徴定義の事後変更なし）。
- `GAP_CLOSURE_ARTIFACT_AUDIT.csv`: TmApp family 要約スコア vs 鎖長 / clash / HL 補正。
- `GAP_CLOSURE_BOOTSTRAP_PROMISING.csv`: **BURIED_UNSAT standalone vs median** のみ B=10000（Primary+Shadow で standalone ΔMAE<0 のため）。**incumbent 増分は依然 NO_SIGNAL**。
