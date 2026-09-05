# Foundation Stability v2 — TmApp 報告

**状態:** `ORGANIZER_TMAPP_FOUNDATION_STABILITY_V2_COMPLETE`  
**証拠境界:** ORGANIZER-EXPLORATORY（Public/Private は post-reveal 複製のみ）  
**範囲:** 実験 TmApp は **Fab**；FeNNix は **Fv**；BioEmu は **鎖ごと VH/VL**（界面なし）  
**HIC:** 対象外  
**PLM 基準:** `ablang2__HL_paired` → fold-local StandardScaler → PCA32 → Ridge（凍結）

---

## 冒頭サマリ

| Family | Standalone (CV MAE) | Δ vs AbLang2 (CV) | robustness / artifact | verdict |
|--------|---------------------|-------------------|------------------------|---------|
| FeNNix-v2 (ESMFold) | 3.556（中央値未達） | **+0.141**（悪化） | generator **FRAGILE**；clash 依存は **大幅低下** | NO_INCREMENT |
| FeNNix-v2 (ABB2) | 3.609（中央値未達） | **+0.096**（悪化） | 同上 | NO_INCREMENT |
| BioEmu-v1.2 | 3.464（≈中央値） | **+0.060**（悪化〜中立） | 構造生成器非依存；ΔG proxy 不採用 | NO_INCREMENT |

**全体結論クラス:** `FOUNDATION_V2_NO_INCREMENT`

---

## FeNNix v2（直接回答）

| 項目 | 結果 |
|------|------|
| Relaxation | 648/648 完走；FIRE 収束率 ABB2 ≈52%、ESMFold ≈1.5%（max_steps=200 で打ち切り；backbone RMSD≈0） |
| Clash / force | ESMFold 中央値 F_rms 3.35→1.24；ABB2 0.77→0.47。severe clash 中央値 Δ≈0（ABB2 は残存） |
| Clash 相関 | v1 `P1_dE`↔OpenMM clash Spearman **≈0.94** → v2 \(K_E\) 最大 |ρ| **≈0.21**（同 proxy）→ **v1 の激しい clash 連動は実質解消** |
| Best standalone MAE | ESMFold **3.556**（vs median 3.438）；ABB2 **3.609** → **standalone 信号なし** |
| Best incremental ΔMAE | いずれも PLM より悪化（ESMFold +0.14；ABB2 +0.10）；残差再構成も同様 |
| ESMFold↔ABB2 | 特徴間 Spearman 中央値 **≈0.33** → **FRAGILE**（v1 同様に generator 非一貫） |
| v1 アーティファクトは解けたか | **clash 依存は解けた**。しかし予測信号は消え、PLM 増分も出ない → 「アーティファクト除去後に安定性が残る」仮説は **支持されない** |
| v1 を上回るか | **上回らない**（v1 ABB2 standalone 3.19 は clash 連動の見かけ信号だった可能性が高い） |

### §24 Q1–Q6

1. **緩和は clash/force を下げたか:** force は明確に低下；severe clash カウントは中央値でほぼ不変（特に ABB2）。  
2. **対称ねじれ曲率で clash 依存は消えたか:** **はい（大幅に）**。  
3. **generator 一貫性は上がったか:** **いいえ（FRAGILE のまま）**。  
4. **standalone TmApp 信号:** **なし**。  
5. **AbLang2 超の情報:** **なし**（増分・残差とも悪化）。  
6. **v1 より優秀か:** 科学的妥当性（アーティファクト分離）では改善；**予測性能では劣る／無信号**。

---

## BioEmu-v1.2（直接回答）

| 項目 | 結果 |
|------|------|
| Selected N | **16**（vs N=64: 記述子 Spearman≥0.95 かつ NAD≤10% を満たす最小） |
| 完了率 | **324/324** Abs、VH+VL **648/648** 鎖 |
| Standalone MAE | CV **3.464**（Δ vs median +0.026, p≈0.36）→ **明確な単独信号なし** |
| Incremental ΔMAE | CV **+0.060** vs PLM（悪化）；Public でわずかに改善方向だが CI は 0 跨ぎ |
| 有用そうな記述子 | contact 系が相対的に最良（単独 MAE 3.411）だが中央値超えは非有意；RMSF/Rg/CDR は弱い〜悪化 |
| VH vs VL | VH 単独 MAE 3.455 ＞ VL 3.473（わずかに VH 優位）が **どちらも無信号** |
| ΔG-like | **`BIOEMU_DG_PROXY_NOT_SCIENTIFICALLY_JUSTIFIED`**（鎖ごとモノマー ≠ Fab Tm） |

### §24 Q7–Q12

7. **必要サンプル数:** 収束パイロット上 **N=16 で十分**（凍結）。  
8. **アンサンブルに standalone 信号:** **実質なし**。  
9. **寄与が大きい鎖:** 数値上は VH がわずかに良いが **主張に足る差ではない**。  
10. **CDR/HCDR3 柔軟性:** CDR 系単独は中央値未達（Spearman 負）→ **有用と言えない**。  
11. **AbLang2 超:** **なし**。  
12. **折りたたみ ΔG 様量:** **科学的に正当化できず未定義**。

---

## Cross-family（§24 Q13）

局所エネルギー曲率（FeNNix-v2）と平衡アンサンブル記述子（BioEmu）を比較すると、**どちらも PLM 増分を持たない**。相対的には BioEmu の方が standalone が中央値に近く増分悪化も小さいが、**「強い方」を選ぶほどの差ではない**。

解釈ルール遵守: 正の結果＝「TmApp と関連する予測情報を含む」であり、Fab 融解の機構説明ではない。本 v2 ではその予測情報も確認できなかった。

---

## 最終推奨

1. **最終 TmApp モデルに FeNNix-v2 / BioEmu-v1.2 特徴を持ち込まない**  
2. v1 の「foundation が効く」見かけは **構造アーティファクト** だった、という解釈が v2 で強化された  
3. v2 は「アーティファクト除去後も foundation が効くか」への答えとして **効かない（増分なし）**  
4. 次の物理系探索は、Fab 規模・界面・実験条件に近い入力が要る（本フェーズ外）。**AbLingua へは自動進行しない**

---

## 成果物

| File | Role |
|------|------|
| `FOUNDATION_STABILITY_V2_SPEC.md` / `.json` | スコア前凍結仕様 |
| `FENNIX_V2_RELAXATION_QC.csv` | R1 緩和 QC |
| `FENNIX_V2_CURVATURE_FEATURES.csv` | 対称ねじれ曲率 |
| `FENNIX_V1_V2_ARTIFACT_COMPARISON.csv` | v1/v2 vs clash・force |
| `FENNIX_V2_GENERATOR_ROBUSTNESS.csv` | ESMFold↔ABB2 |
| `FENNIX_V2_RESULTS.csv` | FeNNix 評価 |
| `BIOEMU_SAMPLE_CONVERGENCE.csv` | N 収束 |
| `BIOEMU_SAMPLE_COUNT_DECISION.md` | N=16 決定 |
| `BIOEMU_V12_FEATURES.csv` | アンサンブル特徴 |
| `BIOEMU_V12_RESULTS.csv` | BioEmu 評価 |
| `BIOEMU_DG_PROXY_STATUS.md` | ΔG 不採用理由 |
| `FOUNDATION_V2_{STANDALONE,INCREMENTAL,RESIDUAL,BOOTSTRAP}.csv` | 共通評価 |
| `FOUNDATION_STABILITY_V2_REPORT_JA.md` | 本報告 |

**Do not modify** prior `feature_prospecting` family results outside this branch.  
**Do not** auto-start AbLingua.
