# ANM-SPECTRUM_v1

Paper: [Atilgan et al., Biophys. J. 2001](https://doi.org/10.1016/S0006-3495(01)76033-X)

Software: [ProDy (GitHub)](https://github.com/prody/ProDy)

Software paper: [Bakan et al., Bioinformatics 2011](https://doi.org/10.1093/bioinformatics/btr168)

**State:** `ORGANIZER_FEATURE_PROSPECTING_GATE2A_ANM_SPECTRUM_V1_COMPLETE`  
**ProDy:** 2.6.1（`.venv_anm`）

## Bottom line

| Target | Mechanistic prior | Empirical verdict | One-line conclusion |
|--------|-------------------|-------------------|---------------------|
| TmApp | `5_LIKELY_RELEVANT` | `NO_EVIDENCE_IN_CURRENT_DATA` | 弱い univariate softness 関連はあるが、canonical Ridge standalone / residual incremental は CV·Public·Private で再現しない |
| HIC | `2_UNLIKELY_PRIMARY` | `NO_EVIDENCE_IN_CURRENT_DATA` | 同 feature の cross-endpoint audit でも signal なし（prior と整合） |

Primary generator 報告は **ESMFold**。ABB2 / Boltz-2 も同方向。

---

## 1. ANMとは何か

Anisotropic Network Model（異方性ネットワークモデル）は、タンパク質を **Cα 原子を点・近接をばね** とみなした粗視化弾性ネットワークである。固有値スペクトルから「ゆっくり変形しやすい方向」と相対的な柔らかさ／硬さを読む。γ=1 は任意スケールであり、実験振動数そのものではない。

## 2. Why TmApp

低周波力学的柔軟性・packing／conformational stability と見かけの熱安定性（TmApp）の関係は物理的に自然で、prior は `5_LIKELY_RELEVANT`。

## 3. Why HIC is low-prior

HIC の主因は表面疎水相互作用であり、グローバルな低周波力学は直接因子ではない（`2_UNLIKELY_PRIMARY`）。本実験では feature を変えず cross-endpoint audit のみ。

## 4. Method / references

- cutoff=15 Å, gamma=1.0, n_modes=20（ProDy standard ANM）
- Fv = VH+VL を単一ネットワーク
- 20 canonical features（`FEATURE_SPEC.json`）
- Structures: `STRUCTURE_INPUT_CROSSWALK_v2.csv`（ESMFold / ABB2 / Boltz-2）

## 5. Three-generator robustness

| Pair | median Spearman |
|------|----------------:|
| ESMFold vs ABB2 | 0.619 |
| ESMFold vs Boltz2 | 0.570 |
| ABB2 vs Boltz2 | 0.514 |

- **minimum pairwise median Spearman = 0.514**
- **Class: `MODERATE`**（discard しない）

Extraction: **324/324 × 3** SUCCESS。extra zero modes = 0。

## 6. Standalone TmApp（ESMFold primary）

| Split | MAE | baseline median MAE | Spearman |
|-------|----:|--------------------:|---------:|
| CV | 3.545 | 3.438 | 0.122 |
| Public | 3.719 | （Dev median） | 0.155 |
| Private | 3.775 | | 0.083 |

Standalone class: `TEST_ONLY_POSTHOC`（CV では baseline を下回れず）。

ABB2 は CV のみ baseline 超え（`CV_ONLY`）だが Public で Spearman 反転。

## 7. Incremental TmApp（CANONICAL_RESIDUAL_RIDGE）

Reference = Round1 `TmApp__META_performance__ridge_100.0`（CV MAE 2.713）。

| Split | delta MAE (candidate − reference) | bootstrap 95% CI |
|-------|----------------------------------:|------------------|
| CV | −0.0059 | [−0.061, 0.047] |
| Public | +0.0127 | — |
| Private | +0.0004 | — |

Incremental class: `CV_ONLY_INCREMENT`（微小かつ Test 非再現）。

Residual Ridge implementation audit: **PASS**（`RESIDUAL_RIDGE_IMPLEMENTATION_AUDIT.json`）。

## 8. Tm range diagnostics

候補予測の low/high Tm bias と quintile MAE を保存（二次解釈）。圧縮レンジの明確な回復は主張しない。

## 9. HIC cross-endpoint audit

Label: `SECONDARY_CROSS_ENDPOINT_AUDIT`

| Split | standalone MAE | delta MAE vs Round1 blend |
|-------|---------------:|--------------------------:|
| CV | 0.596 | +0.044 |
| Public | 0.594 | +0.048 |
| Private | 0.544 | +0.019 |

Standalone / incremental とも `NO_SIGNAL` / `NO_INCREMENT`。

## 10. Confounds

collectivity・高次 λ は **n_CA / n_edges / mean_degree** と |ρ| が大きい（しばしば >0.6）。  
canonical 20 features に長さを入れていないのはこのため。解釈時は network-size confounding に注意。

## 11. Generator-specific findings

- TmApp generator signal: `NO_SIGNAL`
- HIC generator signal: `NO_SIGNAL`
- 3 generator とも residual の再現改善なし

## 12. Scientific interpretation

事前指定の softness（`log_softness_1_20`）と TmApp の Dev Spearman はおよそ 0.16–0.23（generator 依存）で、方向は「柔らかい ↔ Tm やや高い」ではなく softness と Tm の弱い正相関として見えるが、**予測タスクでは median baseline / Round1 residual を再現的に超えず**。  
現 setup（静的予測構造・γ=1 ANM・20 mode summary・Ridge）では、mechanistic prior を empiric に支持する証拠は得られなかった。

## 13. Limitations

- 予測 Fv 構造依存；ANM は相対スペクトルのみ
- γ スケール任意
- collectivity 等は鎖長と強く相関
- v1 内で cutoff / mode 数の後追い調整はしない（必要なら `ANM-SPECTRUM_v2_POSTHOC`）
