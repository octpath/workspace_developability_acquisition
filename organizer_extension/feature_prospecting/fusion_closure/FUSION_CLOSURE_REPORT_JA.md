# Fusion Closure Report（Gate 3A–3E）

**Evidence boundary:** ORGANIZER-EXPLORATORY（Public/Private は事後複製記述。未見テスト検証ではない）  
**Final state:** `ORGANIZER_FEATURE_PROSPECTING_FUSION_CLOSURE_COMPLETE`  
**Spec freeze:** `FUSION_CLOSURE_SPEC_FROZEN_BEFORE_SCORING`（`FUSION_CLOSURE_SPEC_HASH.json`）

PLM: TmApp = AbLang2 `ablang2__HL_paired`（480d） / HIC = ESM2-H `esm2__H`（1280d）  
出典: `virtual_participant/round1_finalization/cache/round1_embeddings.npz`  
前処理: fold-local StandardScaler → PCA32。物理特徴は fold-local StandardScaler。HP は外訓練のみの内 CV。

参考（記述のみ）: 歴史的 Round1 incumbent CV MAE ≈ TmApp 2.71 / HIC 0.43。本バッチの PLM_ONLY は PCA32+小グリッドのため数値は一致しない。**主比較は matched PLM_ONLY との ΔMAE。**

---

## Physical featuresはPLMと違う情報を持つか？

| Target | Family | PLM→physics（median CV R², ESMFold） | Fusion gain（要約） | Interpretation |
|--------|--------|--------------------------------------|--------------------|----------------|
| TmApp | POLAR-SAT | −0.00（NOT_USEFULLY） | Ridge: CV↓だが Public 悪化 → WEAK/MIXED。RBF 無益 | 情報はPLMと異なるが、PLM同時学習では再現可能な増分なし |
| TmApp | PKA-SHIFT（corrected） | −0.03（NOT_USEFULLY） | ESMFold Ridge のみ REPRODUCIBLE。ABB2/Boltz2 悪化 | 発電機特異的。直交化すると増分消失 |
| TmApp | VOID-EXPLICIT | −0.04（NOT_USEFULLY） | ABB2 Ridge のみ REPRODUCIBLE。他は無益 | 発電機特異的コントロール |
| HIC | AROMATIC-TOPO | 0.16（WEAKLY） | ESMFold/ABB2 で Ridge・RBFとも REPRODUCIBLE。Boltz2 は MIXED | **相補的物理の最有力**（2/3 generators） |
| HIC | STATIC-SAP（neg ctrl） | −0.04（NOT_USEFULLY） | Boltz2 で REPRO；ESMFold/ABB2 は MIXED/部分 | 単独物理は NO_EVIDENCE だったが、微小な方向一致あり（CI はしばしば 0 を含む） |
| HIC | HYDRO-FIELD（neg ctrl） | 0.08（WEAKLY） | 点推定では ALL3 方向改善（Δ小さい） | 単独では NO_EVIDENCE。融合の「再現」は符号ベースで過大評価しうる |

---

## 1. Gate 3A — PLM → physics predictability

Dev N=162、`opt_joint_group_k5_s42`、ラベル不使用。PLM→PCA32→multi-output Ridge（α∈{0.1,1,10,100}、標準化物理 MSE）。

### Family summary（中央値）

| Family | Gen | median R² | median Spearman | frac R²>0 | class |
|--------|-----|-----------|-----------------|-----------|-------|
| POLAR | ESM/ABB/Boltz | −0.00 / 0.01 / 0.10 | 0.20 / 0.20 / 0.34 | 0.45–0.73 | NOT / WEAK / WEAK |
| PKA | all3 | −0.03〜−0.08 | 0.12–0.19 | ≤0.40 | NOT_USEFULLY |
| VOID | all3 | ≤0 | ≤0.18 | ≤0.30 | NOT_USEFULLY |
| AROMATIC | ESM/ABB/Boltz | 0.16 / 0.15 / −0.02 | 0.42 / 0.40 / 0.25 | 0.80 / 0.73 / 0.47 | WEAK / WEAK / NOT |
| STATIC-SAP | all3 | ≤0 | 0.18–0.25 | ≤0.50 | NOT_USEFULLY |
| HYDRO | ESM/ABB/Boltz | 0.08 / 0.15 / −0.03 | 0.33 / 0.39 / 0.21 | 0.64 / 0.71 / 0.50 | WEAK / WEAK / NOT |

### 意味特徴（ESMFold）

- **AROMATIC:** `CDR_aromatic_SASA` R²≈0.28、Spearman≈0.52；`largest_aromatic_patch_exposed_SASA` R²≈0.22  
- **POLAR:** `buried_unsatisfied_polar_count` R²≈−0.02；`CDR_buried_unsat_count` R²≈0.07  
- **PKA:** `mean_abs_delta_pKa` 低；`basic_mean_abs_delta` のみ R²≈0.22  

**結論:** どのファミリーも HIGHLY/PARTLY（median R²≥0.20）には達しない。物理記述子は全体として PLM から十分再構成されない → **INFORMATION_NOT_ENCODED_BY_PLM（弱い〜中程度の相関は一部あり）**。

---

## 2. TmApp — POLAR-SAT × AbLang2

| Model | Gen | ΔCV | ΔPublic | ΔPrivate | class |
|-------|-----|-----|---------|----------|-------|
| Ridge | ESMFold | **−0.106** | +0.008 | −0.201 | WEAK_OR_MIXED |
| Ridge | Boltz2 | −0.041 | −0.108 | +0.013 | WEAK_OR_MIXED |
| Ridge | ABB2 | +0.075 | +0.124 | +0.087 | NO |
| RBF | all3 | ≥0（CV） | — | — | NO |

- Orthogonal Ridge（ESMFold）: ΔCV −0.068 だが Public +0.020 → 依然 MIXED  
- **POLAR は AbLang2 に再現可能な増分を与えない。** 非線形 RBF は線形より悪い。  
- 三分類: ASSOCIATION（単独バッチで示唆）あり / NOT_ENCODED_BY_PLM あり / **INCREMENTAL_WITH_PLM なし**

---

## 3. TmApp — corrected PKA-SHIFT

| Model | Gen | ΔCV | ΔPub | ΔPri | class |
|-------|-----|-----|------|------|-------|
| Ridge | **ESMFold** | −0.034 | −0.066 | −0.059 | **REPRODUCIBLE** |
| Ridge | ABB2/Boltz2 | +0.08 | + | mixed | NO |
| RBF | all | 悪化 or NO | | | NO |
| Ortho Ridge ESMFold | | +0.011 | −0.021 | −0.015 | MIXED（増分消失） |

- 発電機一貫性: **GENERATOR_SPECIFIC_FUSION**  
- 解釈: CASE B（ESMFold Ridge）だが ortho で消える → PLM 予測可能成分との共線性／不安定。  
- **補正 PKA の融合益は ESMFold 線形に限定。** RBF は無益。

---

## 4. TmApp — VOID-EXPLICIT

| Model | Gen | class |
|-------|-----|-------|
| Ridge | ABB2 | REPRODUCIBLE（ΔCV −0.052） |
| 他 | | NO / 悪化 |

発電機特異的。canonical 結論には載せないコントロール。

---

## 5. HIC — AROMATIC-TOPO × ESM2-H

| Model | Gen | ΔCV | ΔPub | ΔPri | class |
|-------|-----|-----|------|------|-------|
| Ridge | ESMFold | −0.041 | −0.051 | −0.027 | **REPRODUCIBLE** |
| Ridge | ABB2 | −0.035 | −0.060 | −0.031 | **REPRODUCIBLE** |
| Ridge | Boltz2 | −0.053 | −0.066 | +0.015 | WEAK_OR_MIXED |
| RBF | ESMFold | −0.025 | −0.046 | −0.007 | **REPRODUCIBLE** |
| RBF | ABB2 | −0.000 | −0.044 | −0.028 | **REPRODUCIBLE** |
| Ortho Ridge | ESM/ABB | REPRODUCIBLE | | | 直交成分も有用 |

- 発電機: **FUSION_GAIN_2OF3**  
- ESM2 は AROMATIC を弱くしか予測できない（median R²≈0.16）が、融合は改善 → **STRONG_COMPLEMENTARY_PHYSICS（2/3）**  
- Ridge と RBF の両方で改善。**非線形必須ではない**（線形で十分／同等）。  
- 三分類: ASSOCIATION あり / NOT_ENCODED（部分） / **INCREMENTAL_WITH_PLM あり（2/3 generators）**

---

## 6. HIC — STATIC-SAP / HYDRO-FIELD（negative controls）

**STATIC-SAP:** 単独は NO_EVIDENCE。融合では Boltz2 で REPRO、他は MIXED。Δ は小さい。ortho も Boltz2 Ridge のみ REPRO。

**HYDRO-FIELD:** 単独は NO_EVIDENCE。符号ベースでは ESMFold/ABB2（Ridge+RBF）と Boltz2 RBF が REPRO → generator_consistency **FUSION_GAIN_ALL3**。ただし |ΔCV|≈0.01–0.02 で bootstrap CI は多くの場合 0 を含む。

**解釈注意:** 凍結ルールは符号一致のみ。コントロールの「再現益」は **微小・CI 脆弱**であり、AROMATIC ほどの効果量ではない。単独物理バッチの否定的結論と矛盾する強い主張はしない。

---

## 7. 非線形 PLM×physics 相互作用は有用か？

- **TmApp:** RBF 融合は Ridge より悪化が多い → **NO**（`POSSIBLE_PLM_PHYSICS_INTERACTION_SIGNAL` 不成立）  
- **HIC AROMATIC:** Ridge で既に再現益。RBF も同方向だが追加の必須性なし → 相互作用特異シグナルというより **加法的相補**

---

## 8. PLM-orthogonalized physics は助けたか？

- **AROMATIC:** ortho Ridge が ESMFold/ABB2 で REPRO → **PLM で説明できない残差成分が HIC に効く**証拠  
- **PKA ESMFold:** plus は REPRO だが ortho は MIXED/悪化 → 益が PLM 共変成分に依存  
- **POLAR/VOID:** 概ね無益のまま  
- **HYDRO/SAP:** 微小・発電機依存

---

## 9. 正の融合結果は発電機再現か？

| Family | Consistency |
|--------|-------------|
| AROMATIC | **FUSION_GAIN_2OF3**（主結論の柱） |
| HYDRO | FUSION_GAIN_ALL3（効果量小・注意） |
| STATIC-SAP | FUSION_GAIN_2OF3（弱） |
| PKA / VOID | GENERATOR_SPECIFIC |
| POLAR | NO_FUSION_GAIN |

---

## 10. Gate 3D — Structure-guided PLM pooling

**`PRIOR_STRUCTURE_GUIDED_POOLING_SPEC_NOT_AVAILABLE`**

リポジトリ内に PDB 骨格ベースの 3D ループ検出→PLM pooling の厳密実装は見つからず（CDR 配列プールは別物）。新規検出器は発明しない。ユーザーからの prior 仕様があれば follow-up 可能。**Fusion Closure はブロックしない。**

---

## 11. 直接結論

1. **物理特徴は PLM と異なる情報を持つか？**  
   **Yes（弱い〜部分的）** — median R² は概ね ≤0.16。特に PKA/VOID/SAP は NOT_USEFULLY；AROMATIC/一部 POLAR/HYDRO は弱い予測可能性のみ。

2. **その異なる情報はターゲット予測を改善するか？**  
   - **HIC × AROMATIC:** **Yes（2/3 generators、Ridge でも RBF でも、ortho でも）** — 本クロージャの中心的肯定結果。  
   - **TmApp × POLAR:** **No**（再現可能な PLM 同時融合増分なし）。  
   - **TmApp × PKA/VOID:** 発電機特異的な点推定のみ。canonical な肯定結論にしない。  
   - **HYDRO/SAP コントロール:** 符号上の微小改善あり得るが、単独否定と効果量から **強い相補主張はしない**。

### 三結論の切り分け（ファミリー別）

| Family | A PHYSICS_ASSOCIATION | B NOT_ENCODED_BY_PLM | C INCREMENTAL_WITH_PLM |
|--------|----------------------|----------------------|------------------------|
| POLAR | あり（先行バッチ） | 概ね Yes | **No** |
| PKA | 混合（ESMFold） | Yes | ESMFold Ridge のみ（脆弱） |
| VOID | 脆弱 | Yes | ABB2 のみ |
| AROMATIC | Yes | Yes（弱予測） | **Yes（2/3）** |
| STATIC-SAP | No（単独） | Yes | 弱・発電機依存 |
| HYDRO | No（単独） | 部分 | 符号上あり・効果量小 |

---

## 12. Optional combo

HIC で複数ファミリーが REPRO を出したため、**POST_FAMILY_SELECTION_EXPLORATORY**（ESMFold: AROMATIC+STATIC-SAP+HYDRO）を実行。

| Model | ΔCV | ΔPub | ΔPri | class |
|-------|-----|------|------|-------|
| Ridge | −0.032 | −0.045 | −0.029 | REPRODUCIBLE |
| RBF | −0.029 | −0.028 | +0.004 | WEAK_OR_MIXED |

→ キッチンシンクは AROMATIC 単体を大きく超えず。**元ファミリー主張の検証には使わない。**

TmApp は発電機不一致のため結合モデルは見送り。

---

## 13. Files / audit

| Artifact | Path |
|----------|------|
| Spec | `fusion_closure/FUSION_CLOSURE_SPEC.json` |
| Spec hash | `fusion_closure/FUSION_CLOSURE_SPEC_HASH.json` |
| Registry | `fusion_closure/FUSION_EXPERIMENT_REGISTRY.csv` |
| Summary | `fusion_closure/FEATURE_FUSION_SUMMARY_CURRENT.csv` |
| Leakage | `fusion_closure/FUSION_LEAKAGE_AUDIT.json` |
| Gate3A | `fusion_closure/results/GATE3A_PREDICTABILITY_SUMMARY.csv` |
| Fusion | `fusion_closure/results/FUSION_RESULTS_ALL.csv` |
| Gate3D | `fusion_closure/GATE3D_STATUS.json` |
| Combo | `fusion_closure/results/POST_FAMILY_SELECTION_EXPLORATORY.csv` |
| Runner | `fusion_closure/scripts/run_fusion_closure.py` |

Leakage audit: scaler/PCA/residualizer/HP は outer-train のみ。Public/Private はチューニング不使用。同一外折で matched Δ。

---

## STOP

新物理ファミリー・PLM fine-tune・Optuna・XGB・MD・新規ループ検出器は開始しない。
