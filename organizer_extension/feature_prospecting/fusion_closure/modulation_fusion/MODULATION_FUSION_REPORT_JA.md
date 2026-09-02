# HIC AROMATIC Modulation Fusion Report（Gate M1–M5）

**Evidence:** ORGANIZER-EXPLORATORY  
**Final state:** `ORGANIZER_HIC_AROMATIC_MODULATION_FUSION_COMPLETE`  
**Spec freeze:** `MODULATION_FUSION_SPEC_FROZEN_BEFORE_SCORING=true`

前処理は P0 固定（ESM2→Scaler→PCA32→Scaler = `z`；AROMATIC 15d median+Scaler = `x`；Ridge α=100）。

---

## 冒頭の直接回答

1. **現行 concat Ridge は under-fusion か？** → **No（この実験範囲では）.** 明示相互作用・乗法変調・FiLM はいずれも CURRENT_CONCAT より悪化。
2. **z_i×x_j は改善したか？** → **No.** M2 ElasticNet は全 generator で concat より悪い（ESMFold ΔCV vs concat ≈ +0.054）。
3. **乗法変調は concat を超えたか？** → **No.** M3 r=1/2/4 すべて大幅悪化；rank↑でさらに悪化。
4. **FiLM はさらに改善したか？** → **No.** M4 も同様に悪化（OVERFIT）。
5. **additive-only で足りるか？** → Additive-only（M4_ADD_ONLY_R2）はニューラル中では最良だが **concat には届かない**（ESMFold ΔCV +0.052）。「乗法が特別に効く」証拠なし。
6. **支配的 aromatic 特徴は？** → **小さいサブセット:** `strongly_exposed_aromatic_SASA` / `_count`、次いで CDR/総露出 SASA。LOO では単特徴除去の影響は小さい（max ≈ +0.004 MAE）→ **SMALL_SUBSET_SIGNAL**（単一支配ではない）。
7. **重要特徴は fold/generator で安定か？** → **概ね安定:** 3 generator とも top に strongly-exposed aromatic SASA 系。
8. **変調は 3/3 generator 改善か？** → **どれも concat を再現可能に超えず**（reproducible_vs_CONCAT = 0件）。従来の concat 2/3（ESMFold/ABB2）はそのまま有効。
9. **追加ゲインの大きさは？** → 変調の「ゲイン」は負。従来 concat の PLM 比 ≈ **−0.03〜−0.05 MAE** が依然ベスト。

**Outcome class:** `CONCAT_ALREADY_SUFFICIENT` + `MODULATION_UNSTABLE_OVERFIT`（低 DoF ニューラルでも train 崩壊）。

---

## Primary results（Δ は CURRENT_CONCAT 基準）

### ESMFold

| Model | CV MAE | ΔPLM | ΔCONCAT | ΔPub | ΔPri | ShadowΔ |
|-------|--------|------|---------|------|------|---------|
| **B1_CURRENT_CONCAT** | **0.4557** | −0.041 | 0 | 0 | 0 | 0 |
| B0_PLM_ONLY | 0.4969 | 0 | +0.041 | +0.051 | +0.027 | +0.030 |
| M4_ADD_ONLY_R2 | 0.5080 | +0.011 | +0.052 | +0.015 | +0.067 | +0.096 |
| M2_BILINEAR_EN | 0.5096 | +0.013 | +0.054 | +0.029 | −0.010 | +0.042 |
| M3_MULT_R1 | 0.5816 | +0.085 | +0.126 | +0.169 | +0.203 | +0.151 |
| M3_MULT_R2 | 0.6207 | +0.124 | +0.165 | +0.205 | +0.274 | +0.170 |
| M3_MULT_R4 | 0.7115 | +0.215 | +0.256 | +0.300 | +0.326 | +0.244 |
| M4_FILM_R1 | 0.5631 | +0.066 | +0.107 | +0.160 | +0.211 | +0.118 |
| M4_FILM_R2 | 0.6372 | +0.140 | +0.181 | +0.193 | +0.237 | +0.174 |
| M4_FILM_R4 | 0.7229 | +0.226 | +0.267 | +0.224 | +0.256 | +0.244 |

ABB2 / Boltz2 も同様に **B1 が最良**。変調モデルに CV/Pub/Pri 同時改善なし。

---

## Gate M1 — どの AROMATIC 次元が効くか

### Single-feature vs PLM（ESMFold CV）

| Rank | Feature | ΔMAE vs PLM |
|------|---------|-------------|
| 1 | strongly_exposed_aromatic_SASA | **−0.036** |
| 2 | strongly_exposed_aromatic_count | −0.032 |
| 3 | CDR_aromatic_SASA | −0.029 |
| 4 | aromatic_exposed_SASA_total | −0.025 |
| … | patch counts / Trp | ≈0 |

最良単特徴（−0.036）は 15d concat（−0.041）にわずかに及ばず。

### Leave-one-out

除去時の悪化は最大でも ≈0.004（strongly_exposed_aromatic_SASA）。**単一特徴依存ではない。**

**分類:** `SMALL_SUBSET_SIGNAL`（強く露出した芳香族 SASA/カウント＋CDR 露出が主）。

---

## Gate M2 — 明示 bilinear

527d ElasticNet（凍結グリッド）。相互作用 group ‖C[:,j]‖₂ は CDR_aromatic_fraction / CDR count / patch 等が上位だが、**予測は concat より悪い**。係数の解釈可能性はあるが、HIC 増分証拠にはならない。

---

## Gate M3/M4 — 低ランク変調と過学習

| Model | params | train MAE (mean) | val MAE | gap | seed SD |
|-------|--------|------------------|---------|-----|---------|
| M3_R1 | 111 | 0.27 | 0.58 | −0.31 | 0.01 |
| M3_R2 | 159 | 0.16 | 0.62 | −0.46 | 0.03 |
| M3_R4 | 255 | 0.05 | 0.71 | −0.66 | 0.03 |
| M4_R2 | 255 | 0.15 | 0.64 | −0.48 | 0.04 |
| ADD_R2 | 159 | 0.38 | 0.51 | −0.13 | ≈0 |

**OVERFIT_CONCERN:** rank↑で train 崩壊・val 悪化。seed 分散は中程度。乗法/FiLM は additive-only より不安定。

---

## 物理解釈

- HIC に効くのは **強く露出した芳香族の量（SASA/count）** と **CDR 芳香族露出**。
- PLM 次元への乗法ゲートや bilinear 相互作用を入れても、N=162・固定低 DoF では **線形 concat 以上の情報を安定抽出せず**。
- 前回の「PLM×物理の相互作用が必要」仮説は、**この仕様では支持されない**。

---

## Outcome（切り分け）

| Axis | Result |
|------|--------|
| Predictive | **CONCAT_ALREADY_SUFFICIENT** |
| Modulation attempt | **MODULATION_UNSTABLE_OVERFIT** |
| Generator robustness | concat の従来 2/3 を変調は超えず；**GENERATOR_SPECIFIC_MODULATION の正例なし** |
| Physical | SMALL_SUBSET aromatic exposure signal |

---

## STOP

隠れ層拡大・rank>4・epoch/lr 探索・他物理ファミリー追加は行わない。

## Files

- `modulation_fusion/MODULATION_FUSION_SPEC.json`
- `modulation_fusion/MODULATION_MODEL_REGISTRY.csv`
- `modulation_fusion/MODULATION_FUSION_REPORT_JA.md`
- `modulation_fusion/results/PRIMARY_RESULTS.csv`
- `modulation_fusion/results/M1_FEATURE_IMPORTANCE.csv`
- `modulation_fusion/scripts/run_modulation_fusion.py`
