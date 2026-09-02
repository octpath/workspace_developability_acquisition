# PKA-SHIFT_v1 Technical Correction

PROPKAの99.99は極端なpKaではなく、ジスルフィド結合Cysを示す特殊値（`cysteine_bridge` sentinel）だったため、数値特徴量から除外して再評価した。

Original v1 artifacts は **immutable**。本ディレクトリが補正系列。

---

## Bottom line

| | TmApp | HIC |
|--|-------|-----|
| **Original v1** | `5_LIKELY_RELEVANT` → **MIXED** | `4_PLAUSIBLY_RELEVANT` → **NO_EVIDENCE_IN_CURRENT_DATA** |
| Original limitation | **TECHNICALLY_COMPROMISED_BY_PROPKA_DISULFIDE_SENTINEL** | same |
| **Corrected** | `5_LIKELY_RELEVANT` → **MIXED** | `4_PLAUSIBLY_RELEVANT` → **NO_EVIDENCE_IN_CURRENT_DATA** |

補正後も ESMFold は standalone `REPRODUCIBLE` + incremental `WEAK_OR_MIXED_INCREMENT` → empirical **MIXED**（連続性のためではなく、凍結分類規則の再適用結果）。

---

## 1. 何が間違っていたか

PROPKA 3.5.1 は disulfide Cys に対し `pka_value = 99.99` を置く。これを `ΔpKa = 99.99 − 9.0 = 90.99` として mean/max に入れるのは **物理量の誤解釈**である。

監査: [PKA_SENTINEL_AUDIT_JA.md](PKA_SENTINEL_AUDIT_JA.md) / [PKA_SENTINEL_AUDIT.json](PKA_SENTINEL_AUDIT.json)

- sentinel は抗体あたり典型 2–6 個
- `|ΔpKa|` 総和の **~85–89%** が sentinel 由来
- `max_abs_delta_pKa` 等が全抗体で定数 90.99

## 2. 補正ルール（変更しないもの）

- 同じ 25 canonical feature 名・閾値
- 同じ missing 規則 `ZERO_IF_EMPTY`
- 同じ N+/C- 末端除外
- 非ジスルフィド CYS の通常 pKa は残す
- QC のみ: `disulfide_cys_count`, `disulfide_cys_fraction`（predictor にしない）
- Ridge / alpha / nested CV / residual Ridge / B=10000 は同一

## 3. Robustness: original vs corrected

| Pair | original median ρ | corrected median ρ |
|------|------------------:|-------------------:|
| ESM–ABB2 | 0.449 | 0.365 |
| ESM–Boltz2 | 0.631 | 0.508 |
| ABB–Boltz2 | 0.508 | 0.503 |

Family class: 両方とも **FRAGILE**。

Residue-level ΔpKa（sentinel 除外後） per-ab median Spearman ≈ **0.76 / 0.84 / 0.80**（元の sentinel 込みよりやや低下だが、依然として要約特徴より高い）。

## 4. TmApp corrected（ESMFold primary）

| Split | Standalone MAE (baseline) | ΔMAE residual |
|-------|---------------------------|---------------|
| CV | **3.350** (3.438) | +0.105 |
| Public | **3.680** (3.784) | −0.067 |
| Private | **3.659** (3.772) | −0.028 |

- Standalone: **REPRODUCIBLE**
- Incremental: **WEAK_OR_MIXED_INCREMENT**
- Verdict: **MIXED**
- Generator signal: **ESMFOLD_ONLY_SIGNAL**

ABB2: TEST_ONLY_POSTHOC / NO_INCREMENT → NO_EVIDENCE  
Boltz2: NO_SIGNAL / NO_INCREMENT → UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP（TmApp 規則）

補正後の univariate |ρ| は概して弱い（`mean_abs`≈0.02）。元 v1 で見えた弱い mean_abs 関連の一部は sentinel 汚染由来の可能性がある。

## 5. HIC corrected

全 generator: **NO_SIGNAL / NO_INCREMENT** → **NO_EVIDENCE_IN_CURRENT_DATA**

## 6. 解釈

Sentinel を除いても ESMFold 上の multivariate Ridge は median baseline を三 split で下回り得るが、Round1 residual への安定増分は無く、他 generator でも再現しない。したがって補正後も **MIXED**。ただし original v1 の数値（特に max/mean_abs）は sentinel に支配されており、歴史的結果は **TECHNICALLY_COMPROMISED_BY_PROPKA_DISULFIDE_SENTINEL** として読む。

Residual Ridge audit: **PASS**

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE2C1_PKA_TECHNICAL_CORRECTION_COMPLETE`
