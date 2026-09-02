# Physical Batch 1 Report（Gate 2D–2G）

**Batch state:** `ORGANIZER_FEATURE_PROSPECTING_PHYSICAL_BATCH1_COMPLETE`  
**Anti-posthoc:** 4× `FEATURE_SPEC` を target 前に freeze（`all_family_specs_frozen_before_any_target_score = true`）。

Paper/repo links は各 family 節および `METHOD_REFERENCE_REGISTRY_CURRENT.csv` を参照。

---

## 現時点で何が効いていそうか

### TmApp

| Family | Prior | Empirical verdict | Short interpretation |
|--------|-------|-------------------|----------------------|
| POLAR-SAT（corrected path） | 5 | **PROMISING_BUT_REDUNDANT** | 埋没未充足極性：ESMFold/Boltz で standalone 再現あるが incumbent にほぼ冗長 |
| PKA-SHIFT corrected | 5 | **MIXED** | ESMFoldだけで見えるため保留 |
| VHL-ANGLE | 5 | **PROMISING_BUT_REDUNDANT** | 弱い signal はあるが冗長；3構造ロバストは **PROVISIONAL_TECHNICAL_MAPPING_CONCERN** |
| VOID-EXPLICIT | 5 | **MIXED** | 物理的には妥当だが予測構造依存が強い（FRAGILE） |
| STATIC-SAP（二次） | 3 | MIXED / generator依存 | HIC主だが TmApp 側は ABB2 で偶然 PROMISING — 主結論にしない |
| AROMATIC-TOPO（二次） | 3 | MIXED | HIC主；TmApp は弱い |
| ANM-SPECTRUM | 5 | **NO_EVIDENCE_IN_CURRENT_DATA** | 明確な signal なし |

### HIC

| Family | Prior | Empirical verdict | Short interpretation |
|--------|-------|-------------------|----------------------|
| AROMATIC-TOPO | 5 | **PROMISING_BUT_REDUNDANT** | 露出芳香族トポロジーは standalone 再現あるが Round1 surface にほぼ冗長 |
| STATIC-SAP | 5 | **NO_EVIDENCE_IN_CURRENT_DATA** | prior 5 だが ESMFold では明確な HIC signal なし（静的近似の限界の可能性） |
| PKA-SHIFT corrected | 4 | **NO_EVIDENCE_IN_CURRENT_DATA** | 明確な signal なし |
| POLAR-SAT / VOID / ANM / VHL | ≤3 | NO_EVIDENCE または非主眼 | 交差監査でも HIC 上積みなし |

---

## Bottom line

| Family | Target | Prior | Robustness | Standalone (ESMFold) | Incremental | Verdict (ESMFold primary) |
|--------|--------|-------|------------|----------------------|-------------|---------------------------|
| ANM-SPECTRUM | TmApp | 5 | MODERATE | TEST_ONLY_POSTHOC | CV_ONLY | NO_EVIDENCE |
| VHL-ANGLE | TmApp | 5 | **PROVISIONAL_TECHNICAL_MAPPING_CONCERN** | REPRODUCIBLE | NO_INCREMENT | PROMISING_BUT_REDUNDANT |
| PKA-SHIFT corrected | TmApp | 5 | FRAGILE | REPRODUCIBLE | WEAK_OR_MIXED | **MIXED** |
| PKA-SHIFT corrected | HIC | 4 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE |
| AROMATIC-TOPO | HIC | 5 | MODERATE | REPRODUCIBLE | NO_INCREMENT | **PROMISING_BUT_REDUNDANT** |
| AROMATIC-TOPO | TmApp | 3 | MODERATE | WEAK | NO_INCREMENT | MIXED |
| STATIC-SAP | HIC | 5 | MODERATE | TEST_ONLY_POSTHOC | NO_INCREMENT | **NO_EVIDENCE** |
| STATIC-SAP | TmApp | 3 | MODERATE | WEAK | NO_INCREMENT | MIXED |
| VOID-EXPLICIT | TmApp | 5 | FRAGILE | WEAK | NO_INCREMENT | **MIXED** |
| VOID-EXPLICIT | HIC | 2 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE |
| POLAR-SAT | TmApp | 5 | FRAGILE | REPRODUCIBLE | NO_INCREMENT | **PROMISING_BUT_REDUNDANT** |
| POLAR-SAT | HIC | 3 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE |

Blocked families: **なし**

---

## 1. AROMATIC-TOPO_v1

Paper（representative）: [Jain et al. mAbs 2020](https://doi.org/10.1080/19420862.2020.1743053) — `NO_SINGLE_CANONICAL_AROMATIC_TOPO_ALGORITHM`  
Software: Bio.PDB ShrakeRupley（Round1 再利用）· Repository: N/A

- Robustness: **MODERATE**（ρ≈0.68 / 0.64 / 0.65）
- HIC ESMFold: MAE CV/Pub/Pri **0.504 / 0.492 / 0.466** · standalone **REPRODUCIBLE** · ΔMAE **+0.024 / +0.023 / +0.007** · **NO_INCREMENT**
- Verdict HIC: **PROMISING_BUT_REDUNDANT**
- High-tail: residual candidate ROC≈0.82 だが MAE 増分なし（圧縮は残存）

## 2. STATIC-SAP_v1

**必須表記:** static single-structure approximation inspired by Spatial Aggregation Propensity (SAP)。**canonical dynamic SAP（MD）とは同一ではない。**

Paper: [Chennamsetty et al. PNAS 2009](https://doi.org/10.1073/pnas.0904191106) · Related: [JMB 2009](https://doi.org/10.1016/j.jmb.2009.06.028)  
Scale: Black & Mould 1991（Gly-centered）· R=5/10 Å frozen

Round1 との差: Round1 ADV_SURFACE は Kyte–Doolittle×露出疎水 SASA 局所密度で「SAP 公式ではない」と明記。本 family は Black–Mould×RASA 近傍和。`RELATED_BUT_NOT_IDENTICAL`（HIGH_OVERLAP ではない）。

- Robustness: **MODERATE**（ρ≈0.74 / 0.73 / 0.74）
- HIC ESMFold: **TEST_ONLY_POSTHOC** / NO_INCREMENT → **NO_EVIDENCE_IN_CURRENT_DATA**
- 付記: ABB2 の TmApp が REPRODUCIBLE_INCREMENT（**PROMISING**）だが HIC 主仮説の結論には使わない（generator-dependent）

## 3. VOID-EXPLICIT_v1

Software: [pyKVFinder 0.9.4](https://github.com/LBC-LNBio/pyKVFinder) · Paper: [Guerra et al. 2021](https://doi.org/10.1186/s12859-021-04519-4)

Internality（target-blind）: `max_depth ≥ 2.0 Å` かつ `avg_depth ≥ 0.5 Å` → INTERNAL。浅い pocket は QC のみ。

- Extraction: 324/324×3（ABB2 1件は cavity ゼロ＝成功扱い）
- Robustness: **FRAGILE**（ρ≈0.34 / 0.20 / 0.25）— 構造依存が強い科学的結果
- TmApp ESMFold: WEAK / NO_INCREMENT → **MIXED**
- HIC: NO_EVIDENCE（prior 2 と整合）

## 4. POLAR-SAT_v1

Paper（representative）: [Coventry & Baker 2021](https://doi.org/10.1371/journal.pcbi.1008061)  
Software: [pdb2pqr 3.7.1](https://github.com/Electrostatics/pdb2pqr) `--ff=AMBER` + D–A≤3.5Å / ∠D–H–A≥120°

Round1 との差: Round1 は「埋没極性の CA 近傍に N/O が無い」代理。本 family は明示水素＋幾何 → **METHOD_ADVANCE_OVER_ROUND1**。

- Robustness: **FRAGILE**（ρ≈0.31 / 0.50 / 0.43）
- TmApp ESMFold: MAE **3.235 / 3.757 / 3.432** · **REPRODUCIBLE** · NO_INCREMENT → **PROMISING_BUT_REDUNDANT**
- Boltz2 も REPRODUCIBLE standalone；ABB2 は WEAK
- HIC: NO_EVIDENCE

---

## 5. Cross-endpoint surprises

- STATIC-SAP の **ABB2→TmApp PROMISING** は HIC 主仮説と不一致；構造依存として記録のみ。
- AROMATIC は HIC で相関あり（CV Pearson≈0.42）でも residual 増分なし → Round1 surface との冗長が主因の可能性。

## 6. Generator-dependence concerns

- VOID / POLAR: **FRAGILE**（packing・H-bond は構造敏感）
- VHL: ABB2 ABangle は引き続き provisional
- STATIC-SAP: HIC は弱いが TmApp 増分が ABB2 のみ

## 7. Recommended next（開始しない）

1. **INTERFACE-ENERGY** — VH–VL 界面の安定性（VHL orientation の補完）  
2. **HYDRO-FIELD** / **ELEC-HYDRO-COPATCH** — AROMATIC 冗長の先の連続表面場  
3. **3DI-FROZEN** — 構造トークン埋め込み（物理 family の後）

---

**Final batch state:** `ORGANIZER_FEATURE_PROSPECTING_PHYSICAL_BATCH1_COMPLETE`
