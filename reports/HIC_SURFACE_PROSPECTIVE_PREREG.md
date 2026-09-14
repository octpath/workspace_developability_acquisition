# HIC SURFACE Prospective Replication — Preregistration

**STATUS: `HIC_SURFACE_PROSPECTIVE_PREREGISTERED`**

This document freezes the experiment design, success criteria, and embargo
**before any prospective SURFACE training**. After the introducing commit SHA is
recorded, success criteria and design must not be changed without an explicit
prereg amendment commit *before* training.

Do **not** edit:

- `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md`
- `reports/HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md`
- `reports/HIC_SCIENTIFIC_FREEZE_V2.md`
- `reports/HIC_SURFACE_MATCHED_ABLATION_REAUDIT.md`

---

## 0. Critical embargo (this phase)

**Forbidden until a later human-approved external phase:**

- Public / Private / Test Overall labels and MAE
- Loading `top_models_feature_bundle/solution.csv` or any equivalent solution loader
- Writing Public/Private/Test Overall scores into internal reports or registries

**Allowed:** generate and save Test predictions without scoring.

---

## 1. Scientific question

> In two preregistered representation contexts, does adding **antibody-specific**
> canonical F1_SURFACE (35D) information improve HIC prediction relative to an
> architecture- and parameter-matched **SHAM35** auxiliary branch?

This is **not** a new representation / topology / annotation search.

---

## 2. What is prospective vs what is not

| Element | Status |
|---------|--------|
| Representation choice (AbLang1, AbLingua) | Informed by prior factorial internal/external evidence — **not** prospective |
| Topology = JOINT, Annotation = FULL | Fixed from Freeze v2 / factorial — **not** re-searched |
| **SURFACE treatment contrast within fixed contexts** | **Prospective** (SHAM35 vs REAL F1_SURFACE35) |
| HSP P1/P2/P3 | **Excluded** (historically Test-informed family selection) |

---

## 3. Fixed contexts (parents) — verified before training

| Context | Parent | Representation | Topology | Annotation | Context mode |
|---------|--------|----------------|----------|------------|--------------|
| A | **EXP-H187** | AbLang1 (`ablang1`) | JOINT (`topology: B1`, `arch_joint_hl_dual_reg: true`) | FULL (`annotation_mode: full`) | `SEPARATE_CHAIN` |
| B | **EXP-H167** | AbLingua (`ablingua`) | JOINT (same flags) | FULL | `SEPARATE_CHAIN` |

Shared parent protocol (must be inherited by all 4 arms):

- Platform: `DL_FOLDLOCAL_COSINE_V3`
- Seed: **101**
- Frozen PLM embeddings; `pooling_mode: REG`; `merge_mode: mean`
- `share_hl_encoder: true` / `arch_share_hl_encoder: true`
- `d_model: 128`
- Input assets: `#ablang1` / `#ablingua600m` as on parents

**STOP before training** if registry/config disagree with the above.

Why these representations:

- AbLang1 = prospective factorial **external** best-average representation
- AbLingua = internal factorial best-average representation

---

## 4. Arms (4 training conditions only)

| Code slot (issued at train) | Context | Arm | Aux |
|-----------------------------|---------|-----|-----|
| next HIC codes (H340+) | AbLang1 | **S** SHAM35 | 35D zeros for all antibodies |
| | AbLang1 | **R** REAL | Canonical F1_SURFACE 35D (ARO19+HYDRO16) |
| | AbLingua | **S** SHAM35 | 35D zeros |
| | AbLingua | **R** REAL | Canonical F1_SURFACE 35D |

**No Scratch.** No topology/annotation/surface-family/fusion grid. **No Optuna.** **No HSP.**

### SHAM35 requirements

- No HIC labels; no sequence/structure content
- `aux_dim = 35`; same AuxMLP (`late_concat_aux32`), fusion, optimizer, LR grid,
  early stopping, seed, splits, training code as REAL
- Same preprocessing pathway (fold-local median impute + StandardScaler per block)
- Smoke-test that constant features are handled safely by StandardScaler;
  if not, **STOP** and amend prereg before training (do not silently change control)

### REAL F1_SURFACE35

- Canonical asset via `H047AuxFeatureStore("F1_SURFACE")` /
  `experiments/features/EXP-H047.parquet` (ARO‖HYDRO), as audited in
  `F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT`
- **No new surface feature definitions**
- Fold-local TRAIN-only impute + scale

### Primary contrast

`Δ = REAL − SHAM` (negative = SURFACE improves)

Secondary descriptive only: REAL vs historical sequence-only parents H187/H167
(architecture may differ because parents lack aux branch).

---

## 5. Training protocol

- Primary + Shadow schemes under frozen HIC evaluation protocol (`protocol_v3` / `protocol_v3_ext`)
- Identical search protocol for SHAM and REAL (same LR candidate set / patience / epochs / loss)
- Fold-wise selected LR / early-stop epoch may differ as a *result* of training
- Config diff audit: intended YAML diffs limited to experiment identity + aux bundle source

Late fusion: `fusion_mode: late_concat_aux32` (Linear 35→64→32 + concat with `z_DL`).

---

## 6. Internal endpoints (preregistered)

Per representation:

- `ΔCV_primary`, `ΔCV_shadow`, `ΔCV_mean`, `ΔCV_worst`
- change in `|Primary − Shadow|`

Antibody-level paired bootstrap (Primary and Shadow separately):

- Metric: `AE_REAL − AE_SHAM` on OOF predictions
- `N_BOOT = 10000`, `seed = 101`
- Report mean Δ and 95% CI

### Bootstrap aggregation for STRONG criterion (Primary)

1. For each context, form antibody-aligned Primary OOF deltas  
   `d_i = AE_REAL(i) − AE_SHAM(i)`.
2. Let `t_i = 0.5 * (d_i^{AbLang1} + d_i^{AbLingua})` on the antibody intersection.
3. Bootstrap the mean of `{t_i}` (N=10000, seed=101).
4. STRONG requires the **95% CI upper bound of mean(t) < 0**.

Shadow bootstrap is reported diagnostically; STRONG uses **Primary** aggregation above.

---

## 7. Internal verdict rules (frozen)

### `INTERNAL_SURFACE_REPLICATION_STRONG`

All of:

1. AbLang1 `ΔCV_mean < 0`
2. AbLingua `ΔCV_mean < 0`
3. At least **3/4** of scheme-level contrasts improve  
   (AbLang1 Primary, AbLang1 Shadow, AbLingua Primary, AbLingua Shadow)
4. Representation-averaged Primary antibody-level bootstrap 95% CI upper **< 0**
   (aggregation in §6)

### `INTERNAL_SURFACE_REPLICATION_DIRECTIONAL`

- Both representations have `ΔCV_mean < 0`
- But STRONG criteria not fully met

### `INTERNAL_SURFACE_REPLICATION_MIXED`

- Exactly one representation has `ΔCV_mean < 0`

### `INTERNAL_SURFACE_REPLICATION_NOT_SUPPORTED`

- Zero representations with `ΔCV_mean < 0`

Do **not** revise these rules after seeing results.

---

## 8. HIGH-tail diagnostic (not a success criterion)

Threshold fixed: **HIC > 11.5** only (no new thresholds).

On Primary/Shadow OOF: n_high, MAE_high SHAM/REAL, ΔMAE_high, signed errors,
Δsigned, non-HIGH ΔMAE.

Question (diagnostic): does prospective SURFACE addition appear to rescue
HIGH-HIC systematic underprediction?

---

## 9. External success criteria — freeze now, **do not evaluate**

Primary external metric after future unlock: `Test Overall REAL − SHAM`.

### `PROSPECTIVE_SURFACE_EXTERNAL_STRONG`

- AbLang1 Test Δ < 0 and AbLingua Test Δ < 0
- Both contexts: Public Δ < 0 and Private Δ < 0  
  (Test 2/2 and Pub+Priv both improve 2/2)

### `PROSPECTIVE_SURFACE_EXTERNAL_PARTIAL`

- Test 2/2 improve but some Pub/Priv reversal, **or** Test improve only 1/2

### `PROSPECTIVE_SURFACE_EXTERNAL_NOT_SUPPORTED`

- Test improve 0/2

HIGH-tail external rescue is diagnostic only.

---

## 10. Interpretation limit

Even if internal and (later) external are positive, do **not** claim biological
causality of surface physicochemistry for HIC retention.

Maximum allowed strong claim:

> In these preregistered model contexts, explicit antibody-specific surface
> physicochemical information provides reproducible incremental predictive value
> beyond an architecture- and parameter-matched sham auxiliary branch.

---

## 11. Execution order

1. This prereg document
2. **Sole prereg commit** + record SHA
3. Parent config verification (STOP if fail)
4. SHAM35 scaler smoke test (amend prereg if fail)
5. Train 4 models (no external scoring)
6. Internal analysis + internal freeze report
7. Commit/push internal freeze
8. **STOP** (await human approval for external scoring)

---

## 12. Provenance placeholders

| Field | Value |
|-------|--------|
| Prereg introducing commit | *(filled immediately after sole prereg commit)* |
| Parent AbLang1 | EXP-H187 |
| Parent AbLingua | EXP-H167 |
| F1_SURFACE audit | `developability_drilldown/results/F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT.md` |
| Decision basis | Freeze v2 C2 / surface matched reaudit |
