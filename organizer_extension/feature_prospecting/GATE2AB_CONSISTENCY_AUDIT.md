# Gate2A/B Consistency Audit (pre–Gate2C)

**State:** non-tuning audit only  
**Purpose:** verify common evaluation infrastructure and VHL ABB2 technical status before PKA-SHIFT_v1 target scoring.  
**Does not:** change ANM/VHL features, re-optimize models, or rewrite historical family reports.

---

## 0. Bottom line

| Check | Result |
|-------|--------|
| Common nested Ridge / TRAIN_MEDIAN_BASELINE / residual Ridge | **PASS** |
| Standalone class re-derivation vs SIGNAL_CLASSIFICATION_SPEC | **PASS** (matches saved labels) |
| VHL ESMFold REPRODUCIBLE | **CONFIRMED** under frozen criteria |
| VHL ABB2 ABangle geometry | **TECHNICAL_MAPPING_CONCERN** → register `VHL-ANGLE_v1_TECHNICAL_REVIEW` |
| Blocking for PKA target evaluation? | **NO** (issue is VHL/ABangle-specific) |

---

## 1. Re-derived TRAIN_MEDIAN_BASELINE + correlations

Classifier used: `organizer_extension/feature_prospecting/common/ridge_eval.py::classify_standalone`  
(matches SIGNAL_CLASSIFICATION_SPEC §4.2 operationally).

### 1.1 ANM-SPECTRUM (saved = re-derived)

| Target | Generator | Standalone (saved=rederived) |
|--------|-----------|------------------------------|
| TmApp | esmfold | TEST_ONLY_POSTHOC |
| TmApp | abodybuilder2 | CV_ONLY |
| TmApp | boltz2 | TEST_ONLY_POSTHOC |
| HIC | all three | NO_SIGNAL |

ESMFold TmApp example:

| Split | MAE | baseline | beat? | Pearson | Spearman |
|-------|-----|----------|-------|---------|----------|
| CV | 3.545 | 3.438 | no | 0.077 | 0.122 |
| Public | 3.719 | 3.784 | yes | 0.047 | 0.155 |
| Private | 3.775 | 3.772 | no* | −0.060 | 0.083 |

\*Private MAE slightly worse than baseline → not REPRODUCIBLE; class TEST_ONLY_POSTHOC is consistent.

### 1.2 VHL-ANGLE ESMFold TmApp (reported REPRODUCIBLE)

| Split | MAE | baseline | beat? | Pearson | Spearman |
|-------|-----|----------|-------|---------|----------|
| CV | **3.335** | 3.438 | yes | 0.218 | 0.223 |
| Public | **3.593** | 3.784 | yes | 0.216 | 0.255 |
| Private | **3.741** | 3.772 | yes | 0.196 | 0.276 |

All three splits: MAE < TRAIN_MEDIAN_BASELINE, positive Pearson/Spearman, \|r\| ≥ 0.15 → **REPRODUCIBLE confirmed**.  
**No HISTORICAL_CLASSIFICATION_MISMATCH.**

Other VHL classes also re-derived identically (ABB2 REPRODUCIBLE; Boltz2 WEAK; HIC NO_SIGNAL×3).

---

## 2. VHL ABB2 technical audit (target-blind)

### 2.1 What is OK

- H/L (or A/B) **sequence identity** across ESMFold / ABB2 / Boltz2: exact match on audited IDs (lengths and AA sequences).
- ABangle coreset atom **count** after renumbering: 35/35 H and L for all three generators on `ADI-37123`.
- ESMFold and Boltz2 ABangle ranges: `dc` median ≈ **16.0 Å** (literature-typical packing separation); angle spreads tight.

### 2.2 What is concerning

ABB2 ABangle parameter ranges are physically implausible vs ESM/Boltz:

| Param | ESMFold med | ABB2 med | Boltz2 med |
|-------|-------------|----------|------------|
| dc | 15.99 | **6.79** | 16.00 |
| HL | −58.7 | −65.5 (p5..p95 ≈ −154..+129) | −60.3 |

Pairwise median feature Spearman (frozen): ESM–ABB2 **0.018**, ESM–Boltz **0.558**, ABB2–Boltz **0.035** → FRAGILE driven by ABB2.

### 2.3 Root cause (mapping / numbering)

ABB2 PDBs are already **Chothia-like** with residue-number **gaps** (non-consecutive resseq).  
`Bio.PDB` `AtomIterator` used by vendored ABangle inserts **`X` placeholders** for missing residue numbers:

- Example `ADI-37123` H: `n_res=119` but `seq_len=128` (`QVQLQESGGXAVVPPGRSLR...`)
- ANARCI then returns inflated span/`n_num` (128 vs 119)
- After `renumber_structure`: **duplicate resseq** (e.g. `…,6,6,7…`) and **truncated C-terminal Chothia numbers** (H ends ~104 instead of ~113; L ends ~91 instead of ~107)

ESMFold (A/B sequential) and Boltz2 (H/L sequential) do not have these gaps → renumbering is clean → `dc≈16`.

Framework RMSD QC in v1 used author/Chothia resseq overlap before consistent sequence-index alignment → unstable / often NaN; not evidence of structure dissimilarity alone.

### 2.4 Technical status

| Scope | Status |
|-------|--------|
| ESMFold / Boltz2 ABangle path | **TECHNICALLY_VALID** |
| ABB2 ABangle path under current vendored numbering | **TECHNICAL_MAPPING_CONCERN** (geometry values not trustworthy for cross-generator comparison) |
| Family-level note | Register **`VHL-ANGLE_v1_TECHNICAL_REVIEW`** for later (do **not** fix in place) |

This is **not** treated as a common Ridge/baseline/split bug.

---

## 3. Common evaluation infrastructure

Residual Ridge audit artifacts from Gate2A/B remain **PASS**.  
Re-derivation of standalone classes from saved metrics matches reported labels.  
No evidence of wrong split assignment or TRAIN_MEDIAN_BASELINE formula error in shared `ridge_eval.py`.

**Blocking rule:** common-infra bug → STOP before PKA.  
**Decision:** **continue PKA-SHIFT_v1**.

---

## 4. Follow-ups (not executed now)

1. `VHL-ANGLE_v1_TECHNICAL_REVIEW` — ABB2 pre-Chothia / gap-aware sequence extraction before ABangle.  
2. Optional: framework RMSD QC on sequence-index-aligned backbone (not resseq).  
3. Do not revise Gate2B REPORT_JA historical text; ABB2 VHL scores remain frozen historical with this audit caveat.
