# SOURCE SPEC AUDIT — SAP24 / SCM24

**Track:** `feature_research/hic_sap_scm_source/`  
**Date:** 2026-09-12  
**Mainline next HIC:** `EXP-H114` (must remain unused)

## Fidelity gate summary

| Block | Status | Reason |
|-------|--------|--------|
| **SOURCE_SAP24** | **BLOCKED_SOURCE_UNRESOLVED** | `positive_sum_mean` mathematical definition not found in any authoritative artifact |
| **SOURCE_SCM24** | **BLOCKED_SOURCE_UNRESOLVED** | SCM / charge per-residue property + sign convention unresolved |

Per track rules: **do not invent** unresolved formulas; **do not** label any 24-D block as source-faithful until resolved.

---

## A. SAP per-residue local score

| Parameter | Status | Evidence |
|-----------|--------|----------|
| Expected form (user brief) | SOURCE_CONFIRMED (structure) | User §6: `SAP_i(R)=Σ_j I[d(centroid_i,centroid_j)≤R]·KD_j·RASA_j` |
| Side-chain centroid neighborhood | EXISTING_REPO_CONFIRMED | `static_sap_kd.py` / H094 prereg; HSP geometry uses SC centroid |
| CA neighborhood (alternate) | EXISTING_REPO_CONFIRMED (different family) | Organizer `STATIC-SAP/FEATURE_SPEC.json` uses **CA** + Black–Mould |
| Self-neighbor inclusion | EXISTING_REPO_CONFIRMED | STATIC_SAP_KD / HSP: self included (`d_ii=0`) |
| Radii {5,10} Å | SOURCE_CONFIRMED | User §1; also Chennamsetty/STATIC-SAP common cutoffs |
| No distance decay | EXISTING_REPO_CONFIRMED | STATIC_SAP_KD / HSP |
| Exact “source SAP24” document in repo | **UNRESOLVED** | No file named SAP24 / SOURCE_SAP24 / 24-dim KD×VH/VL/CDR/FR |

**Repo near-misses (not SAP24):**

1. **STATIC_SAP_KD** (`developability_drilldown/models/antibody_transformer/static_sap_kd.py`):  
   `SSKD_i(R)=Σ I[d(centroid)≤R]·KD_norm·clip(SASA/Tien,0,1)`; Ab aggs MAX/MEAN/SUM on ALL/H/L only (3 or 9 dims).
2. **Organizer STATIC-SAP_v1** (`organizer_extension/.../STATIC-SAP/FEATURE_SPEC.json`):  
   Black–Mould Gly-centered × RASA, **CA** neighborhood, R∈{5,10}, 18 canonical features — not KD, not VH/VL/FR×3-stat=24.

---

## B. `positive_sum_mean` — CRITICAL

| Status | **UNRESOLVED** |
|--------|----------------|

- String search `positive_sum_mean` / `POSITIVE_SUM_MEAN` / `pos_sum_mean`: **zero hits** in repository (excluding this audit).
- User brief **names** the statistic but explicitly forbids guessing among:
  - mean over positive residues
  - positive sum / region length
  - positive sum / positive count
  - other

**Near-miss definitions (NOT adopted as source):**

| Name | Formula | Source |
|------|---------|--------|
| `mean_positive_static_SAP` | `mean(SAP_i \| SAP_i>0)` else 0 | `extract_physical_batch1.py` |
| `sum_positive_static_SAP` | `sum(SAP_i \| SAP_i>0)` else 0 | same |
| DeepSP SAP_score (literature) | `\|Σ_domain SAP_i · H(SAP_i)\|` = sum of positive SAP in domain | Kalejaye et al. CSBJ 2024 Eq. (5) |

DeepSP is **sum of positives** (Heaviside), **not** a named `positive_sum_mean`, and uses MD-averaged atomic SAP + different region set (CDRH1–3, CDRL1–3, CDR, Hv, Lv, Fv) → **30** DeepSP descriptors, not 24.

**Gate consequence:** cannot construct a block titled `SOURCE_SAP24` / `FS_HIC_SOURCE_SAP24`.

---

## C–D. SCM / charge property + sign convention

| Status | **UNRESOLVED** → SCM24 **BLOCKED** |
|--------|-------------------------------------|

- No `SCM24` / spatial-charge-map feature family in organizer registry.
- DeepSP predicts sequence-surrogate `SCM_pos` / `SCM_neg` (CNN), not a reproducible static Fv charge formula in this repo.
- Charge near-misses (not SCM): Stage3 `rasa_w_charge_sum` (D/E −1, K/R +1, H 0.1); Stage4 PROPKA/PQR charges.

**Do not assume** K/R=+1, D/E=−1, H=0, PROPKA, or partial charges.

SAP work is allowed to continue once `positive_sum_mean` is resolved; SCM remains independently blocked until charge semantics are source-confirmed.

---

## E. Shrake–Rupley `n_points`

| Status | EXISTING_REPO_CONFIRMED = **100** |
|--------|-----------------------------------|

- `STATIC-SAP/FEATURE_SPEC.json`: probe 1.4, `n_points` 100  
- `structure_utils.py` / H094 STATIC_SAP_KD / HSP geometry cache: same  
- `n_points=960` appears only as numerical QC, not as primary definition

---

## F. KD raw vs transformed

| Status | EXISTING_REPO_CONFIRMED for repo KD-SAP = **min-max [0,1]** |
|--------|--------------------------------------------------------------|

- H094 prereg / `kd_norm()`: `(KD−min)/(max−min)` over 20 AA; KD Ile=4.5, Arg=−4.5  
- User brief §7: “If the prior min-max definition is confirmed” → confirmed for **this repository’s KD-SAP lineage**  
- Organizer STATIC-SAP uses **Black–Mould**, not KD  
- External DeepSP MD SAP uses classical SAP hydrophobicity (Black–Mould lineage), not this KD table

**SOURCE_CONFIRMED for a dedicated SAP24 KD transform document:** **UNRESOLVED** (no SAP24-specific source file); best repo prior for KD×SAP is min-max.

---

## G. RASA / Tien MaxASA

| Status | EXISTING_REPO_CONFIRMED |
|--------|-------------------------|

- STATIC_SAP_KD / HSP: `clip(SASA_j / Tien_MaxASA_j, 0, 1)`  
- Organizer STATIC-SAP: Stage4 `MAX_ASA` table (related Tien-style), probe 1.4  

---

## H. Regions VH / VL / CDR / FR

| Status | SOURCE_CONFIRMED (names) / EXISTING_REPO_CONFIRMED (masks available) |
|--------|----------------------------------------------------------------------|

- User §5: overlapping axes VH/VL and CDR/FR  
- Repo has IMGT CDR / framework via CDR maps (`is_cdr`, chain H/L) in HSP geometry and structure utilities  
- Exact FR = all non-CDR H+L residues: implementable once fidelity gate opens  

---

## I. Statistics MAX / TOP5_MEAN

| Statistic | Status |
|-----------|--------|
| MAX | EXISTING_REPO_CONFIRMED (`spatial_engine.aggregate`, STATIC-SAP max_positive / max) |
| TOP5_MEAN | EXISTING_REPO_CONFIRMED (mean of up to 5 largest scores; fewer than 5 → use all) |
| POSITIVE_SUM_MEAN | **UNRESOLVED** (blocking) |

---

## Parameter checklist (compact)

| Item | Mark |
|------|------|
| SAP local score skeleton | SOURCE_CONFIRMED (user) + EXISTING_REPO_CONFIRMED (KD-SAP centroid form) |
| KD table | EXISTING_REPO_CONFIRMED (Kyte–Doolittle 1982) |
| KD transform | EXISTING_REPO_CONFIRMED (min-max) for repo KD-SAP |
| RASA / Tien | EXISTING_REPO_CONFIRMED |
| probe / n_points | EXISTING_REPO_CONFIRMED (1.4 / 100) |
| centroid vs CA | EXISTING_REPO_CONFIRMED for KD-SAP (centroid); alternate CA for BM STATIC-SAP |
| self-neighbor | EXISTING_REPO_CONFIRMED |
| R ∈ {5,10} | SOURCE_CONFIRMED |
| MAX | EXISTING_REPO_CONFIRMED |
| TOP5_MEAN | EXISTING_REPO_CONFIRMED |
| **POSITIVE_SUM_MEAN** | **UNRESOLVED** |
| VH/VL/CDR/FR axes | SOURCE_CONFIRMED names |
| SCM local score | **UNRESOLVED** |
| SCM sign/magnitude | **UNRESOLVED** |

---

## What remains unknown (must be supplied before SOURCE_* generation)

1. Exact equation for **`positive_sum_mean`** (including denominator and empty-region rule).  
2. Exact **SCM** per-residue property (atomic vs residue; charge source; pH; Heaviside for pos/neg).  
3. Whether source neighborhood is **side-chain centroid** (KD-SAP) or **CA** (organizer STATIC-SAP).  
4. Whether source hydrophobicity for this SAP24 is **KD min-max** or Black–Mould (DeepSP/Chennamsetty lineage).

Until (1) is resolved, **no SOURCE_SAP24 features, Stage-1 screen, or mainline promotion** under the source-faithful label.
