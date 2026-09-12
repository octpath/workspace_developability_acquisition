# SOURCE SPEC AUDIT — SAP24 / SCM24 (UPDATED after fidelity unlock)

**Track:** `feature_research/hic_sap_scm_source/`  
**Date:** 2026-09-12  
**Prior blocked commit:** `83dfed7d` (auditable)  
**Mainline:** EXP-H114 unused

## Fidelity gate (updated)

| Block | Status |
|-------|--------|
| SOURCE_SAP24 | **ACTIVE** — `positive_sum_mean` SOURCE_CONFIRMED |
| SOURCE_SCM24 | **ACTIVE** — charge SOURCE_CONFIRMED; geometry/exposure SOURCE_DERIVED_INFERENCE |

---

## Provenance labels

### SOURCE_CONFIRMED

| Item | Definition |
|------|------------|
| SAP property | Kyte–Doolittle |
| KD transform | min-max over 20 AA → [0,1] |
| RASA denominator | Tien MaxASA |
| RASA clip | [0,1] |
| Radii | {5 Å, 10 Å} |
| Regions | {VH, VL, CDR, FR} overlapping axes |
| Statistics | max, top5_mean, positive_sum_mean |
| positive_sum_mean | `(1/N) Σ max(local_i, 0)` over **all** valid residues N |
| SCM charge | R,K=+1; D,E=−1; else 0 (no H/pH/PROPKA) |

### EXISTING_REPO_CONFIRMED

| Item | Value |
|------|-------|
| Shrake–Rupley probe | 1.4 Å |
| n_points | 100 |
| Neighborhood | side-chain centroid (Gly→CA) |
| Self-neighbor | included (`d_ii=0`) |
| Structures | canonical ESMFold Fv (`residue_geometry_sasa.parquet`) |

### SOURCE_DERIVED_INFERENCE

| Item | Assumption |
|------|------------|
| SCM exposure | same Tien-RASAclip weighting as SAP |
| SCM neighborhood | same side-chain-centroid + radii + region/stat aggregation as SAP |

If a future source contradicts these, version and recompute SCM.

---

## Formulas

### SAP local

`SAP_i(R) = Σ_j I[d(c_i,c_j)≤R] · KDnorm(AA_j) · RASAclip_j`

### SCM local (signed)

`SCM_i(R) = Σ_j I[d(c_i,c_j)≤R] · charge(AA_j) · RASAclip_j`

### Antibody stats (per region × radius)

- MAX = max(local)
- TOP5_MEAN = mean of top min(5,N) **signed** values
- POSITIVE_SUM_MEAN = mean(max(local,0)) over all N

---

## Previously unresolved (now resolved)

1. **positive_sum_mean** — confirmed as mean of clipped-at-zero locals over all N  
2. **SCM charge** — formal ±1 / 0 table confirmed
