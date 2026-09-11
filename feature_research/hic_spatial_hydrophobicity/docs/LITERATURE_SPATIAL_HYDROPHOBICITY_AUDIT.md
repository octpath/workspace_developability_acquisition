# Literature / implementation audit — spatial hydrophobicity for HIC

**Track:** `feature_research/hic_spatial_hydrophobicity`  
**Date:** 2026-09-11  
**Mainline isolation:** EXP-H102 unused; H054–H101 untouched.

---

## A. Original SAP (Chennamsetty / Voynov / Trout)

**Primary:** Chennamsetty et al., PNAS 2009 (doi:10.1073/pnas.0904191106); related JMB 2009 aggregation motifs.

| Aspect | Canonical SAP | Notes for this track |
|--------|---------------|----------------------|
| Centering | **Atom-centered** local environment | Residue-level summaries derived from atoms |
| Neighborhood | Spatial cutoff around atom | Commonly **5 Å** (and 10 Å variants in literature/practice) |
| Exposure | Side-chain SASA / fully exposed reference | Residue RASA approximations used in static ports |
| Hydrophobicity | **Black & Mould** Gly-centered | Hφ_j − Hφ_Gly |
| Aggregation | Atom → residue → domain; **MD-averaged** | Static single-structure ports drop MD only |
| Score | Σ neighbors RASA×Hφ within R | Positive SAP often used for patches |

**Repository STATIC-SAP (FEATURE_SPEC):** static single-structure approximation — **CA distance**, Black–Mould Gly-centered × total-residue RASA (Tien MaxASA), R∈{5,10}, positive + top-k + CDR aggregates. Explicitly **not** identical to MD SAP.

**N3 / LIT-2 in this track:** `STATIC_CANONICAL_SAP_APPROX` following repository BM×RASA neighborhood semantics (documented deviations from atom-MD SAP).

---

## B. TAP / PSH (Raybould et al., PNAS 2019)

**Primary:** Raybould et al., PNAS 2019 (doi:10.1073/pnas.1810576116).

| Aspect | PSH definition |
|--------|----------------|
| Exposure | Side-chain relative ASA ≥ **7.5%** vs Ala-X-Ala (Shrake–Rupley) |
| Neighborhood | **Closest heavy-atom** distance **< 7.5 Å** |
| Score | Σ_{i<j} H(i)H(j)/r² over exposed pairs |
| H normalization | Scale mapped to **[1, 2]** |
| Scales tested | KD, Wimley–White, Hessa, Eisenberg–McLachlan, Black–Mould (highly correlated); **KD default** |
| Scope | CDR vicinity and full Fv |
| Extra | Salt-bridge residues set to Gly hydrophobicity |

**LIT-3:** PSH-like KD implementation; salt-bridge neutralization **omitted** (flagged).

---

## C. Hydrophobicity scales — verification

| Scale | Status | Source of exact 20-AA table | Direction (after orientation) |
|-------|--------|----------------------------|-------------------------------|
| Kyte–Doolittle | **VERIFIED** | Biopython `kd`; STATIC_SAP_KD | larger = hydrophobic |
| Black–Mould | **VERIFIED** | Biopython `bm`; `structure_utils.BLACK_MOULD_01` | larger = hydrophobic |
| Wimley–White INTERFACE | **VERIFIED** | AAIndex `WIMW960101` (interface→water ΔG) | larger = hydrophobic |
| Eisenberg consensus | **VERIFIED** | Biopython `es` (Eisenberg 1984) | larger = hydrophobic |
| Meek HPLC pH7.4 | **VERIFIED** | AAIndex `MEEJ800101` | larger = hydrophobic |
| Miyazawa | **VERIFIED** | Biopython `mi` | larger = hydrophobic |
| Fauchère–Pliska | **VERIFIED** (extra) | Biopython `fc`; HYDRO_FIELD `FAUCHERE_PI` | larger = hydrophobic |
| Jain HIC | **UNAVAILABLE** | No verified 20-AA “Jain HIC” scale; Jain 2017 is assay data. Janin ≠ Jain (not substituted). | — |

---

## D. Related structural descriptors

| Family | Distinction |
|--------|-------------|
| SAP / STATIC-SAP | Neighborhood × exposure × BM (Gly-centered); MD vs static |
| PSH | Pairwise 1/r² among exposed residues; closest heavy atom |
| Positive-SASA | Σ max(h,0)×SASA (no spatial neighborhood) |
| Direct/total surface score | Σ h×SASA signed |
| HPATCH-like | Patch connectivity of hydrophobic surface (HYDRO-FIELD / gap-closure surface sampling) |
| HYDRO_FIELD | Fauchère–Pliska field on FreeSASA exterior samples (not residue-neighborhood SAP) |
| AROMATIC_TOPO | Exposed F/W/Y topology / SASA / patches |

**H094–H101 STATIC_SAP_KD_v1:** centroid R=5, KD min-max [0,1], total rSASA/Tien, global MAX/MEAN/SUM → **not supported**. This audit treats that as one cell of the design space, not a blanket SAP failure.

---

## E. Side-chain MaxASA reference

No authoritative fully-exposed **side-chain-only** ASA table recovered with provenance sufficient for E4.  
**E4 SIDECHAIN_RASA_REFERENCE = not implemented** (no invented denominator).

---

## F. Structure scope

Canonical **ESMFold Fv** via `STRUCTURE_INPUT_CROSSWALK_v2.csv` `esmfold_canonical_path`. No Fab fabrication.
