# Annotation leakage audit

Competition: Antibody Developability — TmApp & HIC  
Package: **1.0**  
Files audited: `dev_annotations.csv`, `test_annotations.csv`

Principle: each distributed annotation must be derivable from sequence + documented reference annotation methods, independent of assay labels and split membership.

| annotation | sequence-derived | target-independent | same Dev/Test procedure | participant-safe | result |
|---|---|---|---|---|---|
| `id` | Join key (not a bio feature) | Yes | Yes | Yes | PASS |
| `heavy_v_family` | Yes (ANARCI on VH) | Yes | Yes | Yes | PASS |
| `heavy_j_family` | Yes (ANARCI on VH) | Yes | Yes | Yes | PASS |
| `light_v_family` | Yes (ANARCI on VL) | Yes | Yes | Yes | PASS |
| `light_j_family` | Yes (ANARCI on VL) | Yes | Yes | Yes | PASS |
| `light_chain_type` | Yes (ANARCI κ/λ) | Yes | Yes | Yes | PASS |
| `h_cdr1_length` | Yes (IMGT CDR segment length of VH) | Yes | Yes | Yes | PASS |
| `h_cdr2_length` | Yes | Yes | Yes | Yes | PASS |
| `h_cdr3_length` | Yes | Yes | Yes | Yes | PASS |
| `l_cdr1_length` | Yes (IMGT CDR segment length of VL) | Yes | Yes | Yes | PASS |
| `l_cdr2_length` | Yes | Yes | Yes | Yes | PASS |
| `l_cdr3_length` | Yes | Yes | Yes | Yes | PASS |
| `heavy_germline_identity` | Yes (ANARCI V identity) | Yes | Yes | Yes | PASS |
| `light_germline_identity` | Yes (ANARCI V identity) | Yes | Yes | Yes | PASS |

## Explicit non-distribution checks

| Forbidden content | Present in annotation CSVs? |
|---|---|
| `TmApp` / `HIC` columns | No |
| `is_public` / `is_private` | No |
| Public/Private ID lists | No |
| donor | No |
| B-cell subset / origin / naive / memory / LLPC | No |
| experimental cohort / affinity-maturation category | No |
| organizer model predictions / embeddings | No |
| split-selection metrics | No |

## Target-leakage sanity (provenance)

Annotations can be recomputed without access to TmApp or HIC. Correlation with targets (if any) does **not** constitute leakage under this audit.

## Participant fairness

Could a participant obtain equivalent annotations from provided sequences using documented public/reference methods?

- Germline families / κλ / identity: **YES** via ANARCI (or equivalent germline assigners) + IMGT.  
- CDR lengths: **YES** via published mmc2 IMGT segments for these sequences and/or independent IMGT/ANARCI region length computation (conventions may differ slightly).

**Overall annotation leakage: PASS**
