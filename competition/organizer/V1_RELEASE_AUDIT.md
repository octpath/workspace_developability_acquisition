# Antibody Developability — TmApp & HIC
## Version 1.0 Release Audit

Release version:
    1.0

Production split:
    GEN_0001_B_20271100

Core competition data unchanged:
    PASS

Sequence-derived annotations:
    PASS

Annotation provenance:
    PASS

Annotation leakage:
    PASS

Participant fairness:
    PASS

English documentation:
    PASS

Japanese documentation:
    PASS

EN/JA synchronization:
    PASS

Competition rules unchanged:
    PASS

Secret leakage:
    PASS

Extracted-ZIP dry run:
    PASS

Deterministic ZIP:
    PASS

Final archive:
    antibody_developability_competition_v1.0.zip

SHA-256:
    c52aa9425836cdc934b42c7d0ead6d8663908b703e72dbb1b55e098001320e79

Git commit:
    15168b88bff46a450f91ec5dfd4d381ca5d7a1b9

Metadata commit:
    cf144ed1f067a09452d78a2038596f4d74ae6b01

Overall:
    RELEASED_V1_READY


---

## Annotation summary

| Column | Meaning | Type | Derivation | Missing N Dev | Missing N Test | Participant-safe |
|---|---|---|---|---:|---:|---|
| `heavy_v_family` | Inferred heavy V-gene family | categorical | ANARCI IMGT germline V → family | 0 | 0 | Yes |
| `heavy_j_gene` | Inferred heavy J-gene family | categorical | ANARCI J allele → JH# | 0 | 0 | Yes |
| `light_v_family` | Inferred light V-gene family | categorical | ANARCI IMGT germline V → family | 0 | 0 | Yes |
| `light_j_gene` | Inferred light J-gene family | categorical | ANARCI J allele → JK#/JL# | 0 | 0 | Yes |
| `light_chain_type` | kappa or lambda | categorical | ANARCI chain type | 0 | 0 | Yes |
| `h_cdr1_length` | Heavy CDR1 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `h_cdr2_length` | Heavy CDR2 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `h_cdr3_length` | Heavy CDR3 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `l_cdr1_length` | Light CDR1 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `l_cdr2_length` | Light CDR2 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `l_cdr3_length` | Light CDR3 AA length | int | AUTHOR_MMC2_IMGT_SEGMENTS | 0 | 0 | Yes |
| `heavy_germline_identity` | VH vs germline V identity (0–1) | float | ANARCI V identity | 0 | 0 | Yes |
| `light_germline_identity` | VL vs germline V identity (0–1) | float | ANARCI V identity | 0 | 0 | Yes |

### Excluded candidates (high level)

- Allele-level V genes (prefer family)
- Redundant germline distance / mutfrac profiles
- Author ORG_* germline strings
- Full FR/CDR AA substrings
- Donor / B-cell subset / cohort / naive-memory-LLPC
- Targets, Public/Private, organizer model features

### Method / database

- Germline: **ANARCI**, IMGT scheme, human; embedded ANARCI germline references
- CDR lengths: **AUTHOR_MMC2_IMGT_SEGMENTS** (Shehata mmc2 IMGT segments)

### Dry-run MAE (finite)

```json
{
  "median": {
    "TmApp_public_mae": 3.7839506172839505,
    "TmApp_private_mae": 3.771604938271605,
    "HIC_public_mae": 0.5348641975308641,
    "HIC_private_mae": 0.5100864197530864
  },
  "ann_example": {
    "TmApp_public_mae": 3.750126464536894,
    "TmApp_private_mae": 3.5663046170374852,
    "HIC_public_mae": 0.6710692531005753,
    "HIC_private_mae": 0.616187774877679
  }
}
```

## Required questions

1. What annotation columns were found in existing workspace artifacts?  
   Many `PL_*` ANARCI/IMGT fields in `gate_b1/data/numbering_germline.csv`, plus author regions, ORG_*, and forbidden experimental metadata in population/balance caches.

2. Which were included?  
   V/J families, light_chain_type, six CDR lengths, heavy/light germline identity (see table).

3. Which were excluded and why?  
   Allele-level V genes (prefer family); redundant distances/mutfrac; author ORG germline; region strings; donor/B-cell/assay/split/organizer-model features.

4. Are all included fields derivable from VH/VL sequence alone?  
   **YES** (plus documented reference databases / published IMGT segment convention for CDR lengths).

5. Was exactly the same annotation procedure used for Dev and Test?  
   **YES** — frozen 324-row table partitioned by existing Dev/Test membership only.

6. Which germline reference / annotation software was used?  
   **ANARCI** (IMGT, human) via Gate B1 `02c_anarci_germline_full.py`; frozen extract packaged.

7. Which CDR numbering / boundary convention was used?  
   **AUTHOR_MMC2_IMGT_SEGMENTS** (author mmc2 IMGT CDR segment lengths).

8. Are there missing annotations?  
   **No** missing values in v1.0 distributed annotation columns.

9. How are missing annotations encoded?  
   N/A for v1.0 (complete). Policy: empty/NA if needed in future.

10. Are any donor / B-cell-subset / source-cohort columns distributed?  
    **No.**

11. Is any target-derived information distributed?  
    **No.**

12. Is any Public/Private information distributed?  
    **No.**

13. Could participants reproduce these annotations independently in principle?  
    **YES.**

14. Are these annotations clearly documented as optional?  
    **YES** (EN + JA).

15. Do English and Japanese READMEs explain germline / CDR concepts?  
    **YES.**

16. Are existing dev/test/sample_submission CSVs byte-identical to rc3?  
    **YES.**

17. Did the new annotation files pass leakage audit?  
    **YES (PASS).**

18. Did the extracted v1.0 ZIP pass the participant dry-run?  
    **YES (PASS).**

19. Is the ZIP deterministic?  
    **YES** (rebuild byte-identical).

20. Is there any remaining blocker to formal release?  
    **No** (external publish not performed).
