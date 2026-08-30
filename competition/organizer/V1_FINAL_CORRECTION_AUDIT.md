# Antibody Developability — TmApp & HIC
## Definitive Version 1.0 Final Correction Audit

Version:
    1.0

External publication before correction:
    NO

J annotation semantics corrected:
    PASS

heavy_j_gene:
    PASS

light_j_gene:
    PASS

Germline identity wording corrected:
    PASS

CDR source disclosed:
    PASS

CDR length source comparison:
    1944 / 1944 exact

Heavy sequence reconstruction:
    324 / 324

Light sequence reconstruction:
    324 / 324

ANARCI internal consistency:
    PASS

Historical ANARCI version:
    not recoverable as a pinned pip/PyPI version (preserved .venv_b1 has no anarci dist-info; package __version__ string "1.b" is source-only and not treated as a proven historical install pin)

Core CSVs byte-identical:
    PASS

Annotation values unchanged except column rename:
    PASS

EN/JA sync:
    PASS

Annotation leakage:
    PASS

Participant leakage:
    PASS

Extracted-ZIP dry run:
    PASS

Deterministic ZIP:
    PASS

Superseded pre-publication v1.0 SHA-256:
    c52aa9425836cdc934b42c7d0ead6d8663908b703e72dbb1b55e098001320e79

Definitive v1.0 archive:
    antibody_developability_competition_v1.0.zip

Definitive SHA-256:
    36f29a211026dbef3d8e13b0fdada4a88828d13396c3d25e5eb00bdac6032a36

Git commit:
    PENDING

Overall:
    DEFINITIVE_V1_READY


---

## Final annotation schema

| Column | Final meaning | Source/method |
|---|---|---|
| heavy_v_family | Heavy V-gene family | ANARCI |
| heavy_j_gene | Heavy J gene, allele removed | ANARCI |
| light_v_family | Light V-gene family | ANARCI |
| light_j_gene | Light J gene, allele removed | ANARCI |
| light_chain_type | kappa/lambda | ANARCI |
| h_cdr1_length | HCDR1 residue count | mmc2 IMGT segment (gap-stripped) |
| h_cdr2_length | HCDR2 residue count | mmc2 IMGT segment (gap-stripped) |
| h_cdr3_length | HCDR3 residue count | mmc2 IMGT segment (gap-stripped) |
| l_cdr1_length | LCDR1 residue count | mmc2 IMGT segment (gap-stripped) |
| l_cdr2_length | LCDR2 residue count | mmc2 IMGT segment (gap-stripped) |
| l_cdr3_length | LCDR3 residue count | mmc2 IMGT segment (gap-stripped) |
| heavy_germline_identity | VH vs assigned germline V identity (0–1; higher = more similar) | ANARCI |
| light_germline_identity | VL vs assigned germline V identity (0–1; higher = more similar) | ANARCI |

## Corrections applied

1. Renamed `heavy_j_family`/`light_j_family` → `heavy_j_gene`/`light_j_gene` (allele-stripped gene stems).
2. Clarified germline identity as **similarity** (not distance) in EN/JA READMEs and DATA_DICTIONARY.
3. Disclosed CDR lengths come from Shehata mmc2 IMGT-segmented regions (gap-stripped).
4. Verified all 1944 CDR lengths and 324/324 sequence reconstructions against mmc2.
5. ANARCI historical pip version remains unrecoverable; frozen annotations unchanged in values.

## Dry-run MAE (finite)

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

## Required confirmations

- J family → J gene corrected: **YES**
- Germline identity wording corrected: **YES**
- CDR audit exact matches: **1944 / 1944**
- Sequence reconstruction: heavy **324/324**, light **324/324**
- Core CSV immutability: **PASS**
- Annotation leakage: **PASS**
- EN/JA sync: **PASS**
- Remaining blockers: **none**
