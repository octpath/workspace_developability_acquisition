# Numbering and germline audit

- Frozen CDR scheme: **AUTHOR_MMC2_IMGT_SEGMENTS**
- H concat exact match to cleaned VH: 100.0%
- L concat exact match to cleaned VL: 100.0%
- anarcii H/L ok: 100.0% / 100.0%
- CDR-H3 length: {'min': 7, 'max': 33, 'median': 15.0}
- kappa/lambda: {'kappa': 275, 'lambda': 125}

CDR regions frozen from author mmc2 IMGT-segmented FR/CDR columns. Germline distance is Hamming fraction vs Naïve-within-family V-region consensus (FR1-CDR1-FR2-CDR2-FR3); CDR3 excluded from germline-distance numerator. Not claimed as exact SHM count.

## Feature classes

- `PL_*`: participant-legal sequence-derived
- `ORG_*`: organizer-only author annotations


## ANARCI germline re-annotation

- anarci_H_ok: 1.0
- anarci_L_ok: 1.0
- family_source: {'author_fallback': 400}
- family_agree_author_H: 0.0
- family_agree_author_L: 0.0

## ANARCI allele-level germline assignment

- H_ok: 1.0
- L_ok: 1.0
- family_source: {'anarci': 400}
- family_agree_author_H: 1.0
- family_agree_author_L: 1.0
- mean_vh_germline_distance: 0.10294477162790969
- mean_vl_germline_distance: 0.07027694468405313

Germline distance = 1 - ANARCI V-allele identity. Mutation fraction uses IMGT positions 1–104 only (CDR3 excluded).
