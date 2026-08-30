# Structure prediction audit

## ABodyBuilder2 (ImmuneBuilder)

- Success: 400/400 (100%)
- Failures: 0
- All 400 paired Fv structures cached under `cache/structures/abodybuilder2/`
- Scheme: IMGT numbering via ANARCI

## ESMFold (`facebook/esmfold_v1`)

- Success: 400/400 (100%)
- Mean pLDDT: 0.7093333982940792
- Runtime: 2057.7s
- **Multichain handling:** HF `EsmForProteinFolding` is not a native multichain decoder.
  Used VH + 25×Gly linker + VL; linker residues excluded from PDB and SASA/RASA.
- Validation note: HF EsmForProteinFolding folds a single polymer with Gly linker; chain boundaries preserved by index bookkeeping, not by native multichain decoder. Linker residues excluded from feature extraction.

## Comparison note

ABB2 and ESMFold structural features are roughly equivalent for HIC ranking (~0.50 CV). Prefer ABB2 for antibody-specialized prior; keep ESMFold as documented comparison branch.
