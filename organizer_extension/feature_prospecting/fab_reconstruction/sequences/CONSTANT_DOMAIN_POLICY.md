# CONSTANT DOMAIN POLICY (FROZEN)

**CONSTANT_DOMAIN_POLICY_FROZEN_BEFORE_STRUCTURE_PREDICTION = true**  
**Freeze date:** 2026-09-05  
**Reconstruction class:** `RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB`

## Decision

Gate R0 establishes recombinant **IgG1** (HIGH) and papain Fab production (HIGH), but **not** exact CH1 / Cκ / Cλ alleles or papain C-terminus.

Therefore:

- **POLICY B** — reconstruct experimental-like Fab using authoritative canonical constant sequences.
- Do **not** claim `EXACT_EXPERIMENTAL_FAB`.
- Do **not** choose constants using TmApp/HIC (target-blind).

## Sequences used

| Role | Sequence ID | Source | Accession | Notes |
|------|-------------|--------|-----------|-------|
| Heavy constant | `IGHG1_CH1_EPKSC_UNIPROT_P01857` | UniProt IGHG1 | P01857 | CH1 domain + `EPKSC` (through Cys220). Exact papain hinge length **unknown**. |
| C-kappa | `IGKC_UNIPROT_P01834` | UniProt IGKC | P01834 | Full Cκ including C-terminal Cys. |
| C-lambda | `IGLC2_UNIPROT_P0DOY2` | UniProt IGLC2 | P0DOY2 | **Single** predeclared Cλ for all λ Abs; cognate IGLC gene unknown. |

## Light-chain assignment rule

1. Prefer project `ORG_kappa_lambda` / `PL_kappa_lambda` from `numbering_germline.csv`.
2. Else gene prefix: IGKV→κ, IGLV→λ.
3. Unresolved → flag; do not invent.

## Explicit uncertainties

- Exact IGHG1 allele / CH1 SNPs
- Exact IGKC / IGLC alleles
- Cloning junctions / scars
- Papain cleavage heterogeneity / hinge residues beyond/short of `EPKSC`
- Whether yeast vectors used engineered constant interfaces (orthogonal CH1 patents exist at Adimab but not tied to this cohort)

## Exact AA strings

See `CONSTANT_DOMAIN_SEQUENCES.fasta` and `CONSTANT_DOMAIN_PROVENANCE.csv`.
