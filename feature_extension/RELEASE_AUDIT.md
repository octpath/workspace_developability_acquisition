# RELEASE_AUDIT — feature_extension v1 (final pre-release)

## ready for participant release: YES

Conditional on organizer confirmation items listed under licensing / unresolved issues.
Scientific feature values were **not** regenerated in this audit (metadata/docs/packaging only).

---

## Surface representation distinction

| Product | Method (authoritative) | Representation |
|---|---|---|
| `continuous_surface.parquet` | Gap Closure Task B / `extract_surface_hic.py` / `STRUCTURE_GAP_CLOSURE_SPEC.md`: FreeSASA **Lee–Richards** atom SASA (probe 1.4 Å) → exterior Fibonacci **SAS sample points** → hydrophobic masks (KD/FP/BM) → connected components on sample points (link 2.0 Å) with area/perimeter/compactness | Continuous **SAS-sampled molecular surface** (not MSMS SES triangulation; not residue-CA graph) |
| `extractors/extract_surface_patch.py` | Bio.PDB Shrake–Rupley residue SASA → exposed hydrophobic residues → CA–CA ≤ 8 Å components | **Residue-adjacency graph** only |

**Verdict:** They are **not** the same and **not** numerically equivalent.
The extractor does **not** reproduce the precomputed block.
Documented in top-level README, `data/precomputed_features/README.md`, `extractors/README.md`.
Block filenames were **not** renamed (avoid breaking published paths); documentation clarifies.

---

## BioEmu dictionary status

- `data/bioemu_isolated/FEATURE_DICTIONARY.csv` — **DONE** (65 rows)
- Families recovered from `eval_reassess.py` + `analyze_convergence.py`:
  - **NEW_PAIRWISE** (10): `*ca_rmsd*`
  - **NEW_CONTACT** (20): `*contact*`
  - **NEW_FLEX** (25): `*ca_rmsf*` / `*rmsf_*`
  - **NEW_SHAPE** (10): `*rg_*`
  - **NEW_COMBINED**: all 65 columns
- Participant snippet in `data/bioemu_isolated/README.md`
- Target used: **NO**; VH/VL sampled independently; frozen Nphys=8

---

## Block coverage audit

See `BLOCK_COVERAGE.csv` (expected universe = 324 crosswalk IDs).

| Issue | Detail |
|---|---|
| Incomplete IDs | `buried_unsatisfied` missing **ADI-47265** only |
| Partial values | `continuous_surface` Fab-prep columns NaN for ADI-47265 (90 cells) |
| All other core tables / PDB dirs | 324/324, no duplicates, no ±inf |

Incomplete coverage is **allowed** and documented; join guidance = LEFT JOIN.

---

## Licensing audit

See expanded table in `RELEASE_NOTES.md`.

| Artifact class | Decision |
|---|---|
| ProteinMPNN / SaProt tables | `CLEAR_FOR_RELEASE` |
| Geometry descriptor tables | `CLEAR_FOR_RELEASE` |
| ESMFold PDBs / ESM-IF1 / BioEmu features | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` + `PARTICIPANT_ONLY_REVIEW_RECOMMENDED` |
| FreeSASA continuous_surface features | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` (numeric only) |

Not legal certainty. No concrete output ban identified; no automatic asset removal.

---

## Clean-unpack Quick Start result

**CLEAN_UNPACK_QUICKSTART = PASS**

Procedure: temp dir outside repo → extract code + esmfold_fv + bioemu + precomputed ZIPs →
`PYTHONPATH` parent → import extractors → load BioEmu/MPNN → LEFT JOIN mock frame →
load `folds.csv` → run `example_simple_tvt_cv.py` with synthetic local targets (not shipped).

---

## validate_release / smoke-test result

- `tools/validate_release.py` → **PASS**
- `tests/test_smoke.py` → **PASS**

---

## Archive rebuild

ZIPs regenerated after documentation/metadata changes; `SHA256SUMS.txt` refreshed.

| Archive | Approx size |
|---|---|
| code | ~34 KB |
| esmfold_fv | ~12 MB |
| esmfold_fab | ~21 MB |
| bioemu_isolated | ~129 KB |
| precomputed_features | ~3.3 MB |

---

## Unresolved issues for organizer judgment before public release

1. Confirm competition/public hosting policy for ESMFold PDBs and BioEmu-derived feature tables
   (`PARTICIPANT_ONLY_REVIEW_RECOMMENDED`).
2. Acknowledge ADI-47265 incompleteness in buried-unsatisfied / Fab-prep continuous-surface cells.
3. FeNNix remains deferred (out of scope).

---

## Known limitations (unchanged scientifically)

- FeNNix not packaged
- Molecular scope ≠ assay molecule
- Fab surrogate constants
- ADI-47265 incomplete in some Fab-physics blocks
