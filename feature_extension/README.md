# feature_extension (v1)

Participant-facing package of **target-blind** antibody structures and lightweight /
precomputed features for the developability competition.

This is a **release package**, not an organizer research dump.

## 1. What is feature_extension?

A curated set of:

- predicted **Fv** and reconstructed **Fab** structures (ESMFold)
- aggregated **BioEmu** descriptors for isolated VH/VL ensembles
- precomputed model/descriptor feature tables (ProteinMPNN, ESM-IF1, SaProt, …)
- portable PDB extractors (SASA, aromatic exposure, …)
- organizer-recommended **5-fold** assignments for training/dev IDs

**No competition target labels** (TmApp / HIC), Public/Private flags, or organizer
OOF predictions are included under `data/`.

## 2. Quick start

```bash
pip install -r requirements.txt

# inspect folds (162 training/dev IDs)
head folds.csv

# extract features from one bundled Fv PDB
python examples/example_extract_features.py

# join your local train.csv with precomputed blocks
python examples/example_join_precomputed.py --train-csv /path/to/train.csv

# Simple TVT Ridge demo (requires your local labels)
python examples/example_simple_tvt_cv.py \
  --train-csv /path/to/train.csv --target TmApp
```

Place this directory on `PYTHONPATH`, or run examples from a checkout where
`feature_extension/` is importable as a top-level package.

## 3. Directory layout

```
feature_extension/
├── README.md
├── RELEASE_NOTES.md
├── MANIFEST.csv
├── folds.csv
├── requirements.txt
├── data/
│   ├── esmfold_fv/
│   ├── esmfold_fab/
│   ├── bioemu_isolated/
│   ├── precomputed_features/
│   └── optional/
├── extractors/
├── examples/
├── tests/
└── tools/
```

## 4. Available structure datasets

| Dataset | Scope | N | Notes |
|---|---|---:|---|
| `data/esmfold_fv/` | **Fv** | 324 | Predicted Fv; pLDDT often in B-factor |
| `data/esmfold_fab/` | **Fab** | 324 | Reconstructed Fab (variable + surrogate constants) |

See each subdirectory README. Fab constants: POLICY B (UniProt CH1 / Cκ / Cλ);
files `CONSTANT_DOMAIN_POLICY.md` and `CONSTANT_DOMAIN_SEQUENCES.fasta` are included.

## 5. Available precomputed feature blocks

See `data/precomputed_features/README.md` and `MANIFEST.csv`.

Core blocks in v1:

- BioEmu isolated ensemble features (`data/bioemu_isolated/features.parquet`)
- ProteinMPNN, ESM-IF1, SaProt, generator disagreement
- AROMATIC-TOPO, STATIC-SAP, HYDRO-FIELD, TITRATION_SHAPE
- Gap Closure: continuous surface, packing/cavity, buried unsatisfied, Fab interface

All generated **without** target labels (`target_used=NO`).

## 6. Lightweight extractors

Python API:

```python
from feature_extension.extractors import extract_sasa, extract_aromatic

feats = extract_sasa(pdb_path="data/esmfold_fv/ADI-37123.pdb")
aro = extract_aromatic(pdb_path="data/esmfold_fv/ADI-37123.pdb")
```

| Module | Role |
|---|---|
| `extract_sasa` | Total / hydrophobic / polar SASA, RASA summaries |
| `extract_aromatic` | AROMATIC-TOPO-style aromatic exposure (HIC-relevant) |
| `extract_surface_patch` | Residue-graph hydrophobic summary (**not** continuous MS) |
| `extract_interface` | Simple heavy–light contact / BSA **proxy** (not classical Sc) |

Continuous molecular-surface patches are distributed as **precomputed**
`continuous_surface.parquet` (FreeSASA LR Gap Closure pipeline).

## 7. Recommended CV folds

File: `folds.csv` — columns `id,fold_primary,fold_shadow`.

- Includes **only training/dev IDs** (N=162) for which participants have labels.
- **Primary** = main recommended 5-fold CV (seed 42).
- **Shadow** = robustness check (seed 2026).
- Frozen in organizer Stage0 **before** this feature-extension release.

From Stage0 `CV_DESIGN_REPORT.md`:

- Sequence grouping: antibodies with **min(VH, VL) identity ≥ 0.9** are kept in the same fold (atomic groups).
- Fold assignment optimized for size / TmApp / HIC distribution balance and HIC high-tail balance — **not** chosen to maximize a particular model’s MAE.

Does **not** include Public/Private flags or target values.

## 8. Simple TVT example

Optional participant guidance (not mandatory competition policy).

For test fold `k` (0–4):

- **TEST** = fold `k`
- **VAL** = fold `(k+1) mod 5`
- **TRAIN** = remaining three folds

See `examples/example_simple_tvt_cv.py`.

## 9. Organizer observations / feature ideas

The following are **organizer-side DEV-CV observations**.
They were **not** selected using Public/Private labels.
They are **hints**, not guaranteed winning recipes.

### TmApp

- AbLang2 H+L / sequence features remain a strong baseline.
- Direct feature fusion can behave differently from late fusion.
- BioEmu **NEW_PAIRWISE** (CA-RMSD ensemble family) was one of the more stable
  additional blocks under organizer Simple TVT.
- ProteinMPNN provided weak/moderate complementary signal in direct fusion.
- A reasonable experiment:

  `sequence / PLM baseline + BioEmu dynamic descriptors + ProteinMPNN structure-compatibility features`

- Packing / cavity / interface static Fab blocks did **not** show convincing
  improvement in organizer CV, but are included for participant experimentation.

### HIC

- Exposed aromatic information is an important organizer-observed structural signal.
- Prefer aromatic exposure / **AROMATIC-TOPO**.
- Continuous hydrophobic-surface descriptors produced weak additional direct-fusion signal.
- Suggested experiment:

  `sequence / PLM baseline + aromatic exposure/topology + continuous surface descriptors`

- TITRATION_SHAPE / HYDRO_FIELD may be weak complementary / experimental blocks.

Avoid treating these as causal biology claims.

## 10. Molecular-scope caveats

| Assay | Experimental molecule |
|---|---|
| TmApp | Fab (DSF) |
| HIC | IgG (hydrophobic interaction chromatography) |

Distributed structures / ensembles may be **Fv**, **reconstructed Fab**, or
**isolated VH/VL**. Therefore:

> **molecular scope ≠ experimental molecule** for some feature families.

Treat structural features as approximations.

## 11. Licensing / attribution

See `RELEASE_NOTES.md` → *Third-party provenance and licensing notes*.
Derived artifacts are redistributed only where project license review supports v1;
unclear cases are omitted (`OMITTED_PENDING_LICENSE_REVIEW`).

## 12. Known limitations

- Full-Fab **FeNNix** features are **not** in v1 (still computing / deferred).
- BioEmu here is **isolated VH/VL only**, not full Fab.
- Fab structures use **surrogate** constant domains.
- Some tables have incomplete IDs (e.g. buried-unsatisfied N=323).
- Extractors use Bio.PDB Shrake–Rupley; continuous MS requires precomputed tables.

## 13. Future additions

Future v1.x may add full-cohort FeNNix Fab features and other newly frozen
target-blind blocks. No unfinished science is promised here.
