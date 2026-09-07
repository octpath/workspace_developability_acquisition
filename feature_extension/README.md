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
# From a directory that contains the unpacked `feature_extension/` folder:
pip install -r feature_extension/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"   # so `import feature_extension` works

# inspect folds (162 training/dev IDs)
head feature_extension/folds.csv

# extract features from one bundled Fv PDB
python feature_extension/examples/example_extract_features.py

# LEFT JOIN your local train.csv with precomputed blocks
python feature_extension/examples/example_join_precomputed.py \
  --train-csv /path/to/train.csv

# Simple TVT Ridge demo (requires your local labels)
python feature_extension/examples/example_simple_tvt_cv.py \
  --train-csv /path/to/train.csv --target TmApp
```

Download only the ZIPs you need (`code` + structures and/or precomputed).
Each archive unpacks under a top-level `feature_extension/` directory.

### Joining feature blocks (avoid silent row loss)

Always **LEFT JOIN** from your competition dataframe onto feature tables by `id`.

Some blocks have fewer than 324 IDs or partial NaNs (`BLOCK_COVERAGE.csv`).
Inner-joining every block will drop antibodies unintentionally.

Missing values: do not invent them in the released tables. Under CV, impute with
**train-fold-local** statistics only.

## 3. Directory layout

```
feature_extension/
├── README.md
├── RELEASE_NOTES.md
├── RELEASE_AUDIT.md
├── MANIFEST.csv
├── BLOCK_COVERAGE.csv
├── folds.csv
├── requirements.txt
├── data/
│   ├── esmfold_fv/
│   ├── esmfold_fab/
│   ├── bioemu_isolated/   # + FEATURE_DICTIONARY.csv
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

See each subdirectory README. Fab constants: POLICY B (UniProt CH1 / Cκ / Cλ).

## 5. Available precomputed feature blocks

See `data/precomputed_features/README.md`, `data/bioemu_isolated/README.md`,
`MANIFEST.csv`, and `BLOCK_COVERAGE.csv`.

### Surface features: do not confuse these two

- **`continuous_surface`** (precomputed): organizer Gap Closure block using FreeSASA
  Lee–Richards + exterior SAS sample points + hydrophobic surface-point patches.
- **`extract_surface_patch.py`**: lightweight **residue-graph** hydrophobic summary
  (CA adjacency among exposed hydrophobic residues).

They are **NOT numerically equivalent**. The extractor does **not** reproduce
`continuous_surface`.

### BioEmu families

Use `data/bioemu_isolated/FEATURE_DICTIONARY.csv`:

```python
dic = pd.read_csv("feature_extension/data/bioemu_isolated/FEATURE_DICTIONARY.csv")
bio_pairwise_cols = dic.loc[dic.family == "NEW_PAIRWISE", "feature"].tolist()
bio_contact_cols  = dic.loc[dic.family == "NEW_CONTACT", "feature"].tolist()
bio_flex_cols     = dic.loc[dic.family == "NEW_FLEX", "feature"].tolist()
```

## 6. Lightweight extractors

```python
from feature_extension.extractors import extract_sasa, extract_aromatic

feats = extract_sasa(pdb_path="feature_extension/data/esmfold_fv/ADI-37123.pdb")
aro = extract_aromatic(pdb_path="feature_extension/data/esmfold_fv/ADI-37123.pdb")
```

See `extractors/README.md`.

## 7. Recommended CV folds

File: `folds.csv` — columns `id,fold_primary,fold_shadow` (DEV N=162 only).

| Column | Role |
|---|---|
| `fold_primary` | **Primary** — main organizer-recommended 5-fold CV (seed 42) |
| `fold_shadow` | **Shadow** — robustness check (seed 2026) |

Frozen in Stage0 before this release. Sequence groups with
**min(VH, VL) identity ≥ 0.9** stay in the same fold. Assignment balanced size /
TmApp / HIC distributions — **not** chosen to maximize a model’s MAE.

Participants may use other sensible CV schemes; Primary/Shadow are recommendations.

### Optional Simple TVT rotation

For test fold `k` (0–4):

- **TEST** = fold `k`
- **VAL** = fold `(k+1) mod 5`
- **TRAIN** = remaining three folds

See `examples/example_simple_tvt_cv.py`.

## 8. Organizer observations / suggested experiments

These are **organizer-side DEV-CV observations**.
They were **not** selected using hidden Test / Public / Private labels.
They are **suggestions for exploration**, not guaranteed winning recipes and
**not** proof of biological mechanism.

### Direct fusion vs late fusion (important)

Organizer experiments found that a structural feature block can be unhelpful as a
**standalone** predictor or **late-fusion / residual** model, yet still improve
performance when **concatenated directly** with sequence/PLM features.

Therefore try:

```text
BASE
vs
BASE + structural block
```

under the **same** CV and model. For high-dimensional blocks (e.g. SaProt), use
conservative regularization / dimensionality handling and **fold-local** preprocessing.

### TmApp hints

- Sequence / PLM representations remain strong baselines.
- Direct feature fusion sometimes behaved differently from late-fusion / residual stacking.
- BioEmu isolated VH/VL descriptors — especially frozen **NEW_PAIRWISE** — showed useful
  complementary behavior in organizer Simple TVT CV.
- ProteinMPNN also showed complementary signal under direct fusion.
- A sensible experiment:

  `PLM / sequence baseline + BioEmu NEW_PAIRWISE + ProteinMPNN`

- **NEW_CONTACT** / **NEW_FLEX** are also reasonable blocks to explore.
- BioEmu here is **independently sampled isolated VH/VL monomer ensembles**, not full Fab dynamics.
- Do **not** say BioEmu directly predicts TmApp; do not treat the mechanism as proven.

### HIC hints

**HIC** = Hydrophobic Interaction Chromatography. Experimental assay molecule was **IgG**;
many distributed structures are Fv / Fab approximations.

- Exposed aromatic structural information was one of the clearest organizer-observed HIC signals.
- **AROMATIC-TOPO** / aromatic exposure are natural features to try.
- Precomputed **continuous molecular-surface** descriptors showed **weak** complementary
  direct-fusion signal in organizer Dev CV.
- A sensible experiment:

  `PLM / sequence baseline + aromatic exposure/topology + continuous-surface descriptors`

- HYDRO_FIELD / TITRATION_SHAPE may be treated as exploratory weak complementary blocks.

### Static Fab descriptors

`packing_cavity`, `buried_unsatisfied`, and `fab_interface` are included because they are
scientifically meaningful, target-blind structural descriptors. Organizer Dev CV did **not**
show strong consistent improvement from these blocks — but participants may find better
models/combinations. Experimentation is encouraged.

## 9. Molecular-scope caveats

| Assay | Experimental molecule |
|---|---|
| TmApp | Fab (DSF) |
| HIC | IgG (hydrophobic interaction chromatography) |

Distributed structures / ensembles may be **Fv**, **reconstructed Fab**, or
**isolated VH/VL**. Therefore **molecular scope ≠ experimental molecule** for some families.

## 10. Licensing / attribution

See `RELEASE_NOTES.md` and `RELEASE_AUDIT.md`. Code license ≠ automatic clearance for
weights or derived outputs; decisions are documented per artifact.

## 11. Known limitations

- Full-Fab **FeNNix** not in v1.
- BioEmu = isolated VH/VL only.
- Fab constants are surrogate UniProt sequences (POLICY B).
- `buried_unsatisfied` missing **ADI-47265**; some `continuous_surface` Fab-prep cells NaN for that ID.
- Extractors use Bio.PDB Shrake–Rupley; continuous SAS surface is precomputed only.

## 12. Future additions

Future v1.x may add FeNNix Fab features when frozen. No unfinished science is promised here.
