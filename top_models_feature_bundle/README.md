# Organizer Top-Model Feature Bundle

日本語版: [`README_JA.md`](README_JA.md)

## What this is

A compact set of **target-blind** features for the strongest organizer-side
**feature-level** Simple TVT CV recipes (Ridge/Lasso endgame), intended for
reproducing or extending those recipes in the advanced-model phase.

## Files

See `feature_manifest.csv` for block → file mapping.

| file | role |
|------|------|
| `folds.csv` | fold_primary / fold_shadow |
| `recipes.csv` | Top-3 per target (CV rank frozen **before** Public/Private) |
| `data/base_sequences.parquet` | id, heavy, light |
| `data/*.parquet` | feature blocks |

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes (advanced-model freeze; CV only)

These are **feature-level** recipes frozen using **canonical Simple TVT CV only**.
They are **not** the all-time Organizer leaderboard across historical prediction
blends/stacks.

**Public/Private scores were not used to select these recipes.**

Historical HIC ≈0.42 MAE results belong to prediction-level SVR blends and are
documented separately under `organizer_extension/linear_model_closure/`.

Authoritative freeze: `organizer_extension/linear_model_closure/ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`
(also mirrored in this bundle’s `recipes.csv`).

### TmApp

1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV Primary/Shadow **2.732072376654289 / 2.784957206877823**
   Features: AbLang2 paired heavy+light chain protein-language-model embedding
   + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries)
   + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors
   + ProteinMPNN sequence–structure compatibility native-score descriptors
   + AbLingua-600M masked-mean global heavy+light concatenated embedding
   + AbLingua-600M CDR3-guided residue pooling embedding

2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV Primary/Shadow **2.7466001170280285 / 2.822725206703513**
   Features: same as (1) **without** AbLingua CDR3-guided pooling
   (AbLang2 + SEQ_BASIC + BioEmu NEW_PAIRWISE + ProteinMPNN + AbLingua GLOBAL)

3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV Primary/Shadow **2.702681240871287 / 2.845936772811553**
   Features: AbLang2 paired heavy+light chain protein-language-model embedding
   + stage-1 SEQ_BASIC sequence descriptors
   + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors
   + ProteinMPNN sequence–structure compatibility native-score descriptors
   (no AbLingua blocks)

### HIC

1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV Primary/Shadow **0.4872225781431885 / 0.4833719610538652**
   Features: ESM-2 Heavy-chain protein-language-model embedding
   + SEQ_ALL sequence descriptors
   + AROMATIC-TOPO exposed-aromatic structural topology descriptors
   + HYDRO_FIELD continuous hydrophobic-field surface descriptors
   + TITRATION_SHAPE electrostatic titration-shape descriptors

2. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV Primary/Shadow **0.4822073134231601 / 0.4898232886862832**
   Features: ESM-2 Heavy-chain protein-language-model embedding
   + SEQ_ALL sequence descriptors
   + AROMATIC-TOPO exposed-aromatic structural topology descriptors
   + CONTINUOUS_SURFACE continuous molecular-surface descriptors

3. `HIC_ESM2_SEQ_AROMATIC__LASSO` — LASSO — CV Primary/Shadow **0.4927303900206625 / 0.4808724742416768**
   Features: ESM-2 Heavy-chain protein-language-model embedding
   + SEQ_ALL sequence descriptors
   + AROMATIC-TOPO exposed-aromatic structural topology descriptors
   (no continuous-surface / hydro-field / titration blocks)

## Quick usage

```python
import pandas as pd
train = pd.read_csv("official_dev.csv")  # competition train table
feat = pd.read_parquet("data/ablang2.parquet")
train = train.merge(feat, on="id", how="left")
```

## Ridge / Lasso examples

- `examples/train_ridge.py`
- `examples/train_lasso.py`

## Blocks in this release

- AROMATIC_TOPO
- AbLang2_HL_paired
- AbLingua_CDR3
- AbLingua_HL_mean
- BIOEMU_NEW_PAIRWISE
- CONTINUOUS_SURFACE
- ESM2_H
- HYDRO_FIELD
- M1_PROTEINMPNN
- SEQ_ALL
- SEQ_BASIC
- TITRATION_SHAPE

## Important

- Fit scaler / PCA / imputation **inside training folds**
- Do **not** use Public/Private metadata for model selection
- Feature files are target-blind
- Some model-derived blocks may have redistribution conditions — see `license_status` in `feature_manifest.csv`
- Organizer Ridge may use PCA32 on AbLingua blocks; this bundle distributes **raw** embeddings

## Reproducing organizer CV

Train / CV from this bundle alone (no PLM / BioEmu / ProteinMPNN recompute):

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv
```

Optional submission pair:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --tm-recipe TM_PARENT_ABLINGUA_CDR3__RIDGE \
    --hic-recipe HIC_HYDRO_TITRATION__LASSO \
    --submission outputs/submission.csv
```

`dev.csv` schema: `id,heavy,light,TmApp,HIC`  
`test.csv` schema: `id,heavy,light`

**`solution.csv` is NOT required for training or CV.** It is used only for
post-hoc Public/Private evaluation (organizer-internal until competition end):

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv
```

See `RELEASE_FILE_POLICY.md` and `BUNDLE_REPRODUCTION_AUDIT.md`.

## Advanced models (XGBoost / Transformers)

See [`ADVANCED_MODELS_README.md`](ADVANCED_MODELS_README.md) / [`ADVANCED_MODELS_README_JA.md`](ADVANCED_MODELS_README_JA.md).

```bash
python advanced_models/validate_environment.py
python advanced_models/run_benchmark.py --stage all --device cuda --dev dev.csv --test test.csv --write-submissions
```

## Precomputed organizer benchmark results

Organizer-side consolidated scores (linear / ensemble / XGBoost / Transformers / fusion / cross-family quickcheck):

- [`results/MODEL_BENCHMARK_SUMMARY.csv`](results/MODEL_BENCHMARK_SUMMARY.csv)
- [`results/MODEL_BENCHMARK_REPORT_JA.md`](results/MODEL_BENCHMARK_REPORT_JA.md)
- [`results/MODEL_BENCHMARK_REPORT.md`](results/MODEL_BENCHMARK_REPORT.md)

### AbLang2 position-aware follow-up

TmApp-only follow-up (frozen AbLang2 residue + IMGT/FR-CDR Transformer):

- [`results/ABLANG2_POSITION_AWARE_FOLLOWUP_JA.md`](results/ABLANG2_POSITION_AWARE_FOLLOWUP_JA.md)
- [`results/ABLANG2_POSITION_AWARE_FOLLOWUP.md`](results/ABLANG2_POSITION_AWARE_FOLLOWUP.md)
- [`results/ablang2_followup/`](results/ablang2_followup/)
- Plan: [`advanced_models/ablang2_followup/ABLANG2_FOLLOWUP_PLAN.md`](advanced_models/ablang2_followup/ABLANG2_FOLLOWUP_PLAN.md)

Follow-up is reproducible from tracked code + residue **part** files (`residue_level/ablang2/*.part0|1`); run `residue_level/assemble_embeddings.sh` if `.npy` is absent. Public/Private are **POSTMORTEM ONLY**. `solution.csv` is not distributed. License for AbLang2-derived residue outputs: **REVIEW_MODEL_OUTPUT**.

These tables are **already available**. All training code remains reproducible if you want to re-run experiments.

**`solution.csv` is not distributed.** Public/Private columns in the summary are **post-competition postmortem** metrics (organizers may place `solution.csv` locally after closure to rescore). They were **not** used for model selection.

## Scope clarification

Earlier chat references to an “Organizer Top-3” meant only the **restricted**
endgame Ridge/Lasso feature-recipe inventory, **not** all historical organizer
linear models. Historical HIC ≈0.42 results were mainly **prediction blends of
SVR bases**, tracked separately in `organizer_extension/linear_model_closure/`.
