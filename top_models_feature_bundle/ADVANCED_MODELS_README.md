# Advanced models — participant guide

Japanese: [`ADVANCED_MODELS_README_JA.md`](ADVANCED_MODELS_README_JA.md)

This suite trains **XGBoost** and small **annotation-aware Transformers** on the
frozen Top-3 feature recipes / residue assets **inside this bundle**.

Public/Private labels (`solution.csv`) are **never** used for model or
hyperparameter selection. If supplied, they are scored **after** CV selection
(**POSTMORTEM ONLY**).

## Environment

```bash
cd top_models_feature_bundle

uv venv
uv pip install -r advanced_models/requirements.txt

# or use an existing env with torch+cuda and xgboost
python advanced_models/validate_environment.py
```

Expected checks: Python, torch, `torch.cuda.is_available()`, GPU name,
XGBoost version, residue assets, DEV/Test N=162.

## Linear reproduction (unchanged)

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv
```

Organizer postmortem:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv
```

## XGBoost

Single recipe:

```bash
python advanced_models/run.py \
    --target TmApp \
    --model xgboost \
    --recipe TM_PARENT_ABLINGUA_CDR3__RIDGE \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model xgboost \
    --recipe HIC_HYDRO_TITRATION__LASSO \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

All six Top-3 recipes:

```bash
python advanced_models/run_benchmark.py \
    --stage xgboost \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## Scratch Transformer

```bash
python advanced_models/run.py \
    --target TmApp \
    --model scratch_transformer \
    --annotation-mode full \
    --merge concat \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target TmApp \
    --model scratch_transformer \
    --annotation-mode full \
    --merge mean \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model scratch_transformer \
    --annotation-mode full \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## Frozen-PLM Transformer

```bash
python advanced_models/run.py \
    --target TmApp \
    --model frozen_transformer \
    --annotation-mode full \
    --merge concat \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model frozen_transformer \
    --annotation-mode full \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## Full benchmark

```bash
python advanced_models/run_benchmark.py \
    --stage all \
    --device cuda \
    --dev dev.csv \
    --test test.csv \
    --write-submissions
```

Organizer-internal postmortem (does **not** affect selection):

```bash
python advanced_models/run_benchmark.py \
    --stage all \
    --device cuda \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv \
    --write-submissions
```

## Smoke test (NOT CANONICAL)

```bash
python advanced_models/run_benchmark.py \
    --stage transformer_sequence \
    --device cuda \
    --quick \
    --dev dev.csv \
    --test test.csv
```

## Outputs

| path | content |
|------|---------|
| `advanced_outputs/ADVANCED_MODEL_RESULTS.csv` | all CV / optional PP scores |
| `advanced_outputs/ADVANCED_CV_SELECTED_CONFIGS.json` | CV winners |
| `advanced_outputs/predictions/` | Test predictions |
| `advanced_outputs/submissions/submission_best_cv.csv` | `id,TmApp,HIC` |
| `advanced_outputs/logs/` | per-fold caches + config hashes |

Submission schema: exactly `id,TmApp,HIC` (162 rows).

## Residue assets / license

See `residue_level/RESIDUE_ASSET_AUDIT.md` and `RELEASE_FILE_POLICY.md`.
AbLingua / ESM-2 residue tensors are marked **REVIEW_MODEL_OUTPUT**.
Local regeneration: `python scripts/build_residue_assets.py --device cuda`.
