# Release file policy — top_models_feature_bundle

## RELEASE (participant-facing before competition end)

- `README.md`
- `README_JA.md`
- `dev.csv`
- `test.csv`
- `folds.csv`
- `recipes.csv`
- `feature_manifest.csv`
- `EXPECTED_SCORES.json` (CV expectations; Public/Private expectations are postmortem metadata)
- `FULL_DEV_ALPHA_POLICY_BUNDLE.json`
- `data/*`
- `examples/*`
- `bundle_simple_tvt.py`
- `reproduce_top_recipes.py`
- `validate_bundle.py`
- `validate_readme_consistency.py`
- `validate_readme_consistency` / unit tests under `tests/`

## INTERNAL UNTIL COMPETITION END

- `solution.csv` — hidden Test labels + Public/Private masks  
  **Do not ship to participants before official closure.**
- Organizer-only scored diagnostics under `outputs/` that embed secret-label metrics  
  (Public/Private columns in `reproduction_scores.csv` when `--solution` was used)

## After official competition closure

Organizer may move `solution.csv` into RELEASE if desired, so that Public/Private
scoring becomes openly reproducible with:

```bash
python reproduce_top_recipes.py --dev dev.csv --test test.csv --solution solution.csv
```

## Advanced-model assets

RELEASE (participant-facing code / tables):
- ADVANCED_MODELS_README.md / ADVANCED_MODELS_README_JA.md
- advanced_models/** (code, presets, requirements, tests)
- scripts/build_residue_assets.py (extraction instructions)

INTERNAL / REVIEW_MODEL_OUTPUT until redistribution approved:
- residue_level/ablingua600m/*
- residue_level/esm2/*
- solution.csv
- advanced_outputs/ (training caches; Public/Private diagnostics)

RELEASE after QC:
- residue_level/annotations.parquet (target-blind IMGT annotations)
- residue_level/ANNOTATION_QC.md
