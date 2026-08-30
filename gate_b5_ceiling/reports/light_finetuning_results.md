# Light fine-tuning results

## Setup
- Model: ESM-2 t30 150M
- Strategy: manual LoRA on last transformer block (`q/k/v/out`, `fc1/fc2`); fallback last-block unfreeze
- Pre-registered LRs: `{3e-5, 1e-4}`; head dropout 0.3; AdamW; grad clip; internal ES
- Validation: 3-fold grouped CV × 1 seed (compute-limited)

## Results

| Target | Best LR | CV MAE | Notes |
|--------|--------:|-------:|-------|
| HIC | 1e-4 | ≈0.52 | ≈ constant median; **no gain** |
| TmApp | 1e-4 | ≈41 | **training collapse / unstable** |

Holdout oneshot for FT finalists was **skipped** (no trustworthy frozen checkpoint).

## Conclusion
Light LoRA / last-block adaptation **did not** improve the rigorous MAE ceiling. Do not claim a fine-tuning ceiling from these runs. Further FT would require more stable recipes and multi-seed confirmation — **not justified** before packaging.
