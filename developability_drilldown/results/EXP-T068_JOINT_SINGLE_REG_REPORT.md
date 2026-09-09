# EXP-T068 — Joint H/L single-REG (frozen AbLingua)

- git: `ecc67b3a67d14058f8797d7524a5d1fe34a8513b`
- primary control: `EXP-T030-REPLAY-001` (contemporary matched; not historical overwrite)
- secondary historical: `EXP-T030`
- change: single encoder `[REG, H..., L...]`; single antibody-level REG; no fusion/RASA
- CV verdict: **NEGATIVE**

## Param counts

- T030 trainable: 492417
- T068 trainable: 475905
- delta (T068−T030): -16512

## Primary vs EXP-T030-REPLAY-001

- T068 P/S/mean/W: 3.539326 / 3.305622 / 3.422474 / 3.539326
- Replay P/S/mean/W: 3.232677 / 3.201958 / 3.217318 / 3.232677
- Δ Primary: +0.306649
- Δ Shadow: +0.103664
- Δ worst: +0.306649
- Both schemes same direction: YES

## Secondary vs historical EXP-T030

- Δ Primary: +0.221557
- Δ Shadow: +0.091012
- Δ worst: +0.221557

## Paired bootstrap vs replay (MAE_T068 − MAE_replay)

- Primary: Δ=+0.306649 CI95=[+0.062068, +0.554657]
- Shadow: Δ=+0.103664 CI95=[-0.129747, +0.342170]

## Test (after freeze)

- Public: 3.587596
- Private: 3.147723
- Overall: 3.367659

## Artifacts

- config / preds present; feature_path EMPTY (no fusion parquet)
- representation: NOT_EXPORTED
- checkpoints: n=0 under `results/EXP-T068_run/checkpoints/`
- checkpoint manifest: `results/EXP-T068_CHECKPOINT_MANIFEST.json`
- SHAREABLE_COMPLETE / REPRODUCED

## Interpretation

CV verdict **NEGATIVE**. Joint cross-chain attention with one REG vs separate H/L encoding.
