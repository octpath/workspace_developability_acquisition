# EXP-T070 — Joint H/L dual-REG (frozen AbLingua)

- git: `461832c538b8d72903c67e18cf67af3b49991a64`
- primary control: `EXP-T030-REPLAY-001`
- secondary: `EXP-T068`, historical `EXP-T030`
- change: joint encoder `[REG_H, H..., REG_L, L...]`; readout REG_H||REG_L; no fusion/RASA
- CV verdict (vs REPLAY): **MIXED**

## Param counts

- T030: 492417 (repr_dim=256)
- T068: 475905 (repr_dim=128)
- T070: 492417 (repr_dim=256)

## Primary vs EXP-T030-REPLAY-001

- T070 P/S/mean/W: 3.329091 / 3.186958 / 3.258024 / 3.329091
- Replay P/S/mean/W: 3.232677 / 3.201958 / 3.217318 / 3.232677
- Δ Primary/Shadow/worst: +0.096413 / -0.015000 / +0.096413
- Both schemes same direction: NO

## Secondary vs EXP-T068

- T068 P/S/W: 3.539326 / 3.305622 / 3.539326
- Δ Primary/Shadow/worst: -0.210235 / -0.118664 / -0.210235

## Contextual vs historical EXP-T030

- Δ Primary/Shadow/worst: +0.011321 / -0.027652 / +0.011321

## Paired bootstrap

- vs REPLAY Primary: Δ=+0.096413 CI95=[-0.125439, +0.315664]
- vs REPLAY Shadow: Δ=-0.015000 CI95=[-0.254274, +0.224857]
- vs T068 Primary: Δ=-0.210235 CI95=[-0.482716, +0.072805]
- vs T068 Shadow: Δ=-0.118664 CI95=[-0.293677, +0.047527]

## Test (after freeze)

- Public: 3.440557
- Private: 3.085275
- Overall: 3.262916

## Artifacts

- config / preds; no fusion parquet
- checkpoints: n=0
- SHAREABLE_COMPLETE / REPRODUCED

## Interpretation

CV vs REPLAY **MIXED** (Primary worse, Shadow slightly better; bootstrap CIs include 0).

- vs EXP-T068: clear point-estimate recovery on Primary/Shadow/worst (dual-REG capacity restored).
- vs REPLAY / historical T030: no clear joint-encoding win when dual REG is preserved.
- Favored reading: T068 penalty was largely the single-REG bottleneck; joint H/L attention itself is not clearly helpful under this regime.

## STOP — no further experiment

