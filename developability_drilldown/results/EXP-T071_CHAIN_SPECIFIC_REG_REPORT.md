# EXP-T071 — Joint H/L chain-specific dual-REG (frozen AbLingua)

- git: `ddb98490b287e5c48dea85461cd3db07da5c74d2`
- primary control: `EXP-T030-REPLAY-001`
- secondary: `EXP-T070`, `EXP-T068`, historical `EXP-T030`
- change: joint encoder with chain-specific REG attention mask; residues may still attend H↔L
- CV verdict (vs REPLAY): **NEGATIVE**

## Param counts

- T030: 492417 (repr_dim=256)
- T068: 475905 (repr_dim=128)
- T070: 492417 (repr_dim=256)
- T071: 492417 (repr_dim=256)

## Primary vs EXP-T030-REPLAY-001

- T071 P/S/mean/W: 3.311991 / 3.218343 / 3.265167 / 3.311991
- Replay P/S/mean/W: 3.232677 / 3.201958 / 3.217318 / 3.232677
- Δ Primary/Shadow/worst: +0.079313 / +0.016386 / +0.079313
- Both schemes same direction: YES

## Primary scientific vs EXP-T070

- T070 P/S/W: 3.329091 / 3.186958 / 3.329091
- Δ Primary/Shadow/worst: -0.017100 / +0.031386 / -0.017100

## Secondary vs EXP-T068

- T068 P/S/W: 3.539326 / 3.305622 / 3.539326
- Δ Primary/Shadow/worst: -0.227335 / -0.087279 / -0.227335

## Contextual vs historical EXP-T030

- Δ Primary/Shadow/worst: -0.005779 / +0.003733 / -0.005779

## Paired bootstrap

- vs REPLAY Primary: Δ=+0.079313 CI95=[-0.071406, +0.228536]
- vs REPLAY Shadow: Δ=+0.016386 CI95=[-0.121293, +0.153937]
- vs T070 Primary: Δ=-0.017100 CI95=[-0.253366, +0.219776]
- vs T070 Shadow: Δ=+0.031386 CI95=[-0.172014, +0.238924]
- vs T068 Primary: Δ=-0.227335 CI95=[-0.492070, +0.029702]
- vs T068 Shadow: Δ=-0.087279 CI95=[-0.336741, +0.163075]

## Test (after freeze)

- Public: 3.609984
- Private: 3.381431
- Overall: 3.495707

## Artifacts

- config / preds; no fusion parquet
- checkpoints: n=0
- SHAREABLE_COMPLETE / REPRODUCED

## Interpretation

CV vs REPLAY **NEGATIVE**. Chain-specific REG mask vs unrestricted dual REG (T070).

## STOP after T072 series — see comparison report

