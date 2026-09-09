# EXP-T072 — Separate H/L + cross-attention bridge (frozen AbLingua)

- git: `ddb98490b287e5c48dea85461cd3db07da5c74d2`
- primary control: `EXP-T030-REPLAY-001`
- secondary: `EXP-T071`, `EXP-T070`, `EXP-T068`, historical `EXP-T030`
- change: separate encoders + zero-gated bidirectional residue cross-attn between layers; REG not in cross Q/K/V
- CV verdict (vs REPLAY): **NEGATIVE**

## Param counts

- T030: 492417 (repr_dim=256)
- T068: 475905 (repr_dim=128)
- T070: 492417 (repr_dim=256)
- T071: 492417 (repr_dim=256)
- T072: 558467 (repr_dim=256)
- T072 param_account: {'encoder': 264960, 'cross_attention': 66048, 'gates': 2, 'head': 33025, 'other': 194432, 'total': 558467}

## Cross-gate statistics (CV)

- n_rows: 30
- g_H mean/median/std/|mean|/sign_consistency: 0.002119354374008253 / 0.0013205084833316505 / 0.0072393714455023924 / 0.005323402806728457 / 0.6
- g_L mean/median/std/|mean|/sign_consistency: 0.0011513243768907463 / 0.0001250708446605131 / 0.011441935862777513 / 0.007240998606236341 / 0.5

## Primary vs EXP-T030-REPLAY-001

- T072 P/S/mean/W: 3.266847 / 3.252076 / 3.259461 / 3.266847
- Replay P/S/mean/W: 3.232677 / 3.201958 / 3.217318 / 3.232677
- Δ Primary/Shadow/worst: +0.034169 / +0.050118 / +0.034169
- Both schemes same direction: YES

## Secondary vs EXP-T071

- T071 P/S/W: 3.311991 / 3.218343 / 3.311991
- Δ Primary/Shadow/worst: -0.045144 / +0.033732 / -0.045144

## Secondary vs EXP-T070

- T070 P/S/W: 3.329091 / 3.186958 / 3.329091
- Δ Primary/Shadow/worst: -0.062244 / +0.065118 / -0.062244

## Secondary vs EXP-T068

- T068 P/S/W: 3.539326 / 3.305622 / 3.539326
- Δ Primary/Shadow/worst: -0.272479 / -0.053546 / -0.272479

## Contextual vs historical EXP-T030

- Δ Primary/Shadow/worst: -0.050922 / +0.037465 / -0.050922

## Paired bootstrap

- vs REPLAY Primary: Δ=+0.034169 CI95=[-0.118446, +0.184954] P(Δ<0)=0.328
- vs REPLAY Shadow: Δ=+0.050118 CI95=[-0.101963, +0.201901] P(Δ<0)=0.264
- vs T071 Primary: Δ=-0.045144 CI95=[-0.200503, +0.109467] P(Δ<0)=0.718
- vs T071 Shadow: Δ=+0.033732 CI95=[-0.118152, +0.180784] P(Δ<0)=0.331
- vs T070 Primary: Δ=-0.062244 CI95=[-0.209239, +0.084004] P(Δ<0)=0.801
- vs T070 Shadow: Δ=+0.065118 CI95=[-0.156458, +0.279300] P(Δ<0)=0.272
- vs T068 Primary: Δ=-0.272479 CI95=[-0.537991, -0.004274] P(Δ<0)=0.977
- vs T068 Shadow: Δ=-0.053546 CI95=[-0.294244, +0.180927] P(Δ<0)=0.660

## Test (after freeze)

- Public: 3.663092
- Private: 3.306090
- Overall: 3.484591

## Artifacts

- config / preds; no fusion parquet
- cross gates: results/EXP-T072_CROSS_GATES.csv ; results/EXP-T072_run/cross_gates.csv
- checkpoints: n=0
- SHAREABLE_COMPLETE / REPRODUCED

## Interpretation

CV vs REPLAY **NEGATIVE**. Separate H/L with learnable zero-init cross-attn gates.

## Registered EXP-T072 — comparison report written separately

- Series T071→T072 complete for this runner; see `results/T030_T068_T070_T071_T072_ARCHITECTURE_COMPARISON.md` (written separately).

