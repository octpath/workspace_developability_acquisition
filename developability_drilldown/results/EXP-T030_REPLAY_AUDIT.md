# EXP-T030 replay audit

- git: `b4c510dd9af24d3aef5762354466799bda5f0d60`
- verdict: **FAIL_STOP**

## Canonical (experiments.csv)
- Primary: 3.317769289998
- Shadow: 3.214610237153
- prior status: RESULT_VERIFIED

## Replay (current AnnotatedTransformer, no fusion)
- Primary: 3.232677420471
- Shadow: 3.201957977358
- config_hash: `0814cb40363d8eef`

## Prediction deltas vs historical OOF
- primary max |Δ|: 2.741976e+00
- shadow max |Δ|: 3.205467e+00
- tolerances: pred≤1e-05, mae≤1e-06

## Reproducibility verdict: FAIL_STOP

