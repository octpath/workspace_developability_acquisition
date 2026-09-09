# EXP-T066 — RASA-weighted residue aggregation prior

- git: `f2cacce6fe158fcd6b76d6dcd3fc878b8183bd05`
- control: `EXP-T037` (stored canonical OOF; not retrained)
- change: per-chain continuous RASA-weighted pool → shared zero-init Linear → REG residual
- encoder input: **unchanged** vs EXP-T037
- CV verdict: **NEGATIVE**

## 1–4. vs EXP-T037

- T066 P/S/mean/W: 2.773911 / 2.903064 / 2.838487 / 2.903064
- Δ Primary: +0.020096
- Δ Shadow: +0.130065
- Δ worst: +0.130065
- Both schemes same direction: YES

## 5. Paired bootstrap (MAE_T066 − MAE_T037)

- Primary: Δ=+0.020096 CI95=[-0.093920, +0.130360]
- Shadow: Δ=+0.130065 CI95=[+0.012402, +0.248770]

## 6. Three-way comparison

| EXP | role | Primary | Shadow | worst |
|-----|------|---------|--------|-------|
| T037 | no RASA | 2.753816 | 2.772998 | 2.772998 |
| T065 | token-additive RASA | 2.814704 | 2.874607 | 2.874607 |
| T066 | aggregation RASA | 2.773911 | 2.903064 | 2.903064 |

Conservative reading: one aggregation experiment does not prove RASA 'works as readout'; compare signed deltas vs T037 and vs T065 only.

- T066−T065 worst: +0.028457

## 7. vs classical EXP-T045

- T045 worst: 2.728948
- T066−T045 worst: +0.174115

## 8. Test (after freeze)

- Public: 3.241731
- Private: 3.188504
- Overall: 3.215117

## 9. RASA QC

- expected/mapped/missing: 75057/75057/0
- AA mismatches: 0
- zero-denom chains: 0
- hashes: {'rasa_heavy.npy': 'b59e4037913d113e417bc5a2939fe372f225fcdcf4223ac933e0ca7aa5c1d523', 'rasa_light.npy': 'f5689ff8b85c2189fb35195d3a31e6fbe3efb457125274f2234cc98c872fc85a', 'rasa_ids.npy': 'ce62b155c7355352d9941f19d3be50263c6835ed0f3ce3429a77f9b0714c665d', 'rasa_meta.json': '1e759063dc00a656c516c6a8137eb19219d7f41258330bd780d7e452ab6e9b0d'}

## 10–11. Artifacts / reproducibility

- config / features / RASA / preds present
- representation: NOT_EXPORTED (no weight checkpoints)
- SHAREABLE_COMPLETE / REPRODUCED / eligible=YES

## 12. Interpretation

CV verdict **NEGATIVE**. Token-additive RASA (T065) was detrimental; aggregation RASA (T066) is a separate mechanism. Do not over-claim from a single experiment.

## Post-hoc (diagnostic only)

```json
{
  "corr_delta_ae_vs_mean_rasa": -0.07942741832591971,
  "corr_delta_ae_vs_cdr3_rasa": -0.058759140063049775,
  "note": "antibody-level; diagnostic only"
}
```

## STOP — no EXP-T067 / sweeps
