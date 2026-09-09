# EXP-T065 — Continuous RASA annotation for AbLingua Transformer

- git: `0dfa933d6b49f54a42ded09bd3fcf13806bbc47c`
- control: `EXP-T037` replay under current implementation
- scientific change: continuous additive `Linear(1,d_model,bias=False)` zero-init RASA projection
- verdict: **NEGATIVE**

## 1. Did current EXP-T037 replay reproduce historical performance?

- Historical P/S/W: 2.753816 / 2.772998 / 2.772998
- Replay P/S/W: 2.753816 / 2.772998 / 2.772998
- CV deltas (replay−hist): P=+0.000000 S=+0.000000 W=+0.000000
- Pred max|Δ|: primary=0 shadow=0
- Sufficient for causal control: YES

## 2–5. Continuous RASA vs replay control

- EXP-T065 P/S/mean/W: 2.814704 / 2.874607 / 2.844656 / 2.874607
- Δ Primary (T065−replay): +0.060889 → worsened
- Δ Shadow: +0.101609
- Δ CV worst: +0.101609
- Both schemes same direction: YES

## 6. Paired bootstrap (MAE_T065 − MAE_replay; negative = T065 better)

- Primary: Δ=+0.060889 CI95=[-0.056669, +0.181396]
- Shadow: Δ=+0.101609 CI95=[+0.007228, +0.197537]

### Fold MAE deltas (Primary)

- fold 0: T065=2.5035 ctl=2.4844 Δ=+0.0192
- fold 1: T065=2.7969 ctl=2.8924 Δ=-0.0955
- fold 2: T065=2.9711 ctl=2.7601 Δ=+0.2110
- fold 3: T065=3.1704 ctl=2.9845 Δ=+0.1859
- fold 4: T065=2.6272 ctl=2.6431 Δ=-0.0159

### Fold MAE deltas (Shadow)

- fold 0: T065=2.7078 ctl=2.6485 Δ=+0.0593
- fold 1: T065=2.5611 ctl=2.2344 Δ=+0.3267
- fold 2: T065=3.3347 ctl=3.2852 Δ=+0.0495
- fold 3: T065=2.5497 ctl=2.5109 Δ=+0.0388
- fold 4: T065=3.2188 ctl=3.1899 Δ=+0.0290

## 7. Beat historical T037 / replay / classical T045?

- vs historical T037 worst: +0.101609 (P +0.060889, S +0.101609)
- vs replay worst: +0.101609
- vs classical T045 worst (2.728948): +0.145659

## 8. Public / Private / Overall (after freeze)

- Public: 3.222722
- Private: 3.249894
- Overall: 3.236308

## 9. RASA alignment/QC

- expected residues: 75057
- mapped: 75057
- missing (contrib=0): 0
- AA mismatches: 0
- range: {'min': 0.0, 'max': 1.0968040227890015, 'mean': 0.2593059539794922}
- source: /workspace_developability_acquisition/feature_extension/data/esmfold_fv
- definition: Bio.PDB.SASA.ShrakeRupley probe=1.4 n_points=100 MaxASA=Tien2013
- hashes: {'rasa_heavy.npy': 'b59e4037913d113e417bc5a2939fe372f225fcdcf4223ac933e0ca7aa5c1d523', 'rasa_light.npy': 'f5689ff8b85c2189fb35195d3a31e6fbe3efb457125274f2234cc98c872fc85a', 'rasa_ids.npy': 'ce62b155c7355352d9941f19d3be50263c6835ed0f3ce3429a77f9b0714c665d', 'rasa_meta.json': '1e759063dc00a656c516c6a8137eb19219d7f41258330bd780d7e452ab6e9b0d'}

## 10. Error concentration (post-hoc, antibody-level)

```json
{
  "delta_ae_mean_high_rasa_abs": -0.020824605054815808,
  "delta_ae_mean_low_rasa_abs": 0.14260203357586598,
  "corr_delta_ae_vs_mean_rasa": -0.12322918999343221,
  "corr_delta_ae_vs_cdr_frac": -0.06773985018065905,
  "corr_delta_ae_vs_cdr3_frac": -0.012231865685292465,
  "note": "antibody-level correlations only; not residue-attribution"
}
```

## 11. Artifact completeness

- config: experiments/configs/EXP-T065.yaml
- fixed branch: experiments/features/EXP-T065.parquet (feature_role=FUSION_FIXED_BRANCH)
- RASA: experiments/inputs/EXP-T065_rasa.parquet
- preds: experiments/predictions/EXP-T065/{oof_primary,oof_shadow,test}.csv
- AbLingua: input_asset_ref in config (not duplicated)
- representation export: NOT_EXPORTED (fold caches store predictions, not weights; exporting would need new checkpoint subsystem)

## 12. Reproducibility

- shareability_status: SHAREABLE_COMPLETE
- reproduction_status: REPRODUCED (score recomputation from saved predictions)
- canonical_benchmark_eligible: YES
- EXP-T037 historical predictions unchanged

## STOP

No EXP-T066 / sweeps / gates. Inspect this result before next experiment.
