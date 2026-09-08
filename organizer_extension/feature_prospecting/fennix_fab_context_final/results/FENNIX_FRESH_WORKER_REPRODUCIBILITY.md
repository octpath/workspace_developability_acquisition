# FeNNix fresh-worker reproducibility

UTC: 2026-09-08T11:14:36.341763+00:00
IDs: ADI-47313, ADI-47060, ADI-45469 (includes prior stack-smash ADI-47313)
audit-tag: fresh_repro_final

## Verdict: **NUMERICALLY_CLOSE**

```
       id condition  n_features  max_abs_diff  median_rel_diff  corr_coef
ADI-45469         B          70  4.007332e-01         0.001417   0.999993
ADI-45469         C         112  3.552714e-15         0.000000   1.000000
ADI-45469         M          70  0.000000e+00         0.000000   1.000000
ADI-47060         B          70  3.205866e-01         0.001773   0.999993
ADI-47060         C         112  3.552714e-15         0.000000   1.000000
ADI-47060         M          70  0.000000e+00         0.000000   1.000000
ADI-47313         B          70  3.552714e-15         0.000000   1.000000
ADI-47313         C         112  2.842171e-14         0.000000   1.000000
ADI-47313         M          70  0.000000e+00         0.000000   1.000000
```

- max_abs_diff=0.400733
- median_rel_diff=0
- min_corr=0.99999327

Notes:
- C and M are bit-identical / float-noise level vs production.
- B shows small FIRE/R1 optimizer drift on some Abs (max |Δ|~0.4 on K aggregates) with corr≥0.99999.
- ADI-47313 (prior native crash) fully matches on B/C/M in this fresh process.
- Production r1/matched artifacts were restored; production feature table unchanged.
