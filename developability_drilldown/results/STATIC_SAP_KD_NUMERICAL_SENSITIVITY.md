# STATIC_SAP_KD numerical / radius sensitivity

## Shrake–Rupley n_points 100 vs 960
```
      feature  pearson  spearman  median_rel_diff  max_rel_diff
 SSKD_ALL_MAX 0.986630  0.980739         0.013986      0.065066
SSKD_ALL_MEAN 0.995870  0.995314         0.003536      0.017082
 SSKD_ALL_SUM 0.996445  0.995525         0.003536      0.017082
```

- unstable_flag: False
- note: MAX shows mild n_points sensitivity (pearson≈0.987); MEAN/SUM stable — NOT an unexpected STOP condition

## Radius R_REF=5 vs R=10 (descriptor QC only)
```
                            pair  pearson  spearman  median_ratio_R10_over_R5  scale_mean_R5  scale_mean_R10
  SSKD_ALL_MAX_vs_SSKD10_ALL_MAX 0.044031  0.036498                  3.351148       0.882634        2.924451
SSKD_ALL_MEAN_vs_SSKD10_ALL_MEAN 0.720548  0.712346                  6.693802       0.201189        1.347255
  SSKD_ALL_SUM_vs_SSKD10_ALL_SUM 0.763224  0.761441                  6.693802      46.609762      312.088874
```

No model selection uses R=10 or n_points=960.
