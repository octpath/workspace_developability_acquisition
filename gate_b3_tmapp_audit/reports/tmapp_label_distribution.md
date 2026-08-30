# TmApp label distribution

   role   n  n_unique  tie_frac_obs_in_tied_values  min  max      mean       sd  median  iqr     p5  p10    p25  p50    p75  p90    p95      skew  range
  Train 162        39                     0.944444 57.5 83.5 69.895062 4.488474    69.5 5.75 61.525 64.5 67.125 69.5 72.875 75.0 76.975  0.018971   26.0
 Public  81        33                     0.839506 56.0 81.5 70.043210 4.666136    70.5 5.50 62.000 64.5 67.000 70.5 72.500 75.5 77.500 -0.141599   25.5
Private  81        35                     0.827160 52.5 80.5 69.734568 5.064453    70.0 6.00 60.000 63.5 67.000 70.0 73.000 75.5 77.500 -0.472980   28.0

## Pairwise

     a       b  wasserstein       ks  ks_pvalue
 Train  Public     0.388889 0.061728   0.984389
 Train Private     0.543210 0.055556   0.995563
Public Private     0.530864 0.061728   0.998116
