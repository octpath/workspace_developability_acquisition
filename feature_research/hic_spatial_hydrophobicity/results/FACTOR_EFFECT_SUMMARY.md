# Factor effect summary (Stage-1 VAL; GENERIC B3 ALONE)

n rows=1008; parsed=1008

## Hydrophobicity scale

estimator scale   median     mean  count
    Ridge    BM 0.532717 0.532804     72
    Ridge    FP 0.539245 0.537814     72
    Ridge   MIY 0.541755 0.540983     72
    Ridge    KD 0.549063 0.547521     72
    Ridge    WW 0.549294 0.548602     72
    Ridge   EIS 0.550511 0.547981     72
    Ridge  MEEK 0.555118 0.556490     72
      SVR    BM 0.509526 0.508897     72
      SVR    FP 0.511687 0.510790     72
      SVR    WW 0.519728 0.517495     72
      SVR   MIY 0.520488 0.520271     72
      SVR  MEEK 0.521742 0.519808     72
      SVR    KD 0.522291 0.522729     72
      SVR   EIS 0.527004 0.525005     72

## RAW vs MINMAX

estimator transform   median     mean  count
    Ridge    MINMAX 0.542554 0.540301    252
    Ridge       RAW 0.550437 0.548897    252
      SVR    MINMAX 0.517124 0.515394    252
      SVR       RAW 0.521386 0.520319    252

## Exposure

estimator            exposure   median     mean  count
    Ridge  SIDECHAIN_SASA_ABS 0.540604 0.538916    168
    Ridge SIDECHAIN_OVER_TIEN 0.548917 0.546896    168
    Ridge     TOTAL_RASA_TIEN 0.548972 0.547985    168
      SVR  SIDECHAIN_SASA_ABS 0.513319 0.512575    168
      SVR SIDECHAIN_OVER_TIEN 0.521541 0.520785    168
      SVR     TOTAL_RASA_TIEN 0.522215 0.520211    168

## Neighborhood

estimator      neigh   median     mean  count
    Ridge CLOSEST_SC 0.543838 0.541239    252
    Ridge   CENTROID 0.549086 0.547959    252
      SVR CLOSEST_SC 0.516901 0.515756    252
      SVR   CENTROID 0.521898 0.519957    252

## Radius

estimator  radius   median     mean  count
    Ridge     5.0 0.541968 0.538437     84
    Ridge    10.0 0.544638 0.543440     84
    Ridge     6.0 0.545703 0.542282     84
    Ridge     8.0 0.547906 0.546337     84
    Ridge     7.5 0.548317 0.544873     84
    Ridge     4.0 0.550227 0.552225     84
      SVR     5.0 0.512535 0.511240     84
      SVR     8.0 0.518385 0.517083     84
      SVR     6.0 0.518390 0.517549     84
      SVR    10.0 0.520846 0.520250     84
      SVR     7.5 0.522280 0.519077     84
      SVR     4.0 0.524760 0.521941     84

## Bundle effect (GENERIC A_ALONE Ridge)

bundle
BR      0.545982
B3      0.546774
BC      0.556274
BH      0.570853
BALL    0.605842

## Incremental over SURFACE (GENERIC B3; negative Δ better)

estimator
Ridge    0.008112
SVR     -0.000948

## Interpretation bullets

- Best scales on VAL alone (Ridge): see scale table (lower median VAL better).
- CLOSEST_SC helps SVR more than Ridge.
- Radii ~5–7.5 Å preferred over 4 Å.
- SIDECHAIN_SASA_ABS slightly better than Tien-normalized exposures.
- Hotspot bundle BH and full BALL overfit vs B3/BR.
