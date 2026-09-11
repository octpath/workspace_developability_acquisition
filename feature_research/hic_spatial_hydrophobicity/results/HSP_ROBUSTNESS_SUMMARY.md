# HSP Robustness Summary

Stage-1 VAL only. No TEST / Public / Private used for decisions.

## Geometry: CLOSEST_SC R5 vs CENTROID R8

```json
{
  "mean_jaccard": 0.5880719451925568,
  "median_jaccard": 0.5714285714285714,
  "q10": 0.375,
  "q90": 0.8,
  "frac_identical": 0.025820376513849476,
  "frac_jaccard_ge_0p8": 0.12477184006821482,
  "frac_jaccard_ge_0p5": 0.7401441571072651,
  "n_residue_centers": 75057
}
```

Mean neighbor counts: {'CLOSEST_SC_R5': 5.796937483505855, 'CENTROID_R8': 9.729340664176764}

## BM classification
**HYPERPARAMETER_SENSITIVE**

```yaml
group: BM
selected:
  neighborhood: CLOSEST_SC
  radius: 5.0
A_ALONE:
  Ridge:
    selected_VAL_mean: 0.4852522050309779
    selected_VAL_P: 0.4823705835955731
    selected_VAL_S: 0.4881338264663826
    scheme_half_range_SE: 0.002881621435404741
    best_VAL_mean_in_grid: 0.4852522050309779
    delta_selected_minus_best: 0.0
    plateau_radii_range: 0.06415833213211852
    plateau_radii_sd: 0.019396903766455833
    selected_minus_median_nearby_same_neigh: -0.03991647883731597
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.05257304638597493
    rank_selected_in_grid: 1
    n_grid: 12
  SVR:
    selected_VAL_mean: 0.4589649124821366
    selected_VAL_P: 0.4581178511716394
    selected_VAL_S: 0.4598119737926339
    scheme_half_range_SE: 0.0008470613104972491
    best_VAL_mean_in_grid: 0.4589649124821366
    delta_selected_minus_best: 0.0
    plateau_radii_range: 0.07244013796812726
    plateau_radii_sd: 0.02193236685456526
    selected_minus_median_nearby_same_neigh: -0.040529616406359825
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.05073467938092385
    rank_selected_in_grid: 1
    n_grid: 12
B_SURFACE:
  Ridge:
    selected_VAL_mean: 0.5351010394465316
    selected_VAL_P: 0.530285272093959
    selected_VAL_S: 0.5399168067991044
    scheme_half_range_SE: 0.004815767352572664
    best_VAL_mean_in_grid: 0.5351010394465316
    delta_selected_minus_best: 0.0
    plateau_radii_range: 0.041657808674759655
    plateau_radii_sd: 0.012815631927481802
    selected_minus_median_nearby_same_neigh: -0.034251966715063364
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.04064112420435717
    rank_selected_in_grid: 1
    n_grid: 12
  SVR:
    selected_VAL_mean: 0.4620770848076043
    selected_VAL_P: 0.4634746049470945
    selected_VAL_S: 0.460679564668114
    scheme_half_range_SE: 0.0013975201394902659
    best_VAL_mean_in_grid: 0.4620770848076043
    delta_selected_minus_best: 0.0
    plateau_radii_range: 0.02074772869702962
    plateau_radii_sd: 0.006203285164758065
    selected_minus_median_nearby_same_neigh: -0.011469068142566707
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.010006488232053135
    rank_selected_in_grid: 1
    n_grid: 12
classification: HYPERPARAMETER_SENSITIVE

```

### BM SURFACE Ridge VAL_mean pivot

```
                   4.0       5.0       6.0       7.5       8.0      10.0
neighborhood                                                            
CLOSEST_SC    0.573206  0.535101  0.569936  0.574717  0.568770  0.565261
CENTROID      0.567054  0.575742  0.570072  0.576759  0.574439  0.563160
```

## EIS classification
**HYPERPARAMETER_SENSITIVE**

```yaml
group: EIS
selected:
  neighborhood: CENTROID
  radius: 8.0
A_ALONE:
  Ridge:
    selected_VAL_mean: 0.5331663987163406
    selected_VAL_P: 0.5366363481870523
    selected_VAL_S: 0.529696449245629
    scheme_half_range_SE: 0.003469949470711664
    best_VAL_mean_in_grid: 0.5227136986382268
    delta_selected_minus_best: 0.01045270007811383
    plateau_radii_range: 0.03542328801201822
    plateau_radii_sd: 0.011002830141590836
    selected_minus_median_nearby_same_neigh: -0.014702809353192126
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.019925399603614835
    rank_selected_in_grid: 2
    n_grid: 12
  SVR:
    selected_VAL_mean: 0.5187735903902118
    selected_VAL_P: 0.5418828812640137
    selected_VAL_S: 0.4956642995164098
    scheme_half_range_SE: 0.02310929087380198
    best_VAL_mean_in_grid: 0.5182802361254324
    delta_selected_minus_best: 0.0004933542647793665
    plateau_radii_range: 0.02823470229883762
    plateau_radii_sd: 0.00905376444962884
    selected_minus_median_nearby_same_neigh: -0.005872464200645133
    n_plateau_radii_compatible_with_selected: 4
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.02259115176757498
    rank_selected_in_grid: 2
    n_grid: 12
B_SURFACE:
  Ridge:
    selected_VAL_mean: 0.5337365015944844
    selected_VAL_P: 0.5241315160657531
    selected_VAL_S: 0.5433414871232156
    scheme_half_range_SE: 0.009604985528731236
    best_VAL_mean_in_grid: 0.5337365015944844
    delta_selected_minus_best: 0.0
    plateau_radii_range: 0.037042303334188675
    plateau_radii_sd: 0.00998480873096592
    selected_minus_median_nearby_same_neigh: -0.02239646098287995
    n_plateau_radii_compatible_with_selected: 1
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.018777453862734772
    rank_selected_in_grid: 1
    n_grid: 12
  SVR:
    selected_VAL_mean: 0.468431819328851
    selected_VAL_P: 0.4705634185894417
    selected_VAL_S: 0.4663002200682603
    scheme_half_range_SE: 0.0021315992605906997
    best_VAL_mean_in_grid: 0.465431444484833
    delta_selected_minus_best: 0.0030003748440179945
    plateau_radii_range: 0.012960240514762134
    plateau_radii_sd: 0.004006824555495259
    selected_minus_median_nearby_same_neigh: -0.0027694114952159232
    n_plateau_radii_compatible_with_selected: 3
    n_plateau_radii: 4
    alt_neighborhood_delta_at_selected_R: 0.007634172414944906
    rank_selected_in_grid: 2
    n_grid: 12
classification: HYPERPARAMETER_SENSITIVE

```

### EIS SURFACE Ridge VAL_mean pivot

```
                   4.0       5.0       6.0       7.5       8.0      10.0
neighborhood                                                            
CLOSEST_SC    0.559501  0.560623  0.562830  0.555875  0.552514  0.554593
CENTROID      0.549950  0.557421  0.570779  0.554845  0.533737  0.553826
```

## BM vs EIS selected antibody-level similarity (MEAN)

      family_group aggregation                                   family_a                                     family_b  pearson  spearman
BM_vs_EIS_SELECTED         MAX HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0 HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0 0.494180  0.344882
BM_vs_EIS_SELECTED        MEAN HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0 HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0 0.600552  0.595332
BM_vs_EIS_SELECTED         SUM HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0 HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0 0.444484  0.436726
