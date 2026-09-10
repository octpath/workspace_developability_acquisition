# Geometry-zero inference ablation (ARCH-6G, no retrain)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- definition: set `cross_geom_weight = 0` at inference; attention path otherwise unchanged.
- ΔMAE = MAE(geom0) − MAE(normal); positive ⇒ zeroing geometry worsens MAE (geometry used).

## Aggregate OOF

| code | scheme | split | MAE normal | MAE geom0 | Δ |
|---|---|---|---:|---:|---:|
| EXP-T105 | primary | VAL_OOF | 2.984888 | 2.984892 | 4.16791e-06 |
| EXP-T105 | primary | TEST_OOF | 3.352230 | 3.352258 | 2.83748e-05 |
| EXP-T105 | shadow | VAL_OOF | 3.025324 | 3.025313 | -1.06906e-05 |
| EXP-T105 | shadow | TEST_OOF | 3.300029 | 3.300037 | 8.28872e-06 |
| EXP-T106 | primary | VAL_OOF | 3.020849 | 3.020858 | 9.4661e-06 |
| EXP-T106 | primary | TEST_OOF | 3.351617 | 3.351626 | 9.4661e-06 |
| EXP-T106 | shadow | VAL_OOF | 3.089010 | 3.088995 | -1.55414e-05 |
| EXP-T106 | shadow | TEST_OOF | 3.277701 | 3.277654 | -4.70008e-05 |
| EXP-T107 | primary | VAL_OOF | 2.978038 | 2.977912 | -0.000125791 |
| EXP-T107 | primary | TEST_OOF | 3.409044 | 3.408616 | -0.000428777 |
| EXP-T107 | shadow | VAL_OOF | 3.044083 | 3.044090 | 7.5352e-06 |
| EXP-T107 | shadow | TEST_OOF | 3.226748 | 3.226765 | 1.70013e-05 |
| EXP-T108 | primary | VAL_OOF | 2.936663 | 2.936611 | -5.22519e-05 |
| EXP-T108 | primary | TEST_OOF | 3.286036 | 3.286110 | 7.38921e-05 |
| EXP-T108 | shadow | VAL_OOF | 3.002687 | 3.002623 | -6.40963e-05 |
| EXP-T108 | shadow | TEST_OOF | 3.292053 | 3.292029 | -2.34062e-05 |
| EXP-T118 | primary | VAL_OOF | 2.828037 | 2.828079 | 4.21501e-05 |
| EXP-T118 | primary | TEST_OOF | 3.217477 | 3.217439 | -3.85237e-05 |
| EXP-T118 | shadow | VAL_OOF | 2.975666 | 2.976002 | 0.000336094 |
| EXP-T118 | shadow | TEST_OOF | 3.381942 | 3.381730 | -0.000212846 |
| EXP-T119 | primary | VAL_OOF | 2.825817 | 2.825880 | 6.32957e-05 |
| EXP-T119 | primary | TEST_OOF | 3.364662 | 3.364802 | 0.000139966 |
| EXP-T119 | shadow | VAL_OOF | 2.869082 | 2.869167 | 8.51478e-05 |
| EXP-T119 | shadow | TEST_OOF | 3.283530 | 3.283612 | 8.20395e-05 |
| EXP-H064 | primary | VAL_OOF | 0.445406 | 0.445413 | 6.95829e-06 |
| EXP-H064 | primary | TEST_OOF | 0.579796 | 0.579783 | -1.31454e-05 |
| EXP-H064 | shadow | VAL_OOF | 0.423780 | 0.423792 | 1.24272e-05 |
| EXP-H064 | shadow | TEST_OOF | 0.519842 | 0.519836 | -6.06937e-06 |
| EXP-H065 | primary | VAL_OOF | 0.472007 | 0.472009 | 2.56668e-06 |
| EXP-H065 | primary | TEST_OOF | 0.543003 | 0.543002 | -8.59484e-07 |
| EXP-H065 | shadow | VAL_OOF | 0.435754 | 0.435701 | -5.31173e-05 |
| EXP-H065 | shadow | TEST_OOF | 0.497096 | 0.497088 | -7.87076e-06 |
| EXP-H078 | primary | VAL_OOF | 0.462607 | 0.462584 | -2.29706e-05 |
| EXP-H078 | primary | TEST_OOF | 0.516870 | 0.516921 | 5.09921e-05 |
| EXP-H078 | shadow | VAL_OOF | 0.447762 | 0.447760 | -2.8111e-06 |
| EXP-H078 | shadow | TEST_OOF | 0.504275 | 0.504322 | 4.7407e-05 |
| EXP-H079 | primary | VAL_OOF | 0.461265 | 0.461276 | 1.14971e-05 |
| EXP-H079 | primary | TEST_OOF | 0.568927 | 0.568890 | -3.75291e-05 |
| EXP-H079 | shadow | VAL_OOF | 0.434060 | 0.434080 | 1.92795e-05 |
| EXP-H079 | shadow | TEST_OOF | 0.502818 | 0.502881 | 6.29131e-05 |

Per-fold detail: `TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.csv`

