# Fab Disulfide Preparation QC

## Topology rule
- Intradomain VH/VL: earliest Cys ≤40 + late Cys nearest canonical (~92 VH / ~88 VL) among Cys ≥70.
- CH1 intradomain: constant offsets 27–83; HL: CH1 terminal Cys (EPKSC +103).
- CL intradomain: κ 27–87 / λ 28–87; HL light terminal Cys.
- κ and λ audited separately; topology from sequence/domain, **not** nearest-neighbor alone.

## Raw ESMFold SG–SG (all bonds in topology table)
```
                       count      mean       std       min       25%       50%       75%        max
light_locus bond                                                                                   
kappa       CH1_intra  238.0  3.516954  0.260738  2.554292  3.556697  3.589778  3.622426   3.733237
            CL_intra   238.0  2.834084  0.045048  2.711639  2.802543  2.834494  2.862345   3.011364
            HL_inter   238.0  2.960389  3.410686  2.178667  2.656210  2.742692  2.856263  55.280897
            VH_intra   238.0  2.157817  0.187666  1.902592  2.054503  2.110074  2.182408   2.858254
            VL_intra   236.0  2.050443  0.086883  1.856084  1.975748  2.064116  2.121781   2.227769
lambda      CH1_intra   86.0  3.457154  0.040934  3.311502  3.428128  3.464577  3.487850   3.521790
            CL_intra    86.0  2.363653  0.032578  2.293202  2.342821  2.363614  2.385195   2.480054
            HL_inter    86.0  3.296124  0.312167  2.323959  3.115294  3.340756  3.516931   3.907422
            VH_intra    86.0  2.194461  0.246722  1.890707  2.050998  2.113016  2.234706   2.972837
            VL_intra    86.0  2.007282  0.149394  1.745446  1.915121  1.992314  2.066228   2.798401
```

## After OpenMM restrained prep
- n=324 ok=323 fail=1
- HL_SG_SG_after: median=2.049 q90=2.057
- backbone_RMSD: median=0.026 q90=0.028
- severe_clash_after: median=0.000 q90=0.000
- kappa HL after: median=2.049 ok=237/238
- lambda HL after: median=2.055 ok=86/86

## Interpretation note
Proximity / SSBOND records alone are insufficient; this prep defines bonds and relaxes local geometry.
Do not hard-code biology from a single cutoff — distributions are primary.
