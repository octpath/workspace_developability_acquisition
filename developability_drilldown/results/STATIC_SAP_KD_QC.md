# STATIC_SAP_KD QC

- antibodies: 324 / 324
- residue rows: 75057
- structure_scope: **Fv** (ESMFold via crosswalk)
- R_REF: 5.0 Å (SOURCE_SPECIFIED from STATIC-SAP FEATURE_SPEC radii_A[0])
- R_SENSITIVITY: 10.0 Å (QC only; not trained)
- Shrake-Rupley: probe=1.4, n_points=100 (**SOURCE_SPECIFIED** / FEATURE_SPEC + repository)
- MaxASA: Tien2013
- hydrophobicity: Kyte–Doolittle min-max → KD_norm∈[0,1]
- centroid: non-H side-chain arithmetic mean; Gly→CA
- include_self: True
- issues: NONE
- extraction_errors: 0
- artifact_hashes: `{"residue": "d69e41fcae1beb6b372bfa42e654f8379c5d6ec0e48b091baee2a71322dc34d0", "global3": "2ae537df33213e4737f2fc1d0b7d8af3f5e7dbf70a66b75841a8b8742e57d1b1", "chain9": "162b4a91413ffb03eaa944083f8af9eca3df410ffbf1923f4194ad11445fa0b5", "n_antibodies": 324, "n_residue_rows": 75057}`

## Distributions (SAP9)
```
       SSKD_ALL_MAX  SSKD_ALL_MEAN  SSKD_ALL_SUM  SSKD_H_MAX  SSKD_H_MEAN  SSKD_H_SUM  SSKD_L_MAX  SSKD_L_MEAN  SSKD_L_SUM
count    324.000000     324.000000    324.000000  324.000000   324.000000  324.000000  324.000000   324.000000  324.000000
mean       0.882634       0.201189     46.609762    0.832394     0.204424   25.064867    0.789642     0.197476   21.544895
std        0.113364       0.011392      2.838571    0.142443     0.013673    1.951562    0.098666     0.016313    1.905814
min        0.628679       0.166455     38.950410    0.552143     0.154004   18.634482    0.604297     0.156461   17.415994
25%        0.808660       0.192493     44.713373    0.737874     0.195310   23.672377    0.697195     0.185439   20.180603
50%        0.862930       0.201331     46.531981    0.826122     0.204552   24.988066    0.795726     0.195581   21.262442
75%        0.944485       0.208322     48.485873    0.915817     0.213749   26.405491    0.853903     0.207718   22.546588
max        1.279070       0.229183     54.171947    1.279070     0.236048   30.214182    1.126428     0.247692   29.179294
```

## Validity
- mean n_valid: 231.7
- mean n_H / n_L: 122.6 / 109.1
