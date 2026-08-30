# Simple and shortcut baselines

## Canonical TRIPLE_CORE (Dev CV Spearman)

### PSR

| rep | model | CV ρ | Public ρ | Private ρ | class |
|-----|-------|------:|---------:|----------:|-------|
| A2_physchem | ElasticNet | 0.253 | 0.216 | 0.196 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | Lasso | 0.227 | -0.081 | 0.080 | PARTICIPANT_LEGAL |
| A2_physchem | Lasso | 0.217 | 0.217 | 0.142 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | Ridge | 0.206 | -0.148 | 0.043 | PARTICIPANT_LEGAL |
| A2_physchem | Ridge | 0.200 | 0.255 | 0.148 | PARTICIPANT_LEGAL |
| A0_length | Ridge | 0.183 | 0.208 | 0.119 | PARTICIPANT_LEGAL |
| A0_length | ElasticNet | 0.177 | 0.173 | 0.118 | PARTICIPANT_LEGAL |
| ORG_subset_only | ElasticNet | 0.173 | -0.036 | -0.015 | ILLEGAL_FOR_PARTICIPANTS |
| C_BIO_SHORTCUT | ElasticNet | 0.173 | nan | nan | PARTICIPANT_LEGAL |
| A0_length | Lasso | 0.163 | 0.173 | 0.118 | PARTICIPANT_LEGAL |
| ORG_subset_only | Ridge | 0.152 | -0.036 | -0.015 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_descriptors | Ridge | 0.142 | 0.177 | 0.215 | PARTICIPANT_LEGAL |
| ORG_subset_only | Lasso | 0.142 | -0.036 | -0.015 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_plus_germline | Lasso | 0.123 | -0.119 | 0.018 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_descriptors | ElasticNet | 0.120 | nan | nan | PARTICIPANT_LEGAL |
| B_cdr_descriptors | Lasso | 0.117 | 0.203 | 0.195 | PARTICIPANT_LEGAL |
| A1_aa_comp | ElasticNet | 0.116 | -0.147 | 0.166 | PARTICIPANT_LEGAL |
| A1_aa_comp | Lasso | 0.111 | -0.096 | 0.140 | PARTICIPANT_LEGAL |
| A1_aa_comp | Ridge | 0.079 | -0.025 | 0.099 | PARTICIPANT_LEGAL |
| ORG_subset_plus_germline | Ridge | 0.048 | 0.001 | -0.030 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_plus_germline | ElasticNet | 0.047 | -0.026 | 0.050 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_germline_author | Lasso | 0.010 | 0.215 | 0.044 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_hydrophobicity_only | Lasso | -0.026 | nan | nan | PARTICIPANT_LEGAL |
| ORG_germline_author | Ridge | -0.046 | 0.163 | 0.069 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_hydrophobicity_only | ElasticNet | -0.051 | -0.036 | -0.165 | PARTICIPANT_LEGAL |
| B_cdr_hydrophobicity_only | Ridge | -0.100 | -0.093 | -0.089 | PARTICIPANT_LEGAL |
| ORG_germline_author | ElasticNet | -0.229 | 0.244 | 0.015 | ILLEGAL_FOR_PARTICIPANTS |

### HIC

| rep | model | CV ρ | Public ρ | Private ρ | class |
|-----|-------|------:|---------:|----------:|-------|
| B_cdr_descriptors | Ridge | 0.358 | 0.455 | 0.459 | PARTICIPANT_LEGAL |
| A2_physchem | ElasticNet | 0.338 | 0.390 | 0.361 | PARTICIPANT_LEGAL |
| A2_physchem | Lasso | 0.328 | 0.371 | 0.351 | PARTICIPANT_LEGAL |
| A2_physchem | Ridge | 0.302 | 0.362 | 0.390 | PARTICIPANT_LEGAL |
| A0_length | Ridge | 0.287 | 0.331 | 0.244 | PARTICIPANT_LEGAL |
| A0_length | ElasticNet | 0.276 | 0.331 | 0.244 | PARTICIPANT_LEGAL |
| A1_aa_comp | Lasso | 0.269 | 0.258 | 0.365 | PARTICIPANT_LEGAL |
| A1_aa_comp | ElasticNet | 0.264 | 0.277 | 0.375 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | Lasso | 0.264 | 0.349 | 0.204 | PARTICIPANT_LEGAL |
| A0_length | Lasso | 0.264 | 0.331 | 0.244 | PARTICIPANT_LEGAL |
| B_cdr_descriptors | Lasso | 0.260 | 0.411 | 0.517 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | ElasticNet | 0.259 | 0.361 | 0.197 | PARTICIPANT_LEGAL |
| ORG_subset_plus_germline | ElasticNet | 0.258 | 0.231 | 0.404 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_descriptors | ElasticNet | 0.251 | 0.419 | 0.479 | PARTICIPANT_LEGAL |
| ORG_subset_plus_germline | Ridge | 0.245 | 0.219 | 0.388 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_plus_germline | Lasso | 0.220 | 0.289 | 0.373 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_only | Lasso | 0.217 | 0.090 | 0.100 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_only | Ridge | 0.217 | 0.090 | 0.100 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_only | ElasticNet | 0.211 | 0.013 | 0.103 | ILLEGAL_FOR_PARTICIPANTS |
| C_BIO_SHORTCUT | Ridge | 0.194 | 0.257 | 0.283 | PARTICIPANT_LEGAL |
| A1_aa_comp | Ridge | 0.186 | 0.298 | 0.389 | PARTICIPANT_LEGAL |
| ORG_germline_author | Ridge | 0.156 | 0.241 | 0.416 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_germline_author | Lasso | 0.155 | 0.269 | 0.418 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_germline_author | ElasticNet | 0.127 | 0.263 | 0.427 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_hydrophobicity_only | Ridge | 0.049 | 0.124 | 0.172 | PARTICIPANT_LEGAL |
| B_cdr_hydrophobicity_only | ElasticNet | -0.029 | 0.108 | 0.133 | PARTICIPANT_LEGAL |
| B_cdr_hydrophobicity_only | Lasso | -0.041 | 0.125 | 0.157 | PARTICIPANT_LEGAL |

### TmApp

| rep | model | CV ρ | Public ρ | Private ρ | class |
|-----|-------|------:|---------:|----------:|-------|
| B_cdr_descriptors | Ridge | 0.403 | 0.304 | 0.631 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | Lasso | 0.403 | 0.350 | 0.470 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | ElasticNet | 0.369 | 0.357 | 0.472 | PARTICIPANT_LEGAL |
| B_cdr_descriptors | ElasticNet | 0.363 | 0.245 | 0.619 | PARTICIPANT_LEGAL |
| C_BIO_SHORTCUT | Ridge | 0.361 | 0.367 | 0.477 | PARTICIPANT_LEGAL |
| A1_aa_comp | Lasso | 0.348 | 0.272 | 0.615 | PARTICIPANT_LEGAL |
| A1_aa_comp | Ridge | 0.348 | 0.264 | 0.630 | PARTICIPANT_LEGAL |
| A1_aa_comp | ElasticNet | 0.326 | 0.277 | 0.612 | PARTICIPANT_LEGAL |
| ORG_subset_plus_germline | ElasticNet | 0.291 | 0.255 | 0.433 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_plus_germline | Lasso | 0.288 | 0.239 | 0.411 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_germline_author | Lasso | 0.285 | 0.167 | 0.298 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_descriptors | Lasso | 0.280 | 0.032 | 0.511 | PARTICIPANT_LEGAL |
| A2_physchem | ElasticNet | 0.269 | 0.212 | 0.379 | PARTICIPANT_LEGAL |
| ORG_subset_plus_germline | Ridge | 0.260 | 0.203 | 0.402 | ILLEGAL_FOR_PARTICIPANTS |
| A2_physchem | Lasso | 0.252 | 0.245 | 0.350 | PARTICIPANT_LEGAL |
| A2_physchem | Ridge | 0.235 | 0.226 | 0.432 | PARTICIPANT_LEGAL |
| B_cdr_hydrophobicity_only | ElasticNet | 0.200 | -0.028 | 0.028 | PARTICIPANT_LEGAL |
| ORG_germline_author | ElasticNet | 0.199 | 0.174 | 0.315 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_hydrophobicity_only | Ridge | 0.195 | 0.132 | 0.156 | PARTICIPANT_LEGAL |
| ORG_germline_author | Ridge | 0.182 | 0.189 | 0.320 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_only | Lasso | 0.161 | 0.298 | 0.325 | ILLEGAL_FOR_PARTICIPANTS |
| ORG_subset_only | Ridge | 0.147 | 0.327 | 0.319 | ILLEGAL_FOR_PARTICIPANTS |
| B_cdr_hydrophobicity_only | Lasso | 0.145 | -0.028 | 0.028 | PARTICIPANT_LEGAL |
| ORG_subset_only | ElasticNet | 0.054 | nan | nan | ILLEGAL_FOR_PARTICIPANTS |
| A0_length | Lasso | 0.048 | -0.281 | -0.106 | PARTICIPANT_LEGAL |
| A0_length | ElasticNet | 0.045 | -0.281 | -0.106 | PARTICIPANT_LEGAL |
| A0_length | Ridge | 0.038 | -0.260 | -0.082 | PARTICIPANT_LEGAL |

