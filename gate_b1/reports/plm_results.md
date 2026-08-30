# PLM results

Canonical TRIPLE_CORE best per model family.

## PSR

| rep | model | CV ρ | Pub ρ | Priv ρ |
|-----|-------|------:|------:|-------:|
| PLM_esm2_CDR6 | Ridge | 0.225 | 0.322 | 0.279 |
| PLM_esm2_CDR6 | Lasso | 0.217 | 0.346 | 0.184 |
| PLM_esm2_t33_650M_UR50D | Ridge | 0.189 | 0.147 | 0.259 |
| PLM_esm1b_t33_650M_UR50S | ElasticNet | 0.178 | 0.138 | 0.203 |
| PLM_esm1b_t33_650M_UR50S | Ridge | 0.174 | 0.077 | 0.224 |
| PLM_esm2_t33_650M_UR50D | XGBoost | 0.170 | 0.084 | 0.233 |
| PLM_esm1b_t33_650M_UR50S | Lasso | 0.147 | 0.096 | 0.199 |
| PLM_esm2_CDR6 | ElasticNet | 0.134 | 0.370 | 0.232 |
| PLM_ablang2_default | Ridge | 0.068 | 0.027 | 0.313 |
| PLM_ablang2_default | ElasticNet | 0.057 | 0.009 | 0.248 |
| PLM_esm2_t33_650M_UR50D | ElasticNet | 0.041 | 0.194 | 0.039 |
| PLM_esm2_t33_650M_UR50D | Lasso | 0.038 | 0.105 | 0.058 |
| PLM_ablang2_default | Lasso | 0.025 | -0.007 | 0.312 |

## HIC

| rep | model | CV ρ | Pub ρ | Priv ρ |
|-----|-------|------:|------:|-------:|
| PLM_esm2_t33_650M_UR50D | Lasso | 0.475 | 0.378 | 0.442 |
| PLM_esm2_t33_650M_UR50D | Ridge | 0.444 | 0.434 | 0.526 |
| PLM_esm2_t33_650M_UR50D | ElasticNet | 0.409 | 0.362 | 0.320 |
| PLM_esm2_CDR6 | Ridge | 0.406 | 0.426 | 0.501 |
| PLM_esm2_CDR6 | Lasso | 0.366 | 0.362 | 0.452 |
| PLM_esm1b_t33_650M_UR50S | Ridge | 0.365 | 0.432 | 0.386 |
| PLM_ablang2_default | Lasso | 0.360 | 0.253 | 0.255 |
| PLM_esm2_t33_650M_UR50D | XGBoost | 0.350 | 0.513 | 0.454 |
| PLM_esm2_CDR6 | ElasticNet | 0.332 | 0.370 | 0.474 |
| PLM_esm1b_t33_650M_UR50S | ElasticNet | 0.322 | 0.502 | 0.311 |
| PLM_esm1b_t33_650M_UR50S | Lasso | 0.319 | 0.413 | 0.377 |
| PLM_ablang2_default | Ridge | 0.311 | 0.298 | 0.234 |
| PLM_ablang2_default | ElasticNet | 0.299 | 0.303 | 0.272 |

## TmApp

| rep | model | CV ρ | Pub ρ | Priv ρ |
|-----|-------|------:|------:|-------:|
| PLM_esm1b_t33_650M_UR50S | ElasticNet | 0.489 | 0.427 | 0.637 |
| PLM_esm1b_t33_650M_UR50S | Ridge | 0.488 | 0.442 | 0.656 |
| PLM_ablang2_default | Ridge | 0.456 | 0.470 | 0.647 |
| PLM_esm2_CDR6 | Ridge | 0.437 | 0.432 | 0.605 |
| PLM_esm2_t33_650M_UR50D | ElasticNet | 0.425 | 0.427 | 0.544 |
| PLM_esm2_t33_650M_UR50D | Ridge | 0.424 | 0.325 | 0.558 |
| PLM_ablang2_default | ElasticNet | 0.420 | 0.462 | 0.581 |
| PLM_esm2_CDR6 | ElasticNet | 0.416 | 0.427 | 0.545 |
| PLM_esm2_t33_650M_UR50D | XGBoost | 0.375 | 0.308 | 0.566 |
| PLM_esm1b_t33_650M_UR50S | Lasso | 0.351 | 0.393 | 0.598 |
| PLM_esm2_CDR6 | Lasso | 0.351 | 0.273 | 0.504 |
| PLM_ablang2_default | Lasso | 0.330 | 0.253 | 0.561 |
| PLM_esm2_t33_650M_UR50D | Lasso | 0.299 | 0.343 | 0.459 |

## Notes

- AbLang original skipped (download hang).
- AbLang2 + ESM-1b + ESM-2 + ESM-2 CDR6 completed for all 400 sequences.

