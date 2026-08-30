# Canonical freeze

## PSR
| role | representation | model | CV ρ | Pub ρ | Priv ρ |
|------|----------------|-------|------:|------:|-------:|
| P0_best_simple | A2_physchem | ElasticNet | 0.253 | 0.216 | 0.196 |
| P1_BIO_SHORTCUT | C_BIO_SHORTCUT | Lasso | 0.227 | -0.081 | 0.080 |
| P2_ab_PLM | PLM_ablang2_default | Ridge | 0.068 | 0.027 | 0.313 |
| P3_generic_PLM | PLM_esm2_CDR6 | Ridge | 0.225 | 0.322 | 0.279 |
| P4_structure | STR_ESMF_SASA | Ridge | 0.180 | 0.175 | 0.041 |
| P5_fusion | FUSION_esm2_ABB_SURFACE | Ridge | 0.188 | 0.151 | 0.263 |
| P6_nonlinear | PLM_esm2_t33_650M_UR50D | XGBoost | 0.170 | 0.084 | 0.233 |

## HIC
| role | representation | model | CV ρ | Pub ρ | Priv ρ |
|------|----------------|-------|------:|------:|-------:|
| P0_best_simple | B_cdr_descriptors | Ridge | 0.358 | 0.455 | 0.459 |
| P1_BIO_SHORTCUT | C_BIO_SHORTCUT | Lasso | 0.264 | 0.349 | 0.204 |
| P2_ab_PLM | PLM_ablang2_default | Lasso | 0.360 | 0.253 | 0.255 |
| P3_generic_PLM | FUSION_esm2_ABB_SURFACE | Ridge | 0.503 | 0.509 | 0.519 |
| P4_structure | STR_ESMF_SASA | ElasticNet | 0.513 | 0.319 | 0.436 |
| P5_fusion | FUSION_A2_ABB_SURFACE | ElasticNet | 0.471 | 0.321 | 0.527 |
| P6_nonlinear | FUSION_esm2_ABB_SURFACE | XGBoost | 0.424 | 0.518 | 0.511 |

## TmApp
| role | representation | model | CV ρ | Pub ρ | Priv ρ |
|------|----------------|-------|------:|------:|-------:|
| P0_best_simple | B_cdr_descriptors | Ridge | 0.403 | 0.304 | 0.631 |
| P1_BIO_SHORTCUT | C_BIO_SHORTCUT | Lasso | 0.403 | 0.350 | 0.470 |
| P2_ab_PLM | PLM_ablang2_default | Ridge | 0.456 | 0.470 | 0.647 |
| P3_generic_PLM | PLM_esm1b_t33_650M_UR50S | ElasticNet | 0.489 | 0.427 | 0.637 |
| P4_structure | STR_ABB_SASA_RASA | ElasticNet | 0.424 | 0.177 | 0.500 |
| P5_fusion | FUSION_esm2_ABB_SURFACE | Ridge | 0.423 | 0.343 | 0.567 |
| P6_nonlinear | FUSION_esm2_ABB_SURFACE | XGBoost | 0.380 | 0.362 | 0.588 |

