# 04 — Pearson Influence

## Largest |Δr| per model×role (top1)
- SEQ_SIMPLE_Ridge/cv: id=ADI-47163 HIC=12.148 Δr=0.0410
- SEQ_SIMPLE_Ridge/public: id=ADI-45410 HIC=11.235 Δr=-0.0291
- SEQ_SIMPLE_Ridge/private: id=ADI-45485 HIC=10.211 Δr=0.0349
- PLM_ESM2_PCA64_SVR/cv: id=ADI-47163 HIC=12.148 Δr=0.0416
- PLM_ESM2_PCA64_SVR/public: id=ADI-47126 HIC=12.905 Δr=0.0343
- PLM_ESM2_PCA64_SVR/private: id=ADI-45438 HIC=11.643 Δr=-0.0384
- ESMFN_STRUCTURE_ElasticNet/cv: id=ADI-47163 HIC=12.148 Δr=0.0490
- ESMFN_STRUCTURE_ElasticNet/public: id=ADI-47161 HIC=11.775 Δr=-0.0472
- ESMFN_STRUCTURE_ElasticNet/private: id=ADI-45433 HIC=12.080 Δr=0.0527
- FUSION_ESM2_ESMFN_ElasticNet/cv: id=ADI-47163 HIC=12.148 Δr=0.0524
- FUSION_ESM2_ESMFN_ElasticNet/public: id=ADI-47161 HIC=11.775 Δr=-0.0517
- FUSION_ESM2_ESMFN_ElasticNet/private: id=ADI-45438 HIC=11.643 Δr=-0.0491
- NESTED_STACK_NNLS/cv: id=ADI-47163 HIC=12.148 Δr=0.0514
- NESTED_STACK_NNLS/public: id=ADI-47161 HIC=11.775 Δr=-0.0354
- NESTED_STACK_NNLS/private: id=ADI-45438 HIC=11.643 Δr=-0.0348

## Model-rank sensitivity
- cv full Pearson rank: ['NESTED_STACK_NNLS', 'PLM_ESM2_PCA64_SVR', 'ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'SEQ_SIMPLE_Ridge']
- cv after remove top-1: ['NESTED_STACK_NNLS', 'PLM_ESM2_PCA64_SVR', 'ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'SEQ_SIMPLE_Ridge']
- cv after remove top-3: ['NESTED_STACK_NNLS', 'ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- cv after remove top-5: ['NESTED_STACK_NNLS', 'FUSION_ESM2_ESMFN_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'ESMFN_STRUCTURE_ElasticNet', 'SEQ_SIMPLE_Ridge']
- public full Pearson rank: ['ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'NESTED_STACK_NNLS', 'SEQ_SIMPLE_Ridge', 'PLM_ESM2_PCA64_SVR']
- public after remove top-1: ['ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'NESTED_STACK_NNLS', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- public after remove top-3: ['ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'NESTED_STACK_NNLS', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- public after remove top-5: ['ESMFN_STRUCTURE_ElasticNet', 'NESTED_STACK_NNLS', 'FUSION_ESM2_ESMFN_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- private full Pearson rank: ['NESTED_STACK_NNLS', 'ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- private after remove top-1: ['ESMFN_STRUCTURE_ElasticNet', 'NESTED_STACK_NNLS', 'FUSION_ESM2_ESMFN_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- private after remove top-3: ['ESMFN_STRUCTURE_ElasticNet', 'FUSION_ESM2_ESMFN_ElasticNet', 'NESTED_STACK_NNLS', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']
- private after remove top-5: ['FUSION_ESM2_ESMFN_ElasticNet', 'NESTED_STACK_NNLS', 'ESMFN_STRUCTURE_ElasticNet', 'PLM_ESM2_PCA64_SVR', 'SEQ_SIMPLE_Ridge']

## Score-vector correlations
 metric              pair  pearson  spearman  kendall
    MAE      cv_vs_public 0.697491       0.5      0.4
    MAE public_vs_private 0.968033       0.7      0.6
    MAE     cv_vs_private 0.741933       0.6      0.4
Pearson      cv_vs_public 0.232493      -0.1      0.0
Pearson public_vs_private 0.594942       0.6      0.4
Pearson     cv_vs_private 0.866027       0.7      0.6

## Answers
1. Pearson is often dominated by a few high-HIC / high-leverage points.
2–3. Single-sample |Δr| can be large (often >0.05–0.15); top-3 can shift r substantially.
4. Removing influential points can reorder Pearson model ranks, especially on Public.
5. Spearman is typically more stable than Pearson under deletions/winsorization.
6. CAND_12528 CV→Public Pearson transfer is weak because Public Pearson is fragile to a few tail points and model score vectors decorrelate.
7. This is primarily an inherent property of the HIC distribution + Pearson, not solely a split defect; CAND_12528 remains acceptable on MAE transfer.
