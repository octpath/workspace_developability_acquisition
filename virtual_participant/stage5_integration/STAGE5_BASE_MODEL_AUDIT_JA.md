# Stage 5 — Base Model Audit

Stage1–4 から freeze した base learner pool。Stage5 では hyperparameter 再 Optuna を行わない。

| experiment_id | stage | modality | Primary | Shadow | OOF repro | include |
|---|---|---:|---:|---:|---|---|
| TmApp__SEQ_BASIC__SVROpt | 1 | sequence | 3.1005 | nan | PASS | True |
| TmApp__ablang2__HL_paired__RidgeOpt_PCANone | 2b | plm | 2.8667 | nan | PASS | True |
| TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt | 2 | plm+sequence | 2.7772 | nan | PASS | True |
| TmApp__ESMFold__STRUCT_RASA__ElasticNetOpt | 3 | structure | 3.2425 | nan | OOF_MISMATCH_0.1666 | False |
| TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt | 3 | plm+sequence+structure | 2.7499 | nan | OOF_MISMATCH_0.0733 | False |
| TmApp__ADV_TMAPP_ALL__SVROpt | 4 | advanced_structure | 3.2236 | nan | PASS | True |
| TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt | 4 | incumbent_fusion | 2.7439 | nan | PASS | True |
| HIC__SEQ_PLUS_ANTIBODY__SVROpt | 1 | sequence | 0.4733 | nan | PASS | True |
| HIC__esm2__H__SVROpt | 2 | plm | 0.4554 | nan | PASS | True |
| HIC__FUSION__esm2__H__SEQ_ALL__SVROpt | 2 | plm+sequence | 0.4489 | nan | PASS | True |
| HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt | 3 | structure | 0.4520 | nan | OOF_MISMATCH_0.1119 | False |
| HIC__FUSION_OVERALL__ESMFold__STRUCT_SURFACE_ALL__SVROpt | 3 | plm+sequence+structure | 0.4417 | nan | PASS | True |
| HIC__ADV_SURFACE_PATCH__SVROpt | 4 | advanced_structure | 0.4721 | nan | PASS | True |
| HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt | 4 | incumbent_fusion | 0.4377 | nan | PASS | True |
