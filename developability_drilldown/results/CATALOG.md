# Developability Drilldown Catalog

Source: `results/experiments.csv` (single master registry).

**Identifiers:** permanent `experiment_code` (EXP001…) + descriptive `experiment_id`.

**Current evaluation mode:** `POSTCOMP_EXPLORATORY`

Public/Private are post-competition exploratory benchmarks.
They do not rewrite `selection_policy_at_creation`.

## TmApp

| Code | Experiment | family | model_type | feature_set_id / input | CV Primary | CV Shadow | CV worst | Public | Private | Test overall | artifact_status | feature_path | test_prediction_path |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| EXP-T045 | LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE | LINEAR | RIDGE | FS_EXP-T045 | 2.706015 | 2.728948 | 2.728948 | 3.019662 | 3.273609 | 3.146636 | FULL | experiments/features/EXP-T045.parquet | experiments/predictions/EXP-T045/test.csv |
| EXP-T050 | LIN_TM_TM_ABLINGUA_CDR3_AL2_FR_CDR_SPLIT_AL_CDR3_RIDGE | LINEAR | RIDGE | FS_EXP-T050 | 2.723338 | 2.734795 | 2.734795 | 3.032149 | 3.258653 | 3.145401 | FULL | experiments/features/EXP-T050.parquet | experiments/predictions/EXP-T050/test.csv |
| EXP-T001 | LIN_TM_ABLINGUA_CDR3_RIDGE | LINEAR | RIDGE | FS_TM_ABLINGUA_CDR3 | 2.732072 | 2.784957 | 2.784957 | 3.118525 | 3.289918 | 3.204221 | FULL | experiments/features/EXP-T001.parquet | experiments/predictions/EXP-T001/test.csv |
| EXP-T047 | ENET_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_ENET | LINEAR | ENET | FS_EXP-T047 | 2.780803 | 2.820702 | 2.820702 | 3.124540 | 3.199994 | 3.162267 | FULL | experiments/features/EXP-T047.parquet | experiments/predictions/EXP-T047/test.csv |
| EXP-T002 | LIN_TM_ABLINGUA_GLOBAL_RIDGE | LINEAR | RIDGE | FS_TM_ABLINGUA_GLOBAL | 2.746600 | 2.822725 | 2.822725 | 3.106900 | 3.326375 | 3.216637 | FULL | experiments/features/EXP-T002.parquet | experiments/predictions/EXP-T002/test.csv |
| EXP-T003 | LIN_TM_BIOEMU_MPNN_RIDGE | LINEAR | RIDGE | FS_TM_BIOEMU_MPNN | 2.702681 | 2.845937 | 2.845937 | 3.119944 | 3.432660 | 3.276302 | FULL | experiments/features/EXP-T003.parquet | experiments/predictions/EXP-T003/test.csv |
| EXP-T012 | LIN_TM_BASE_BIOEMU_RIDGE | LINEAR | RIDGE | FS_TM_BASE_BIOEMU | 2.728610 | 2.864448 | 2.864448 | 3.134896 | 3.482912 | 3.308904 | FULL | experiments/features/EXP-T012.parquet | experiments/predictions/EXP-T012/test.csv |
| EXP-T046 | LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_LASSO | LINEAR | LASSO | FS_EXP-T046 | 2.866859 | 2.762480 | 2.866859 | 3.202505 | 3.099941 | 3.151223 | FULL | experiments/features/EXP-T046.parquet | experiments/predictions/EXP-T046/test.csv |
| EXP-T016 | LIN_TM_BASE_BIOEMU_MPNN_LASSO | LINEAR | LASSO | FS_TM_BIOEMU_MPNN | 2.869737 | 2.875165 | 2.875165 | 3.215083 | 3.302052 | 3.258568 | FULL | experiments/features/EXP-T016.parquet | experiments/predictions/EXP-T016/test.csv |
| EXP-T014 | LIN_TM_BASE_MPNN_RIDGE | LINEAR | RIDGE | FS_TM_BASE_MPNN | 2.737131 | 2.878130 | 2.878130 | 3.255680 | 3.444757 | 3.350219 | FULL | experiments/features/EXP-T014.parquet | experiments/predictions/EXP-T014/test.csv |
| EXP-T013 | LIN_TM_BASE_BIOEMU_LASSO | LINEAR | LASSO | FS_TM_BASE_BIOEMU | 2.881855 | 2.894073 | 2.894073 | 3.225832 | 3.308701 | 3.267267 | FULL | experiments/features/EXP-T013.parquet | experiments/predictions/EXP-T013/test.csv |
| EXP-T051 | LIN_TM_TM_ABLINGUA_CDR3_AL2_FR_CDR_SPLIT_AL_CDR3_LASSO | LINEAR | LASSO | FS_EXP-T051 | 2.895591 | 2.866209 | 2.895591 | 3.164352 | 3.113100 | 3.138726 | FULL | experiments/features/EXP-T051.parquet | experiments/predictions/EXP-T051/test.csv |
| EXP-T055 | LIN_TM_ALONE_AL2_RASA_P1_RIDGE | LINEAR | RIDGE | FS_EXP-T055 | 2.888409 | 2.896488 | 2.896488 | 3.196108 | 3.170488 | 3.183298 | FULL | experiments/features/EXP-T055.parquet | experiments/predictions/EXP-T055/test.csv |
| EXP-T010 | LIN_TM_ABLANG2_SEQ_RIDGE | LINEAR | RIDGE | FS_TM_ABLANG2_SEQ | 2.762977 | 2.897364 | 2.897364 | 3.266464 | 3.500058 | 3.383261 | FULL | experiments/features/EXP-T010.parquet | experiments/predictions/EXP-T010/test.csv |
| EXP-T019 | LIN_TM_OPENMM_DOMAIN_FLEX_RIDGE | LINEAR | RIDGE | FS_TM_OPENMM_DOMAIN_FLEX | 2.776953 | 2.897763 | 2.897763 | NA | NA | NA | SCORE_ONLY | nan | nan |
| EXP-T057 | ENET_TM_ALONE_AL2_RASA_P1_ENET | LINEAR | ENET | FS_EXP-T057 | 2.902548 | 2.916557 | 2.916557 | 3.202304 | 3.139001 | 3.170653 | FULL | experiments/features/EXP-T057.parquet | experiments/predictions/EXP-T057/test.csv |
| EXP-T056 | LIN_TM_ALONE_AL2_RASA_P1_LASSO | LINEAR | LASSO | FS_EXP-T056 | 2.900148 | 2.930292 | 2.930292 | 3.200297 | 3.181912 | 3.191104 | FULL | experiments/features/EXP-T056.parquet | experiments/predictions/EXP-T056/test.csv |
| EXP-T015 | LIN_TM_BASE_MPNN_LASSO | LINEAR | LASSO | FS_TM_BASE_MPNN | 2.918538 | 2.951121 | 2.951121 | 3.336363 | 3.372658 | 3.354510 | FULL | experiments/features/EXP-T015.parquet | experiments/predictions/EXP-T015/test.csv |
| EXP-T048 | SVR_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_SVR | LINEAR | SVR | FS_EXP-T048 | 2.762417 | 2.954549 | 2.954549 | 3.073132 | 3.071227 | 3.072179 | FULL | experiments/features/EXP-T048.parquet | experiments/predictions/EXP-T048/test.csv |
| EXP-T053 | SVR_TM_TM_ABLINGUA_CDR3_AL2_FR_CDR_SPLIT_AL_CDR3_SVR | LINEAR | SVR | FS_EXP-T053 | 2.756885 | 2.954748 | 2.954748 | 3.103013 | 3.050446 | 3.076730 | FULL | experiments/features/EXP-T053.parquet | experiments/predictions/EXP-T053/test.csv |
| EXP-T011 | LIN_TM_ABLANG2_SEQ_LASSO | LINEAR | LASSO | FS_TM_ABLANG2_SEQ | 2.952645 | 2.974841 | 2.974841 | 3.350227 | 3.409705 | 3.379966 | FULL | experiments/features/EXP-T011.parquet | experiments/predictions/EXP-T011/test.csv |
| EXP-T021 | LIN_TM_FENNIX_CONTEXT_RIDGE | LINEAR | RIDGE | FS_TM_FENNIX_CONTEXT | 2.873122 | 2.992189 | 2.992189 | 3.268517 | 3.615351 | 3.441934 | FULL | experiments/features/EXP-T021.parquet | experiments/predictions/EXP-T021/test.csv |
| EXP-T020 | LIN_TM_OPENMM_DOMAIN_FLEX_LASSO | LINEAR | LASSO | FS_TM_OPENMM_DOMAIN_FLEX | 3.001956 | 2.970638 | 3.001956 | NA | NA | NA | SCORE_ONLY | nan | nan |
| EXP-T022 | LIN_TM_FENNIX_CONTEXT_LASSO | LINEAR | LASSO | FS_TM_FENNIX_CONTEXT | 3.006764 | 3.002007 | 3.006764 | 3.312389 | 3.392163 | 3.352276 | FULL | experiments/features/EXP-T022.parquet | experiments/predictions/EXP-T022/test.csv |
| EXP-T060 | LIN_TM_ALONE_AL2_FR_CDR_SPLIT_RIDGE | LINEAR | RIDGE | FS_EXP-T060 | 3.013536 | 2.884254 | 3.013536 | 3.281927 | 3.188925 | 3.235426 | FULL | experiments/features/EXP-T060.parquet | experiments/predictions/EXP-T060/test.csv |
| EXP-T061 | LIN_TM_ALONE_AL2_FR_CDR_SPLIT_LASSO | LINEAR | LASSO | FS_EXP-T061 | 3.036033 | 2.882948 | 3.036033 | 3.287479 | 3.224279 | 3.255879 | FULL | experiments/features/EXP-T061.parquet | experiments/predictions/EXP-T061/test.csv |
| EXP-T062 | ENET_TM_ALONE_AL2_FR_CDR_SPLIT_ENET | LINEAR | ENET | FS_EXP-T062 | 3.038585 | 2.880553 | 3.038585 | 3.375393 | 3.267841 | 3.321617 | FULL | experiments/features/EXP-T062.parquet | experiments/predictions/EXP-T062/test.csv |
| EXP-T063 | SVR_TM_ALONE_AL2_FR_CDR_SPLIT_SVR | LINEAR | SVR | FS_EXP-T063 | 3.073435 | 2.906945 | 3.073435 | 3.213107 | 3.088348 | 3.150728 | FULL | experiments/features/EXP-T063.parquet | experiments/predictions/EXP-T063/test.csv |
| EXP-T058 | SVR_TM_ALONE_AL2_RASA_P1_SVR | LINEAR | SVR | FS_EXP-T058 | 3.083542 | 3.014411 | 3.083542 | 3.221941 | 3.029190 | 3.125565 | FULL | experiments/features/EXP-T058.parquet | experiments/predictions/EXP-T058/test.csv |
| EXP-T052 | ENET_TM_TM_ABLINGUA_CDR3_AL2_FR_CDR_SPLIT_AL_CDR3_ENET | LINEAR | ENET | FS_EXP-T052 | 2.753466 | 3.096587 | 3.096587 | 3.121153 | 3.151031 | 3.136092 | FULL | experiments/features/EXP-T052.parquet | experiments/predictions/EXP-T052/test.csv |
| EXP-T008 | LIN_TM_ABLANG2_RIDGE | LINEAR | RIDGE | FS_TM_ABLANG2 | 2.851958 | 3.110807 | 3.110807 | 3.492877 | 3.353582 | 3.423229 | FULL | experiments/features/EXP-T008.parquet | experiments/predictions/EXP-T008/test.csv |
| EXP-T006 | LIN_TM_SEQ_BASIC_RIDGE | LINEAR | RIDGE | FS_TM_SEQ_BASIC | 3.141355 | 3.197074 | 3.197074 | 3.526859 | 3.571235 | 3.549047 | FULL | experiments/features/EXP-T006.parquet | experiments/predictions/EXP-T006/test.csv |
| EXP-T017 | LIN_TM_PARENT_ABLINGUA_GLOBAL_LASSO | LINEAR | LASSO | FS_TM_ABLINGUA_GLOBAL | 3.174988 | 3.243738 | 3.243738 | 3.260224 | 3.151212 | 3.205718 | FULL | experiments/features/EXP-T017.parquet | experiments/predictions/EXP-T017/test.csv |
| EXP-T007 | LIN_TM_SEQ_BASIC_LASSO | LINEAR | LASSO | FS_TM_SEQ_BASIC | 3.347852 | 3.213742 | 3.347852 | 3.570181 | 3.533866 | 3.552023 | FULL | experiments/features/EXP-T007.parquet | experiments/predictions/EXP-T007/test.csv |
| EXP-T018 | LIN_TM_PARENT_ABLINGUA_CDR3_LASSO | LINEAR | LASSO | FS_TM_ABLINGUA_CDR3 | 3.231059 | 3.387462 | 3.387462 | 3.395886 | 3.185620 | 3.290753 | FULL | experiments/features/EXP-T018.parquet | experiments/predictions/EXP-T018/test.csv |
| EXP-T009 | LIN_TM_ABLANG2_LASSO | LINEAR | LASSO | FS_TM_ABLANG2 | 3.177920 | 3.436515 | 3.436515 | 3.507543 | 3.500945 | 3.504244 | FULL | experiments/features/EXP-T009.parquet | experiments/predictions/EXP-T009/test.csv |
| EXP-T004 | LIN_TM_CONSTANT_RIDGE | LINEAR | RIDGE | FS_TM_CONSTANT | 3.438272 | 3.438272 | 3.438272 | 3.783951 | 3.771605 | 3.777778 | FULL | experiments/features/EXP-T004.parquet | experiments/predictions/EXP-T004/test.csv |
| EXP-T005 | LIN_TM_CONSTANT_LASSO | LINEAR | LASSO | FS_TM_CONSTANT | 3.438272 | 3.438272 | 3.438272 | 3.783951 | 3.771605 | 3.777778 | FULL | experiments/features/EXP-T005.parquet | experiments/predictions/EXP-T005/test.csv |
| EXP-T037 | TRF_TM_ABLINGUA_FULL_CONCAT_FUS_BIOEMU_MPNN | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_BIOEMU_MPNN | 2.753816 | 2.772998 | 2.772998 | 3.267985 | 3.280770 | 3.274377 | FULL | nan | experiments/predictions/EXP-T037/test.csv |
| EXP-T036 | TRF_TM_ABLINGUA_FULL_CONCAT_FUS_ABLINGUA_GLOBAL | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_GLOBAL | 2.758065 | 2.799007 | 2.799007 | 3.286646 | 3.209072 | 3.247859 | FULL | nan | experiments/predictions/EXP-T036/test.csv |
| EXP-T035 | TRF_TM_ABLINGUA_FULL_CONCAT_FUS_ABLINGUA_CDR3 | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_CDR3 | 2.746870 | 2.823049 | 2.823049 | 3.284899 | 3.188082 | 3.236491 | FULL | nan | experiments/predictions/EXP-T035/test.csv |
| EXP-T044 | TRF_TM_ABLANG2_FULL_MEAN_FUS_BIOEMU_MPNN | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_BIOEMU_MPNN | 2.721084 | 2.834616 | 2.834616 | 3.141636 | 3.236446 | 3.189041 | FULL | nan | experiments/predictions/EXP-T044/test.csv |
| EXP-T065 | TRF_TM_ABLINGUA_RASA_CONT_FULL_CONCAT_FUS_BIOEMU_MPNN | TRANSFORMER | AnnotatedTransformer+Fusion+ContinuousRASA | RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA + FS_TM_BIOEMU_MPNN | 2.814704 | 2.874607 | 2.874607 | 3.222722 | 3.249894 | 3.236308 | FULL | experiments/features/EXP-T065.parquet | experiments/predictions/EXP-T065/test.csv |
| EXP-T033 | TRF_TM_SCRATCH_FULL_MEAN_FUS_ABLINGUA_GLOBAL | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_GLOBAL | 2.726402 | 2.877958 | 2.877958 | 3.191843 | 3.153571 | 3.172707 | FULL | nan | experiments/predictions/EXP-T033/test.csv |
| EXP-T042 | TRF_TM_ABLANG2_FULL_MEAN_FUS_ABLINGUA_CDR3 | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_CDR3 | 2.807186 | 2.884017 | 2.884017 | 3.231555 | 3.109766 | 3.170661 | FULL | nan | experiments/predictions/EXP-T042/test.csv |
| EXP-T032 | TRF_TM_SCRATCH_FULL_MEAN_FUS_ABLINGUA_CDR3 | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_CDR3 | 2.769435 | 2.900662 | 2.900662 | 3.076469 | 3.182338 | 3.129404 | FULL | nan | experiments/predictions/EXP-T032/test.csv |
| EXP-T066 | TRF_TM_ABLINGUA_RASA_POOL_FULL_CONCAT_FUS_BIOEMU_MPNN | TRANSFORMER | AnnotatedTransformer+Fusion+RASAWeightedPool | RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA_POOL + FS_TM_BIOEMU_MPNN | 2.773911 | 2.903064 | 2.903064 | 3.241731 | 3.188504 | 3.215117 | FULL | experiments/features/EXP-T066.parquet | experiments/predictions/EXP-T066/test.csv |
| EXP-T043 | TRF_TM_ABLANG2_FULL_MEAN_FUS_ABLINGUA_GLOBAL | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_ABLINGUA_GLOBAL | 2.723234 | 2.955301 | 2.955301 | 3.235263 | 3.216251 | 3.225757 | FULL | nan | experiments/predictions/EXP-T043/test.csv |
| EXP-T034 | TRF_TM_SCRATCH_FULL_MEAN_FUS_BIOEMU_MPNN | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_TM_BIOEMU_MPNN | 2.752221 | 2.966773 | 2.966773 | 3.116622 | 3.267218 | 3.191920 | FULL | nan | experiments/predictions/EXP-T034/test.csv |
| EXP-T040 | TRF_TM_ABLANG2_FULL_MEAN | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 2.994088 | 3.042828 | 3.042828 | 3.521579 | 3.251830 | 3.386705 | FULL | nan | experiments/predictions/EXP-T040/test.csv |
| EXP-T039 | TRF_TM_ABLANG2_FULL_CONCAT | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.194821 | 3.107918 | 3.194821 | 3.267595 | 3.092979 | 3.180287 | FULL | nan | experiments/predictions/EXP-T039/test.csv |
| EXP-T028 | TRF_TM_SCRATCH_FULL_MEAN | TRANSFORMER | AnnotatedTransformer | SCRATCH_RESIDUE_SEQUENCE | 3.248280 | 3.268499 | 3.268499 | 3.625833 | 3.343123 | 3.484478 | FULL | nan | experiments/predictions/EXP-T028/test.csv |
| EXP-T041 | TRF_TM_ABLANG2_FULL_REGION_GATE | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.099612 | 3.294070 | 3.294070 | 3.756091 | 3.676044 | 3.716067 | FULL | nan | experiments/predictions/EXP-T041/test.csv |
| EXP-T030 | TRF_TM_ABLINGUA_FULL_CONCAT | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.317769 | 3.214610 | 3.317769 | 3.595529 | 3.255967 | 3.425748 | FULL | nan | experiments/predictions/EXP-T030/test.csv |
| EXP-T038 | TRF_TM_ABLANG2_MIN_CONCAT | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.156795 | 3.318618 | 3.318618 | 3.037159 | 3.300404 | 3.168781 | FULL | nan | experiments/predictions/EXP-T038/test.csv |
| EXP-T031 | TRF_TM_ABLINGUA_FULL_MEAN | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.259409 | 3.319854 | 3.319854 | 3.882903 | 3.357094 | 3.619998 | FULL | nan | experiments/predictions/EXP-T031/test.csv |
| EXP-T027 | TRF_TM_SCRATCH_FULL_CONCAT | TRANSFORMER | AnnotatedTransformer | SCRATCH_RESIDUE_SEQUENCE | 3.322197 | 3.249101 | 3.322197 | 3.652687 | 3.399137 | 3.525912 | FULL | nan | experiments/predictions/EXP-T027/test.csv |
| EXP-T026 | TRF_TM_SCRATCH_MIN_CONCAT | TRANSFORMER | AnnotatedTransformer | SCRATCH_RESIDUE_SEQUENCE | 3.355010 | 3.291361 | 3.355010 | 3.689453 | 3.478788 | 3.584121 | FULL | nan | experiments/predictions/EXP-T026/test.csv |
| EXP-T029 | TRF_TM_ABLINGUA_MIN_CONCAT | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 3.436614 | 3.415659 | 3.436614 | 3.534222 | 3.297411 | 3.415817 | FULL | nan | experiments/predictions/EXP-T029/test.csv |
| EXP-T054 | XGB_TM_TM_ABLINGUA_CDR3_AL2_FR_CDR_SPLIT_AL_CDR3_XGB | XGBOOST | XGB | FS_EXP-T054 | 2.956558 | 2.953942 | 2.956558 | 3.525743 | 3.125213 | 3.325478 | FULL | experiments/features/EXP-T054.parquet | experiments/predictions/EXP-T054/test.csv |
| EXP-T049 | XGB_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_XGB | XGBOOST | XGB | FS_EXP-T049 | 2.920465 | 3.037624 | 3.037624 | 3.394615 | 3.160859 | 3.277737 | FULL | experiments/features/EXP-T049.parquet | experiments/predictions/EXP-T049/test.csv |
| EXP-T023 | XGB_TM_BIOEMU_MPNN | XGBOOST | XGBRegressor | FS_TM_BIOEMU_MPNN | 2.926207 | 3.043586 | 3.043586 | 3.497555 | 3.195029 | 3.346292 | FULL | experiments/features/EXP-T023.parquet | experiments/predictions/EXP-T023/test.csv |
| EXP-T024 | XGB_TM_ABLINGUA_GLOBAL | XGBOOST | XGBRegressor | FS_TM_ABLINGUA_GLOBAL | 2.894707 | 3.110550 | 3.110550 | 3.435619 | 3.090113 | 3.262866 | FULL | experiments/features/EXP-T024.parquet | experiments/predictions/EXP-T024/test.csv |
| EXP-T025 | XGB_TM_ABLINGUA_CDR3 | XGBOOST | XGBRegressor | FS_TM_ABLINGUA_CDR3 | 2.929891 | 3.122976 | 3.122976 | 3.467446 | 3.138629 | 3.303037 | FULL | experiments/features/EXP-T025.parquet | experiments/predictions/EXP-T025/test.csv |
| EXP-T064 | XGB_TM_ALONE_AL2_FR_CDR_SPLIT_XGB | XGBOOST | XGB | FS_EXP-T064 | 3.100845 | 3.144547 | 3.144547 | 3.417611 | 3.205078 | 3.311344 | FULL | experiments/features/EXP-T064.parquet | experiments/predictions/EXP-T064/test.csv |
| EXP-T059 | XGB_TM_ALONE_AL2_RASA_P1_XGB | XGBOOST | XGB | FS_EXP-T059 | 3.232337 | 3.414399 | 3.414399 | 3.486772 | 3.323332 | 3.405052 | FULL | experiments/features/EXP-T059.parquet | experiments/predictions/EXP-T059/test.csv |

### BEST (auto)

- **CV Primary:** `EXP-T003` — `LIN_TM_BIOEMU_MPNN_RIDGE` = 2.702681 (LINEAR, FULL)
- **CV Shadow:** `EXP-T045` — `LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE` = 2.728948 (LINEAR, FULL)
- **CV worst:** `EXP-T045` — `LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE` = 2.728948 (LINEAR, FULL)
- **Public (POSTCOMP_EXPLORATORY):** `EXP-T045` — `LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE` = 3.019662 (LINEAR, FULL)
- **Private (POSTCOMP_EXPLORATORY):** `EXP-T058` — `SVR_TM_ALONE_AL2_RASA_P1_SVR` = 3.029190 (LINEAR, FULL)
- **Test overall:** `EXP-T048` — `SVR_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_SVR` = 3.072179 (LINEAR, FULL)

## HIC

| Code | Experiment | family | model_type | feature_set_id / input | CV Primary | CV Shadow | CV worst | Public | Private | Test overall | artifact_status | feature_path | test_prediction_path |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| EXP-H047 | SVR_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_SVR | LINEAR | SVR | FS_EXP-H047 | 0.452878 | 0.443979 | 0.452878 | 0.407578 | 0.447367 | 0.427473 | FULL | experiments/features/EXP-H047.parquet | experiments/predictions/EXP-H047/test.csv |
| EXP-H052 | SVR_HIC_HIC_CONTINUOUS_SURFACE_ESM2_RASA_CDR3_SVR | LINEAR | SVR | FS_EXP-H052 | 0.453879 | 0.445981 | 0.453879 | 0.414822 | 0.450783 | 0.432803 | FULL | experiments/features/EXP-H052.parquet | experiments/predictions/EXP-H052/test.csv |
| EXP-H042 | SVR_HIC_ALONE_ESM2_RASA_P1_SVR | LINEAR | SVR | FS_EXP-H042 | 0.455465 | 0.463681 | 0.463681 | 0.458914 | 0.435076 | 0.446995 | FULL | experiments/features/EXP-H042.parquet | experiments/predictions/EXP-H042/test.csv |
| EXP-H037 | SVR_HIC_ALONE_ESM2_GLOBAL_SVR | LINEAR | SVR | FS_EXP-H037 | 0.461065 | 0.468939 | 0.468939 | 0.467529 | 0.454961 | 0.461245 | FULL | experiments/features/EXP-H037.parquet | experiments/predictions/EXP-H037/test.csv |
| EXP-H001 | LIN_HIC_HYDRO_TITRATION_LASSO | LINEAR | LASSO | FS_HIC_HYDRO_TITRATION | 0.487223 | 0.483372 | 0.487223 | 0.470238 | 0.464171 | 0.467205 | FULL | experiments/features/EXP-H001.parquet | experiments/predictions/EXP-H001/test.csv |
| EXP-H045 | LIN_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_LASSO | LINEAR | LASSO | FS_EXP-H045 | 0.488417 | 0.486769 | 0.488417 | 0.470238 | 0.464171 | 0.467205 | FULL | experiments/features/EXP-H045.parquet | experiments/predictions/EXP-H045/test.csv |
| EXP-H017 | LIN_HIC_ARO_TITRATION_LASSO | LINEAR | LASSO | FS_HIC_ARO_TITRATION | 0.489388 | 0.482443 | 0.489388 | 0.479935 | 0.475745 | 0.477840 | FULL | experiments/features/EXP-H017.parquet | experiments/predictions/EXP-H017/test.csv |
| EXP-H002 | LIN_HIC_CONTINUOUS_SURFACE_LASSO | LINEAR | LASSO | FS_HIC_CONTINUOUS_SURFACE | 0.482207 | 0.489823 | 0.489823 | 0.479931 | 0.475744 | 0.477838 | FULL | experiments/features/EXP-H002.parquet | experiments/predictions/EXP-H002/test.csv |
| EXP-H019 | LIN_HIC_ARO_CONT_TITR_LASSO | LINEAR | LASSO | FS_HIC_ARO_CONT_TITR | 0.482196 | 0.491237 | 0.491237 | 0.479931 | 0.475744 | 0.477838 | FULL | experiments/features/EXP-H019.parquet | experiments/predictions/EXP-H019/test.csv |
| EXP-H050 | LIN_HIC_HIC_CONTINUOUS_SURFACE_ESM2_RASA_CDR3_LASSO | LINEAR | LASSO | FS_EXP-H050 | 0.487442 | 0.492384 | 0.492384 | 0.479931 | 0.475744 | 0.477838 | FULL | experiments/features/EXP-H050.parquet | experiments/predictions/EXP-H050/test.csv |
| EXP-H003 | LIN_HIC_ESM2_SEQ_AROMATIC_LASSO | LINEAR | LASSO | FS_HIC_ESM2_SEQ_AROMATIC | 0.492730 | 0.480872 | 0.492730 | 0.479935 | 0.475745 | 0.477840 | FULL | experiments/features/EXP-H003.parquet | experiments/predictions/EXP-H003/test.csv |
| EXP-H051 | ENET_HIC_HIC_CONTINUOUS_SURFACE_ESM2_RASA_CDR3_ENET | LINEAR | ENET | FS_EXP-H051 | 0.490879 | 0.493632 | 0.493632 | 0.454452 | 0.476494 | 0.465473 | FULL | experiments/features/EXP-H051.parquet | experiments/predictions/EXP-H051/test.csv |
| EXP-H034 | LIN_HIC_ALONE_ESM2_GLOBAL_RIDGE | LINEAR | RIDGE | FS_EXP-H034 | 0.491436 | 0.496944 | 0.496944 | 0.529712 | 0.527874 | 0.528793 | FULL | experiments/features/EXP-H034.parquet | experiments/predictions/EXP-H034/test.csv |
| EXP-H039 | LIN_HIC_ALONE_ESM2_RASA_P1_RIDGE | LINEAR | RIDGE | FS_EXP-H039 | 0.499256 | 0.499834 | 0.499834 | 0.508834 | 0.508359 | 0.508596 | FULL | experiments/features/EXP-H039.parquet | experiments/predictions/EXP-H039/test.csv |
| EXP-H041 | ENET_HIC_ALONE_ESM2_RASA_P1_ENET | LINEAR | ENET | FS_EXP-H041 | 0.507505 | 0.499092 | 0.507505 | 0.512085 | 0.520826 | 0.516455 | FULL | experiments/features/EXP-H041.parquet | experiments/predictions/EXP-H041/test.csv |
| EXP-H011 | LIN_HIC_ESM2_SEQ_LASSO | LINEAR | LASSO | FS_HIC_ESM2_SEQ | 0.514170 | 0.489558 | 0.514170 | 0.520645 | 0.512726 | 0.516685 | FULL | experiments/features/EXP-H011.parquet | experiments/predictions/EXP-H011/test.csv |
| EXP-H009 | LIN_HIC_ESM2_H_LASSO | LINEAR | LASSO | FS_HIC_ESM2_H | 0.514455 | 0.488831 | 0.514455 | 0.530253 | 0.516931 | 0.523592 | FULL | experiments/features/EXP-H009.parquet | experiments/predictions/EXP-H009/test.csv |
| EXP-H036 | ENET_HIC_ALONE_ESM2_GLOBAL_ENET | LINEAR | ENET | FS_EXP-H036 | 0.515488 | 0.502142 | 0.515488 | 0.531408 | 0.514264 | 0.522836 | FULL | experiments/features/EXP-H036.parquet | experiments/predictions/EXP-H036/test.csv |
| EXP-H040 | LIN_HIC_ALONE_ESM2_RASA_P1_LASSO | LINEAR | LASSO | FS_EXP-H040 | 0.516795 | 0.507608 | 0.516795 | 0.512257 | 0.521294 | 0.516775 | FULL | experiments/features/EXP-H040.parquet | experiments/predictions/EXP-H040/test.csv |
| EXP-H004 | LIN_HIC_CONSTANT_RIDGE | LINEAR | RIDGE | FS_HIC_CONSTANT | 0.518167 | 0.517648 | 0.518167 | 0.534864 | 0.510086 | 0.522475 | FULL | experiments/features/EXP-H004.parquet | experiments/predictions/EXP-H004/test.csv |
| EXP-H005 | LIN_HIC_CONSTANT_LASSO | LINEAR | LASSO | FS_HIC_CONSTANT | 0.518167 | 0.517648 | 0.518167 | 0.534864 | 0.510086 | 0.522475 | FULL | experiments/features/EXP-H005.parquet | experiments/predictions/EXP-H005/test.csv |
| EXP-H035 | LIN_HIC_ALONE_ESM2_GLOBAL_LASSO | LINEAR | LASSO | FS_EXP-H035 | 0.518779 | 0.496145 | 0.518779 | 0.535102 | 0.514200 | 0.524651 | FULL | experiments/features/EXP-H035.parquet | experiments/predictions/EXP-H035/test.csv |
| EXP-H006 | LIN_HIC_SEQ_RIDGE | LINEAR | RIDGE | FS_HIC_SEQ | 0.529402 | 0.526773 | 0.529402 | 0.573304 | 0.561627 | 0.567466 | FULL | experiments/features/EXP-H006.parquet | experiments/predictions/EXP-H006/test.csv |
| EXP-H008 | LIN_HIC_ESM2_H_RIDGE | LINEAR | RIDGE | FS_HIC_ESM2_H | 0.522909 | 0.530752 | 0.530752 | 0.553509 | 0.587292 | 0.570400 | FULL | experiments/features/EXP-H008.parquet | experiments/predictions/EXP-H008/test.csv |
| EXP-H012 | LIN_HIC_AROMATIC_RIDGE | LINEAR | RIDGE | FS_HIC_AROMATIC | 0.521095 | 0.532752 | 0.532752 | 0.491456 | 0.453386 | 0.472421 | FULL | experiments/features/EXP-H012.parquet | experiments/predictions/EXP-H012/test.csv |
| EXP-H046 | ENET_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_ENET | LINEAR | ENET | FS_EXP-H046 | 0.496233 | 0.534979 | 0.534979 | 0.440416 | 0.462014 | 0.451215 | FULL | experiments/features/EXP-H046.parquet | experiments/predictions/EXP-H046/test.csv |
| EXP-H044 | LIN_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_RIDGE | LINEAR | RIDGE | FS_EXP-H044 | 0.512534 | 0.548593 | 0.548593 | 0.499432 | 0.560445 | 0.529939 | FULL | experiments/features/EXP-H044.parquet | experiments/predictions/EXP-H044/test.csv |
| EXP-H020 | LIN_HIC_HYDRO_TITRATION_RIDGE | LINEAR | RIDGE | FS_HIC_HYDRO_TITRATION | 0.519669 | 0.550351 | 0.550351 | 0.498162 | 0.570045 | 0.534103 | FULL | experiments/features/EXP-H020.parquet | experiments/predictions/EXP-H020/test.csv |
| EXP-H016 | LIN_HIC_ARO_TITRATION_RIDGE | LINEAR | RIDGE | FS_HIC_ARO_TITRATION | 0.526166 | 0.550387 | 0.550387 | 0.520177 | 0.583141 | 0.551659 | FULL | experiments/features/EXP-H016.parquet | experiments/predictions/EXP-H016/test.csv |
| EXP-H018 | LIN_HIC_ARO_CONT_TITR_RIDGE | LINEAR | RIDGE | FS_HIC_ARO_CONT_TITR | 0.506992 | 0.552385 | 0.552385 | 0.537270 | 0.587467 | 0.562368 | FULL | experiments/features/EXP-H018.parquet | experiments/predictions/EXP-H018/test.csv |
| EXP-H013 | LIN_HIC_AROMATIC_LASSO | LINEAR | LASSO | FS_HIC_AROMATIC | 0.515178 | 0.553591 | 0.553591 | 0.506990 | 0.468754 | 0.487872 | FULL | experiments/features/EXP-H013.parquet | experiments/predictions/EXP-H013/test.csv |
| EXP-H007 | LIN_HIC_SEQ_LASSO | LINEAR | LASSO | FS_HIC_SEQ | 0.554047 | 0.529794 | 0.554047 | 0.573675 | 0.535802 | 0.554739 | FULL | experiments/features/EXP-H007.parquet | experiments/predictions/EXP-H007/test.csv |
| EXP-H049 | LIN_HIC_HIC_CONTINUOUS_SURFACE_ESM2_RASA_CDR3_RIDGE | LINEAR | RIDGE | FS_EXP-H049 | 0.509320 | 0.557088 | 0.557088 | 0.542739 | 0.580463 | 0.561601 | FULL | experiments/features/EXP-H049.parquet | experiments/predictions/EXP-H049/test.csv |
| EXP-H015 | LIN_HIC_ARO_CONTINUOUS_SURFACE_RIDGE | LINEAR | RIDGE | FS_HIC_CONTINUOUS_SURFACE | 0.511759 | 0.557291 | 0.557291 | 0.546346 | 0.597409 | 0.571878 | FULL | experiments/features/EXP-H015.parquet | experiments/predictions/EXP-H015/test.csv |
| EXP-H014 | LIN_HIC_ESM2_SEQ_AROMATIC_RIDGE | LINEAR | RIDGE | FS_HIC_ESM2_SEQ_AROMATIC | 0.528387 | 0.559056 | 0.559056 | 0.527881 | 0.591482 | 0.559681 | FULL | experiments/features/EXP-H014.parquet | experiments/predictions/EXP-H014/test.csv |
| EXP-H010 | LIN_HIC_ESM2_SEQ_RIDGE | LINEAR | RIDGE | FS_HIC_ESM2_SEQ | 0.534534 | 0.559061 | 0.559061 | 0.546196 | 0.610156 | 0.578176 | FULL | experiments/features/EXP-H010.parquet | experiments/predictions/EXP-H010/test.csv |
| EXP-H030 | TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_ESM2_SEQ_AROMATIC | 0.442203 | 0.441328 | 0.442203 | 0.397606 | 0.480034 | 0.438820 | FULL | nan | experiments/predictions/EXP-H030/test.csv |
| EXP-H033 | TRF_HIC_ESM2_FULL_HONLY_FUS_ESM2_SEQ_ARO | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_ESM2_SEQ_AROMATIC | 0.434526 | 0.442401 | 0.442401 | 0.412947 | 0.498045 | 0.455496 | FULL | nan | experiments/predictions/EXP-H033/test.csv |
| EXP-H028 | TRF_HIC_SCRATCH_FULL_HONLY_FUS_HYDRO_TITRATION | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_HYDRO_TITRATION | 0.443477 | 0.450252 | 0.450252 | 0.413165 | 0.471550 | 0.442358 | FULL | nan | experiments/predictions/EXP-H028/test.csv |
| EXP-H031 | TRF_HIC_ESM2_FULL_HONLY_FUS_HYDRO_TITRATION | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_HYDRO_TITRATION | 0.449375 | 0.451044 | 0.451044 | 0.410711 | 0.471676 | 0.441194 | FULL | nan | experiments/predictions/EXP-H031/test.csv |
| EXP-H029 | TRF_HIC_SCRATCH_FULL_HONLY_FUS_CONTINUOUS_SURFACE | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_CONTINUOUS_SURFACE | 0.436769 | 0.460205 | 0.460205 | 0.429135 | 0.477224 | 0.453180 | FULL | nan | experiments/predictions/EXP-H029/test.csv |
| EXP-H032 | TRF_HIC_ESM2_FULL_HONLY_FUS_CONTINUOUS_SURFACE | TRANSFORMER | AnnotatedTransformer+Fusion | RESIDUE_PLUS_FIXED_FEATURES + FS_HIC_CONTINUOUS_SURFACE | 0.447572 | 0.472379 | 0.472379 | 0.429567 | 0.478406 | 0.453987 | FULL | nan | experiments/predictions/EXP-H032/test.csv |
| EXP-H025 | TRF_HIC_SCRATCH_FULL_HONLY | TRANSFORMER | AnnotatedTransformer | SCRATCH_RESIDUE_SEQUENCE | 0.499442 | 0.510955 | 0.510955 | 0.510622 | 0.518422 | 0.514522 | FULL | nan | experiments/predictions/EXP-H025/test.csv |
| EXP-H027 | TRF_HIC_ESM2_FULL_HONLY | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 0.509699 | 0.515568 | 0.515568 | 0.467079 | 0.472960 | 0.470019 | FULL | nan | experiments/predictions/EXP-H027/test.csv |
| EXP-H024 | TRF_HIC_SCRATCH_MIN_HONLY | TRANSFORMER | AnnotatedTransformer | SCRATCH_RESIDUE_SEQUENCE | 0.500896 | 0.516852 | 0.516852 | 0.506715 | 0.494542 | 0.500629 | FULL | nan | experiments/predictions/EXP-H024/test.csv |
| EXP-H026 | TRF_HIC_ESM2_MIN_HONLY | TRANSFORMER | AnnotatedTransformer | FROZEN_RESIDUE_EMBEDDING | 0.528764 | 0.514456 | 0.528764 | 0.454823 | 0.524323 | 0.489573 | FULL | nan | experiments/predictions/EXP-H026/test.csv |
| EXP-H021 | XGB_HIC_CONTINUOUS_SURFACE | XGBOOST | XGBRegressor | FS_HIC_CONTINUOUS_SURFACE | 0.445753 | 0.444893 | 0.445753 | 0.447909 | 0.464281 | 0.456095 | FULL | experiments/features/EXP-H021.parquet | experiments/predictions/EXP-H021/test.csv |
| EXP-H022 | XGB_HIC_HYDRO_TITRATION | XGBOOST | XGBRegressor | FS_HIC_HYDRO_TITRATION | 0.453777 | 0.454693 | 0.454693 | 0.426390 | 0.452537 | 0.439463 | FULL | experiments/features/EXP-H022.parquet | experiments/predictions/EXP-H022/test.csv |
| EXP-H023 | XGB_HIC_ESM2_SEQ_AROMATIC | XGBOOST | XGBRegressor | FS_HIC_ESM2_SEQ_AROMATIC | 0.459521 | 0.460001 | 0.460001 | 0.446864 | 0.456237 | 0.451551 | FULL | experiments/features/EXP-H023.parquet | experiments/predictions/EXP-H023/test.csv |
| EXP-H048 | XGB_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_XGB | XGBOOST | XGB | FS_EXP-H048 | 0.503612 | 0.487956 | 0.503612 | 0.453312 | 0.476752 | 0.465032 | FULL | experiments/features/EXP-H048.parquet | experiments/predictions/EXP-H048/test.csv |
| EXP-H053 | XGB_HIC_HIC_CONTINUOUS_SURFACE_ESM2_RASA_CDR3_XGB | XGBOOST | XGB | FS_EXP-H053 | 0.510794 | 0.494189 | 0.510794 | 0.458893 | 0.483985 | 0.471439 | FULL | experiments/features/EXP-H053.parquet | experiments/predictions/EXP-H053/test.csv |
| EXP-H043 | XGB_HIC_ALONE_ESM2_RASA_P1_XGB | XGBOOST | XGB | FS_EXP-H043 | 0.515773 | 0.510945 | 0.515773 | 0.470772 | 0.481434 | 0.476103 | FULL | experiments/features/EXP-H043.parquet | experiments/predictions/EXP-H043/test.csv |
| EXP-H038 | XGB_HIC_ALONE_ESM2_GLOBAL_XGB | XGBOOST | XGB | FS_EXP-H038 | 0.507808 | 0.529046 | 0.529046 | 0.538695 | 0.493091 | 0.515893 | FULL | experiments/features/EXP-H038.parquet | experiments/predictions/EXP-H038/test.csv |

### BEST (auto)

- **CV Primary:** `EXP-H033` — `TRF_HIC_ESM2_FULL_HONLY_FUS_ESM2_SEQ_ARO` = 0.434526 (TRANSFORMER, FULL)
- **CV Shadow:** `EXP-H030` — `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` = 0.441328 (TRANSFORMER, FULL)
- **CV worst:** `EXP-H030` — `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` = 0.442203 (TRANSFORMER, FULL)
- **Public (POSTCOMP_EXPLORATORY):** `EXP-H030` — `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` = 0.397606 (TRANSFORMER, FULL)
- **Private (POSTCOMP_EXPLORATORY):** `EXP-H042` — `SVR_HIC_ALONE_ESM2_RASA_P1_SVR` = 0.435076 (LINEAR, FULL)
- **Test overall:** `EXP-H047` — `SVR_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_SVR` = 0.427473 (LINEAR, FULL)

