# Organizer Model Postmortem（Ridge / Lasso Endgame）

CV 順位は **Public/Private 評価前**に凍結（`CV_SELECTED_RECIPES_FREEZE.json`）。


## TmApp

| Rank(CV) | Feature recipe | Regressor | Preprocess | CV Primary | CV Shadow | Public | Private | Notes |
|----------|----------------|-----------|------------|------------|-----------|--------|---------|-------|
| 1 | `TM_PARENT_ABLINGUA_CDR3` | RIDGE | recipe_raw + PCA32_per_abl_block + Ridge | 2.7321 | 2.7850 | 3.1185 | 3.2899 | TOP3 |
| 2 | `TM_PARENT_ABLINGUA_GLOBAL` | RIDGE | recipe_raw + PCA32(AbLingua) + Ridge | 2.7466 | 2.8227 | 3.1069 | 3.3264 | TOP3 |
| 3 | `TM_BASE_BIOEMU_MPNN` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7027 | 2.8459 | 3.1199 | 3.4327 | TOP3 |
| 4 | `TM_BASE_BIOEMU` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7286 | 2.8644 | 3.1349 | 3.4829 |  |
| 5 | `TM_BASE_BIOEMU_MPNN` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.8697 | 2.8752 | 3.2151 | 3.3021 |  |
| 6 | `TM_BASE_MPNN` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7371 | 2.8781 | 3.2557 | 3.4448 |  |
| 7 | `TM_BASE_BIOEMU` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.8819 | 2.8941 | 3.2258 | 3.3087 |  |
| 8 | `TM_ABLANG2_SEQ` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7630 | 2.8974 | 3.2665 | 3.5001 |  |
| 9 | `TM_OPENMM_DOMAIN_FLEX` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7770 | 2.8978 | NA | NA | CV_ONLY_NO_TEST_FEATURES |
| 10 | `TM_BASE_MPNN` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.9185 | 2.9511 | 3.3364 | 3.3727 |  |
| 11 | `TM_ABLANG2_SEQ` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.9526 | 2.9748 | 3.3502 | 3.4097 |  |
| 12 | `TM_FENNIX_CONTEXT` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.8731 | 2.9922 | 3.2685 | 3.6154 |  |
| 13 | `TM_OPENMM_DOMAIN_FLEX` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.0020 | 2.9706 | NA | NA | CV_ONLY_NO_TEST_FEATURES |
| 14 | `TM_FENNIX_CONTEXT` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.0068 | 3.0020 | 3.3124 | 3.3922 |  |
| 15 | `TM_ABLANG2` | RIDGE | impute+StandardScaler+Ridge(raw) | 2.8520 | 3.1108 | 3.4929 | 3.3536 |  |
| 16 | `TM_SEQ_BASIC` | RIDGE | impute+StandardScaler+Ridge(raw) | 3.1414 | 3.1971 | 3.5269 | 3.5712 |  |
| 17 | `TM_PARENT_ABLINGUA_GLOBAL` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.1750 | 3.2437 | 3.2602 | 3.1512 |  |
| 18 | `TM_SEQ_BASIC` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.3479 | 3.2137 | 3.5702 | 3.5339 |  |
| 19 | `TM_PARENT_ABLINGUA_CDR3` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.2311 | 3.3875 | 3.3959 | 3.1856 |  |
| 20 | `TM_ABLANG2` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.1779 | 3.4365 | 3.5075 | 3.5009 |  |
| 21 | `TM_CONSTANT` | RIDGE | fold_tv_median | 3.4383 | 3.4383 | 3.7840 | 3.7716 |  |
| 22 | `TM_CONSTANT` | LASSO | fold_tv_median | 3.4383 | 3.4383 | 3.7840 | 3.7716 |  |

## HIC

| Rank(CV) | Feature recipe | Regressor | Preprocess | CV Primary | CV Shadow | Public | Private | Notes |
|----------|----------------|-----------|------------|------------|-----------|--------|---------|-------|
| 1 | `HIC_HYDRO_TITRATION` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4872 | 0.4834 | 0.4702 | 0.4642 | TOP3 |
| 2 | `HIC_ARO_TITRATION` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4894 | 0.4824 | 0.4799 | 0.4757 | TOP3 |
| 3 | `HIC_ARO_CONTINUOUS_SURFACE` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4822 | 0.4898 | 0.4799 | 0.4757 | TOP3 |
| 4 | `HIC_ARO_CONT_TITR` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4822 | 0.4912 | 0.4799 | 0.4757 |  |
| 5 | `HIC_ESM2_SEQ_AROMATIC` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4927 | 0.4809 | 0.4799 | 0.4757 |  |
| 6 | `HIC_ESM2_SEQ` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5142 | 0.4896 | 0.5206 | 0.5127 |  |
| 7 | `HIC_ESM2_H` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5145 | 0.4888 | 0.5303 | 0.5169 |  |
| 8 | `HIC_CONSTANT` | RIDGE | fold_tv_median | 0.5182 | 0.5176 | 0.5349 | 0.5101 |  |
| 9 | `HIC_CONSTANT` | LASSO | fold_tv_median | 0.5182 | 0.5176 | 0.5349 | 0.5101 |  |
| 10 | `HIC_SEQ` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5294 | 0.5268 | 0.5733 | 0.5616 |  |
| 11 | `HIC_ESM2_H` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5229 | 0.5308 | 0.5535 | 0.5873 |  |
| 12 | `HIC_AROMATIC` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5211 | 0.5328 | 0.4915 | 0.4534 |  |
| 13 | `HIC_HYDRO_TITRATION` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5197 | 0.5504 | 0.4982 | 0.5700 |  |
| 14 | `HIC_ARO_TITRATION` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5262 | 0.5504 | 0.5202 | 0.5831 |  |
| 15 | `HIC_ARO_CONT_TITR` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5070 | 0.5524 | 0.5373 | 0.5875 |  |
| 16 | `HIC_AROMATIC` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5152 | 0.5536 | 0.5070 | 0.4688 |  |
| 17 | `HIC_SEQ` | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5540 | 0.5298 | 0.5737 | 0.5358 |  |
| 18 | `HIC_ARO_CONTINUOUS_SURFACE` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5118 | 0.5573 | 0.5463 | 0.5974 |  |
| 19 | `HIC_ESM2_SEQ_AROMATIC` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5284 | 0.5591 | 0.5279 | 0.5915 |  |
| 20 | `HIC_ESM2_SEQ` | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5345 | 0.5591 | 0.5462 | 0.6102 |  |
