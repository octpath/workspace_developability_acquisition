# Linear Model Top-10 Tables

CV tables = **FEATURE_LINEAR canonical Simple TVT only**.

Public/Private tables below = **ORGANIZER_LINEAR_OVERALL** (includes historical prediction blends).

See also FEATURE_LEVEL_*_PUBLIC/PRIVATE_TOP10.csv for feature-only postmortem.


## TmApp — CV Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | recipe_raw + PCA32_per_abl_block + Ridge | 2.732072376654289 | 2.784957206877823 | 3.1185249613955217 | 3.289917689937472 | FEATURE_LINEAR |
| 2 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | recipe_raw + PCA32(AbLingua) + Ridge | 2.7466001170280285 | 2.822725206703513 | 3.106900023997267 | 3.326374924950283 | FEATURE_LINEAR |
| 3 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.702681240871287 | 2.845936772811553 | 3.1199439738928896 | 3.432660341190389 | FEATURE_LINEAR |
| 4 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.728609764529513 | 2.864447740706632 | 3.1348956756048434 | 3.482911833404577 | FEATURE_LINEAR |
| 5 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.869736881558336 | 2.875164722539417 | 3.215082849230016 | 3.3020522485399617 | FEATURE_LINEAR |
| 6 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.737131275300181 | 2.878130055093428 | 3.255680141998243 | 3.4447572323707307 | FEATURE_LINEAR |
| 7 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.881855042017764 | 2.894073206694681 | 3.225832442554766 | 3.3087009106950167 | FEATURE_LINEAR |
| 8 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7629769100217336 | 2.8973641403873263 | 3.266464055413186 | 3.5000584900344416 | FEATURE_LINEAR |
| 9 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.7769534001738405 | 2.897762804526902 | nan | nan | FEATURE_LINEAR |
| 10 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.918538205203646 | 2.951121476968849 | 3.336363199164652 | 3.372657686296628 | FEATURE_LINEAR |

## TmApp — Public Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only (no SEQ_BASIC / BioEmu / ProteinMPNN / AbLi… | RidgeOpt | historical Stage pipeline (often SVR bas | 2.9375619787158387 | 3.030556011199951 | 3.0853249349711853 | 3.159989439410928 | HISTORICAL_LINEAR_OTHER |
| 2 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | recipe_raw + PCA32(AbLingua) + Ridge | 2.7466001170280285 | 2.822725206703513 | 3.106900023997267 | 3.326374924950283 | FEATURE_LINEAR |
| 3 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | recipe_raw + PCA32_per_abl_block + Ridge | 2.732072376654289 | 2.784957206877823 | 3.1185249613955217 | 3.289917689937472 | FEATURE_LINEAR |
| 4 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.702681240871287 | 2.845936772811553 | 3.1199439738928896 | 3.432660341190389 | FEATURE_LINEAR |
| 5 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | RIDGE | impute+StandardScaler+Ridge(raw) | 2.728609764529513 | 2.864447740706632 | 3.1348956756048434 | 3.482911833404577 | FEATURE_LINEAR |
| 6 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.869736881558336 | 2.875164722539417 | 3.215082849230016 | 3.3020522485399617 | FEATURE_LINEAR |
| 7 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 2.881855042017764 | 2.894073206694681 | 3.225832442554766 | 3.3087009106950167 | FEATURE_LINEAR |
| 8 | prediction-level meta-learner (convex_mae) over out-of-fold base predictions: Stage-3 incremental fusion ADV_INTERACTION… | convex_mae | historical Stage pipeline (often SVR bas | 2.797917422527372 | nan | 3.2306885841030826 | 3.211552262966718 | PREDICTION_LINEAR_BLEND |
| 9 | prediction-level meta-learner (nnls) over out-of-fold base predictions: Stage-3 incremental fusion ADV_INTERACTIONS desc… | nnls | historical Stage pipeline (often SVR bas | 2.8169388911735287 | nan | 3.2306885841030826 | 3.211552262966718 | PREDICTION_LINEAR_BLEND |
| 10 | prediction-level meta-learner (nnls_norm) over out-of-fold base predictions: Stage-3 incremental fusion ADV_INTERACTIONS… | nnls_norm | historical Stage pipeline (often SVR bas | 2.8127860497083845 | nan | 3.2306885841030826 | 3.211552262966718 | PREDICTION_LINEAR_BLEND |

## TmApp — Private Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | prediction-level meta-learner (convex_mae) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors und… | convex_mae | historical Stage pipeline (often SVR bas | 2.813262920374202 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 2 | prediction-level meta-learner (nnls) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors under sup… | nnls | historical Stage pipeline (often SVR bas | 2.770561793699584 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 3 | prediction-level meta-learner (nnls_norm) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors unde… | nnls_norm | historical Stage pipeline (often SVR bas | 2.765858603133218 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 4 | prediction-level meta-learner (quantile_0.001) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors… | quantile_0.001 | historical Stage pipeline (often SVR bas | 2.761290710767892 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 5 | prediction-level meta-learner (quantile_0.01) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors … | quantile_0.01 | historical Stage pipeline (often SVR bas | 2.7999627900556066 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 6 | prediction-level meta-learner (quantile_0.0) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors u… | quantile_0.0 | historical Stage pipeline (often SVR bas | 2.762350534374748 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 7 | prediction-level meta-learner (quantile_0.1) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors u… | quantile_0.1 | historical Stage pipeline (often SVR bas | 2.8343571167137216 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 8 | prediction-level meta-learner (ridge_100.0) over out-of-fold base predictions: stage-1 SEQ_BASIC sequence descriptors un… | ridge_100.0 | historical Stage pipeline (often SVR bas | 2.743341179228337 | nan | 3.345595945037222 | 3.15007534338225 | PREDICTION_LINEAR_BLEND |
| 9 | AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, comp… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 3.1749884526917262 | 3.243738103777528 | 3.260223703657545 | 3.151212494407123 | FEATURE_LINEAR |
| 10 | Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only (no SEQ_BASIC / BioEmu / ProteinMPNN / AbLi… | RidgeOpt | historical Stage pipeline (often SVR bas | 2.9375619787158387 | 3.030556011199951 | 3.0853249349711853 | 3.159989439410928 | HISTORICAL_LINEAR_OTHER |

## HIC — CV Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4872225781431885 | 0.4833719610538652 | 0.4702379457310565 | 0.4641712708181718 | FEATURE_LINEAR |
| 2 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4893879340185224 | 0.4824432044355602 | 0.4799350305277475 | 0.4757445162156721 | FEATURE_LINEAR |
| 3 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4822073134231601 | 0.4898232886862832 | 0.4799310265774686 | 0.4757443667060864 | FEATURE_LINEAR |
| 4 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4821960095627124 | 0.4912372446404668 | 0.4799310265774686 | 0.4757443667060864 | FEATURE_LINEAR |
| 5 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4927303900206625 | 0.4808724742416768 | 0.4799350305277475 | 0.4757445162156721 | FEATURE_LINEAR |
| 6 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5141703750172416 | 0.489557681348777 | 0.520644653912147 | 0.5127260092304973 | FEATURE_LINEAR |
| 7 | ESM-2 Heavy-chain protein-language-model embedding… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.5144546157057649 | 0.4888306190268648 | 0.5302534821978132 | 0.5169309822146028 | FEATURE_LINEAR |
| 8 | no input features (fold-local median baseline)… | LASSO | fold_tv_median | 0.5181666666666666 | 0.5176481481481482 | 0.5348641975308641 | 0.5100864197530864 | FEATURE_LINEAR |
| 9 | no input features (fold-local median baseline)… | RIDGE | fold_tv_median | 0.5181666666666666 | 0.5176481481481482 | 0.5348641975308641 | 0.5100864197530864 | FEATURE_LINEAR |
| 10 | stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set)… | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5294019506021663 | 0.5267730257826606 | 0.573304245586205 | 0.5616270776725183 | FEATURE_LINEAR |

## HIC — Public Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence desc… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4257721922584362 | 0.4328749669956691 | 0.4148392100594437 | 0.4318145768709808 | PREDICTION_LINEAR_BLEND |
| 2 | Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence desc… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4251776987709104 | 0.4320764671385887 | 0.4204394325858053 | 0.4239298110263242 | PREDICTION_LINEAR_BLEND |
| 3 | Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVRO… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4324716368936545 | nan | 0.4237480344059232 | 0.4180226409189143 | PREDICTION_LINEAR_BLEND |
| 4 | prediction-level meta-learner (ridge_10.0) over out-of-fold base predictions: Stage-3 incremental fusion advanced surfac… | ridge_10.0 | historical Stage pipeline (often SVR bas | 0.4498532320554946 | nan | 0.4266119633976947 | 0.4505211720174822 | PREDICTION_LINEAR_BLEND |
| 5 | prediction-level meta-learner (ridge_1.0) over out-of-fold base predictions: Stage-3 incremental fusion advanced surface… | ridge_1.0 | historical Stage pipeline (often SVR bas | 0.4489343976734367 | nan | 0.4353639440997759 | 0.4667721379568801 | PREDICTION_LINEAR_BLEND |
| 6 | prediction-level meta-learner (ridge_0.1) over out-of-fold base predictions: Stage-3 incremental fusion advanced surface… | ridge_0.1 | historical Stage pipeline (often SVR bas | 0.4546159557904721 | nan | 0.4408757532713614 | 0.4710210242653874 | PREDICTION_LINEAR_BLEND |
| 7 | prediction-level meta-learner (ridge_1.0) over out-of-fold base predictions: antibody-augmented sequence descriptors (SE… | ridge_1.0 | historical Stage pipeline (often SVR bas | 0.4516549498793794 | nan | 0.4413297828399947 | 0.4486261442719902 | PREDICTION_LINEAR_BLEND |
| 8 | prediction-level meta-learner (ridge_10.0) over out-of-fold base predictions: antibody-augmented sequence descriptors (S… | ridge_10.0 | historical Stage pipeline (often SVR bas | 0.45314046379 | nan | 0.441432544528268 | 0.4323884901514161 | PREDICTION_LINEAR_BLEND |
| 9 | prediction-level meta-learner (ridge_0.1) over out-of-fold base predictions: antibody-augmented sequence descriptors (SE… | ridge_0.1 | historical Stage pipeline (often SVR bas | 0.4511556703060195 | nan | 0.4441895843210703 | 0.4556701654233223 | PREDICTION_LINEAR_BLEND |
| 10 | ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence de… | LASSO | impute+StandardScaler+Lasso(no_PCA) | 0.4872225781431885 | 0.4833719610538652 | 0.4702379457310565 | 0.4641712708181718 | FEATURE_LINEAR |

## HIC — Private Top-10

| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |
|------|------------------|-----------|------------|------|------|--------|---------|-------|
| 1 | Equal-mean prediction blend of: ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVRO… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4324716368936545 | nan | 0.4237480344059232 | 0.4180226409189143 | PREDICTION_LINEAR_BLEND |
| 2 | Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence desc… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4251776987709104 | 0.4320764671385887 | 0.4204394325858053 | 0.4239298110263242 | PREDICTION_LINEAR_BLEND |
| 3 | Equal-mean prediction blend of: fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence desc… | equal_mean_blend | historical Stage pipeline (often SVR bas | 0.4257721922584362 | 0.4328749669956691 | 0.4148392100594437 | 0.4318145768709808 | PREDICTION_LINEAR_BLEND |
| 4 | prediction-level meta-learner (ridge_10.0) over out-of-fold base predictions: antibody-augmented sequence descriptors (S… | ridge_10.0 | historical Stage pipeline (often SVR bas | 0.45314046379 | nan | 0.441432544528268 | 0.4323884901514161 | PREDICTION_LINEAR_BLEND |
| 5 | HIC / ADV_SURFACE_PATCH / RidgeOpt… | RidgeOpt | historical Stage pipeline (often SVR bas | 0.5067822562939059 | 0.5141865101906903 | 0.4895884475554022 | 0.4414819401108522 | HISTORICAL_LINEAR_OTHER |
| 6 | HIC / ESMFold / STRUCT_SURFACE_ALL / RidgeOpt… | RidgeOpt | historical Stage pipeline (often SVR bas | 0.5117897847899989 | nan | 0.5166077123854826 | 0.44288511320064 | HISTORICAL_LINEAR_OTHER |
| 7 | prediction-level meta-learner (ridge_1.0) over out-of-fold base predictions: antibody-augmented sequence descriptors (SE… | ridge_1.0 | historical Stage pipeline (often SVR bas | 0.4516549498793794 | nan | 0.4413297828399947 | 0.4486261442719902 | PREDICTION_LINEAR_BLEND |
| 8 | prediction-level meta-learner (ridge_10.0) over out-of-fold base predictions: Stage-3 incremental fusion advanced surfac… | ridge_10.0 | historical Stage pipeline (often SVR bas | 0.4498532320554946 | nan | 0.4266119633976947 | 0.4505211720174822 | PREDICTION_LINEAR_BLEND |
| 9 | AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv)… | RIDGE | impute+StandardScaler+Ridge(raw) | 0.5210947401418343 | 0.5327520861649826 | 0.4914562753657208 | 0.4533861022151522 | FEATURE_LINEAR |
| 10 | prediction-level meta-learner (ridge_0.1) over out-of-fold base predictions: antibody-augmented sequence descriptors (SE… | ridge_0.1 | historical Stage pipeline (often SVR bas | 0.4511556703060195 | nan | 0.4441895843210703 | 0.4556701654233223 | PREDICTION_LINEAR_BLEND |