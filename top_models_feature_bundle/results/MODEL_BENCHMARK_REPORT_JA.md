# モデルベンチマーク統合報告

## Executive summary

**TmApp**: 正準 CV 勝者は follow-up 後 `CROSS_FAMILY_EQUAL_MEAN__TmApp__4m__FOLLOWUP`（worst≈2.7348; members=T1+XGB+TMF2-fusion+AL2F3）。Phase-1 勝者 `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m`（worst≈2.7379）から **極小改善**（winner changed YES）。Public/Private は事後のみ。

**HIC**: 変更なし。正準 CV 勝者は `CROSS_FAMILY_EQUAL_MEAN__HIC__2m`（worst=0.4346）。

### AbLang2 position-aware follow-up（要約）

- 詳細: [`ABLANG2_POSITION_AWARE_FOLLOWUP_JA.md`](ABLANG2_POSITION_AWARE_FOLLOWUP_JA.md)
- Best AbLang2 sequence: `AL2F3_FULL_MEAN`（P/S/worst ≈ 2.994 / 3.043 / 3.043）
- Annotation AL2F1→AL2F2: Shadow 改善・Primary は非改善（mixed）
- Region-gate: 性能非改善、weight≈一様（CDR3 突出なし）
- Best AbLang2 fusion: `AL2F3⊕TM_BASE_BIOEMU_MPNN`（worst≈2.835; 旧 TMF2 fusion には未達）
- 科学的判定: **PARTIALLY_SUPPORTED / INCONCLUSIVE**（位置 annotation 再集約だけでは AbLang2 の TmApp 強さを説明しきれない）

## Unified benchmark table

| model | family | Primary | Shadow | worst | Public | Private | submission |
|---|---|---:|---:|---:|---:|---:|---|
| `CROSS_FAMILY_EQUAL_MEAN__HIC__2m` | CROSS_FAMILY_ENSEMBLE | 0.4273 | 0.4346 | 0.4346 | 0.40666442283950605 | 0.45606406790123455 | results/cross_family_ensemble/submission_cross_family_equal_mean.csv |
| `HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO` | TRANSFORMER_FUSION | 0.4422 | 0.4413 | 0.4422 | 0.3976063614833502 | 0.4800338898646977 | advanced_outputs/submissions/submission_best_cv.csv |
| `HICF2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO` | TRANSFORMER_FUSION | 0.4345 | 0.4424 | 0.4424 | 0.4129469598840785 | 0.4980453283992814 | advanced_outputs/submissions/submission_best_cv.csv |
| `XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO` | XGBOOST | 0.4458 | 0.4449 | 0.4458 | 0.4479089538197459 | 0.4642807152830525 | advanced_outputs/submissions/submission_xgboost.csv |
| `HICS2__FUSION__HIC_HYDRO_TITRATION__LASSO` | TRANSFORMER_FUSION | 0.4435 | 0.4503 | 0.4503 | 0.4131651963598934 | 0.4715501337875553 | advanced_outputs/submissions/submission_best_cv.csv |
| `HICF2__FUSION__HIC_HYDRO_TITRATION__LASSO` | TRANSFORMER_FUSION | 0.4494 | 0.4510 | 0.4510 | 0.4107111196635681 | 0.4716763438472039 | advanced_outputs/submissions/submission_best_cv.csv |
| `XGB__HIC_HYDRO_TITRATION__LASSO` | XGBOOST | 0.4538 | 0.4547 | 0.4547 | 0.426389658515836 | 0.4525372655421127 | advanced_outputs/submissions/submission_xgboost.csv |
| `XGB__HIC_ESM2_SEQ_AROMATIC__LASSO` | XGBOOST | 0.4595 | 0.4600 | 0.4600 | 0.4468639876754196 | 0.4562370774068949 | advanced_outputs/submissions/submission_xgboost.csv |
| `HICS2__FUSION__HIC_ARO_CONTINUOUS_SURFACE__LASSO` | TRANSFORMER_FUSION | 0.4368 | 0.4602 | 0.4602 | 0.4291351329662181 | 0.4772241528357988 | advanced_outputs/submissions/submission_best_cv.csv |
| `HICF2__FUSION__HIC_ARO_CONTINUOUS_SURFACE__LASSO` | TRANSFORMER_FUSION | 0.4476 | 0.4724 | 0.4724 | 0.429567148750211 | 0.4784063694329908 | advanced_outputs/submissions/submission_best_cv.csv |
| `LINEAR_ENSEMBLE__H1+H2` | LINEAR_ENSEMBLE | 0.4813 | 0.4782 | 0.4813 | 0.4737108787659602 | 0.4685091035883209 | organizer_extension/top3_ensemble_quickcheck/final_predictions/final_ensemble_submission.csv |
| `HIC_HYDRO_TITRATION__LASSO` | LINEAR | 0.4872 | 0.4834 | 0.4872 | 0.4702379457310565 | 0.4641712708181718 | NA |
| `HIC_ARO_CONTINUOUS_SURFACE__LASSO` | LINEAR | 0.4822 | 0.4898 | 0.4898 | 0.4799310265774686 | 0.4757443667060864 | NA |
| `HIC_ESM2_SEQ_AROMATIC__LASSO` | LINEAR | 0.4927 | 0.4809 | 0.4927 | 0.4799350305277475 | 0.4757445162156721 | NA |
| `HICS2` | TRANSFORMER_SEQUENCE | 0.4994 | 0.5110 | 0.5110 | 0.5106221749576522 | 0.5184216823813356 | advanced_outputs/submissions/submission_scratch_transformer.csv |
| `HICF2` | TRANSFORMER_SEQUENCE | 0.5097 | 0.5156 | 0.5156 | 0.4670790704797816 | 0.4729598936858 | advanced_outputs/submissions/submission_frozen_transformer.csv |
| `HICS1` | TRANSFORMER_SEQUENCE | 0.5009 | 0.5169 | 0.5169 | 0.5067153060347946 | 0.4945421767058194 | advanced_outputs/submissions/submission_scratch_transformer.csv |
| `HICF1` | TRANSFORMER_SEQUENCE | 0.5288 | 0.5145 | 0.5288 | 0.4548230620019229 | 0.524322588791082 | advanced_outputs/submissions/submission_frozen_transformer.csv |
| `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m` | CROSS_FAMILY_ENSEMBLE | 2.6735 | 2.7379 | 2.7379 | 3.1702912171669455 | 3.2605070313875357 | results/cross_family_ensemble/submission_cross_family_equal_mean.csv |
| `TMF2__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE` | TRANSFORMER_FUSION | 2.7538 | 2.7730 | 2.7730 | 3.267984555091387 | 3.2807702900450906 | advanced_outputs/submissions/submission_best_cv.csv |
| `TM_PARENT_ABLINGUA_CDR3__RIDGE` | LINEAR | 2.7321 | 2.7850 | 2.7850 | 3.1185249613955217 | 3.289917689937472 | NA |
| `LINEAR_ENSEMBLE_EQUAL_MEAN__T1+T3` | LINEAR_ENSEMBLE | 2.7039 | 2.7986 | 2.7986 | NA | NA | NA |
| `TMF2__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE` | TRANSFORMER_FUSION | 2.7581 | 2.7990 | 2.7990 | 3.2866464308750483 | 3.2090721601321373 | advanced_outputs/submissions/submission_best_cv.csv |
| `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` | LINEAR | 2.7466 | 2.8227 | 2.8227 | 3.106900023997267 | 3.326374924950283 | NA |
| `TMF2__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE` | TRANSFORMER_FUSION | 2.7469 | 2.8230 | 2.8230 | 3.284899393717448 | 3.188081764880522 | advanced_outputs/submissions/submission_best_cv.csv |
| `TM_BASE_BIOEMU_MPNN__RIDGE` | LINEAR | 2.7027 | 2.8459 | 2.8459 | 3.1199439738928896 | 3.432660341190389 | NA |
| `TMS3__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE` | TRANSFORMER_FUSION | 2.7264 | 2.8780 | 2.8780 | 3.1918426325291764 | 3.1535706696686923 | advanced_outputs/submissions/submission_best_cv.csv |
| `TMS3__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE` | TRANSFORMER_FUSION | 2.7694 | 2.9007 | 2.9007 | 3.076469233006607 | 3.182338196554302 | advanced_outputs/submissions/submission_best_cv.csv |
| `TMS3__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE` | TRANSFORMER_FUSION | 2.7522 | 2.9668 | 2.9668 | 3.116622359664352 | 3.267218318986304 | advanced_outputs/submissions/submission_best_cv.csv |
| `XGB__TM_BASE_BIOEMU_MPNN__RIDGE` | XGBOOST | 2.9262 | 3.0436 | 3.0436 | 3.497555391288098 | 3.195029176311728 | advanced_outputs/submissions/submission_xgboost.csv |
| `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE` | XGBOOST | 2.8947 | 3.1105 | 3.1105 | 3.43561883620274 | 3.0901126626097124 | advanced_outputs/submissions/submission_xgboost.csv |
| `XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE` | XGBOOST | 2.9299 | 3.1230 | 3.1230 | 3.467445561915268 | 3.138628924334491 | advanced_outputs/submissions/submission_xgboost.csv |
| `TMS3` | TRANSFORMER_SEQUENCE | 3.2483 | 3.2685 | 3.2685 | 3.625833158139829 | 3.3431229767975985 | advanced_outputs/submissions/submission_scratch_transformer.csv |
| `TMF2` | TRANSFORMER_SEQUENCE | 3.3178 | 3.2146 | 3.3178 | 3.5955287792064525 | 3.255966657473717 | advanced_outputs/submissions/submission_frozen_transformer.csv |
| `TMF3` | TRANSFORMER_SEQUENCE | 3.2594 | 3.3199 | 3.3199 | 3.882902922453704 | 3.357093905225212 | advanced_outputs/submissions/submission_frozen_transformer.csv |
| `TMS2` | TRANSFORMER_SEQUENCE | 3.3222 | 3.2491 | 3.3222 | 3.6526866959936815 | 3.3991371437355324 | advanced_outputs/submissions/submission_scratch_transformer.csv |
| `TMS1` | TRANSFORMER_SEQUENCE | 3.3550 | 3.2914 | 3.3550 | 3.689453125 | 3.4787879284517267 | advanced_outputs/submissions/submission_scratch_transformer.csv |
| `TMF1` | TRANSFORMER_SEQUENCE | 3.4366 | 3.4157 | 3.4366 | 3.5342223555953414 | 3.29741131817853 | advanced_outputs/submissions/submission_frozen_transformer.csv |

## Best canonical CV model

- **TmApp**: `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m` — Primary 2.673506 / Shadow 2.737890 / worst 2.737890 (family=CROSS_FAMILY_ENSEMBLE).
  - Shadow より Primary が良い（非対称）。
- **HIC**: `CROSS_FAMILY_EQUAL_MEAN__HIC__2m` — Primary 0.427282 / Shadow 0.434573 / worst 0.434573 (family=CROSS_FAMILY_ENSEMBLE).
  - Shadow より Primary が良い（非対称）。

## Best Public model

POSTMORTEM ONLY.

- **TmApp canonical**: `TMS3__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE` Public=3.076469
- **HIC canonical**: `HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO` Public=0.397606

## Best Private model

POSTMORTEM ONLY.

- **TmApp canonical**: `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE` Private=3.090113
- **HIC canonical**: `XGB__HIC_HYDRO_TITRATION__LASSO` Private=0.452537

## Historical all-time Public/Private winners

- `HIST__TmApp_PUBLIC_WINNER` (TmApp): Public=3.0853 Private=3.1600 — cv_comparable=NO
- `HIST__TmApp_PRIVATE_WINNER` (TmApp): Public=3.3456 Private=3.1501 — cv_comparable=NO
- `HIST__HIC_PUBLIC_WINNER` (HIC): Public=0.4148 Private=0.4318 — cv_comparable=NO
- `HIST__HIC_PRIVATE_WINNER` (HIC): Public=0.4237 Private=0.4180 — cv_comparable=NO
- `HIST__HIC_SIMPLE_blend_seq_surf_adv` (HIC): Public=0.4239 Private=0.4222 — cv_comparable=NO

## Linear vs XGBoost

- **HIC**: XGBoost が Top-3 線形を一貫して改善。
- **TmApp**: XGBoost は線形より悪化。

## Sequence Transformer

- scratch/frozen とも単独では線形に未達。full annotation がわずかに有利。

## Fusion Transformer

- 固定長 Top-3 との結合で大きく改善（特に HIC）。系列単独の強さというより相補。

## Ensemble

- **TmApp cross-family equal-mean**: members=['TM_PARENT_ABLINGUA_CDR3__RIDGE', 'TMF2__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE'], worst=2.737890, Pub=3.1702912171669455, Priv=3.2605070313875357; CV winner changed=True
- **HIC cross-family equal-mean**: members=['XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO', 'HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO'], worst=0.434573, Pub=0.40666442283950605, Priv=0.45606406790123455; CV winner changed=True

## Final recommendation

### TmApp
- BEST_CV: `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m`
- BEST_PUBLIC_POSTMORTEM (canonical): `TMS3__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE`
- BEST_PRIVATE_POSTMORTEM (canonical): `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- historical Public/Private (descriptive): `TMS3__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE` / `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- RECOMMENDED_REPRODUCIBLE_MODEL: `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m` (CV only; not PP)

### HIC
- BEST_CV: `CROSS_FAMILY_EQUAL_MEAN__HIC__2m`
- BEST_PUBLIC_POSTMORTEM (canonical): `HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO`
- BEST_PRIVATE_POSTMORTEM (canonical): `XGB__HIC_HYDRO_TITRATION__LASSO`
- historical Public/Private (descriptive): `HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO` / `HIST__HIC_PRIVATE_WINNER`
- RECOMMENDED_REPRODUCIBLE_MODEL: `CROSS_FAMILY_EQUAL_MEAN__HIC__2m` (CV only; not PP)

