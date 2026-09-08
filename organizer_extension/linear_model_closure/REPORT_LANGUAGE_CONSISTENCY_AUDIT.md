# Report Language Consistency Audit

**EN_JA_NUMERIC_CONSISTENCY = PASS**

Compared winner model_ids and key numeric strings between EN and JA winner/closure reports.

| winner_key | field | value | in_EN | in_JA | pass |
|---|---|---|---|---|---|
| tm_cv | model_id | `TM_PARENT_ABLINGUA_CDR3__RIDGE` | True | True | True |
| tm_cv | cv_primary_mae | `2.732072376654289` | True | True | True |
| tm_cv | cv_shadow_mae | `2.784957206877823` | True | True | True |
| tm_cv | public_mae | `3.1185249613955217` | True | True | True |
| tm_cv | private_mae | `3.289917689937472` | True | True | True |
| tm_cv | raw_dimension | `5689.0` | True | True | True |
| tm_cv | effective_dimension | `633.0` | True | True | True |
| tm_cv | final_alpha | `100.0` | True | True | True |
| tm_pub | model_id | `TmApp__ablang2__HL__RidgeOpt__HIST` | True | True | True |
| tm_pub | cv_primary_mae | `2.9375619787158387` | True | True | True |
| tm_pub | cv_shadow_mae | `3.030556011199951` | True | True | True |
| tm_pub | public_mae | `3.0853249349711853` | True | True | True |
| tm_pub | private_mae | `3.159989439410928` | True | True | True |
| tm_pub | N_public | `81` | True | True | True |
| tm_pub | N_private | `81` | True | True | True |
| tm_priv | model_id | `TmApp__META_diversity__convex_mae__HIST` | True | True | True |
| tm_priv | cv_primary_mae | `2.813262920374202` | True | True | True |
| tm_priv | cv_shadow_mae | `nan` | True | True | True |
| tm_priv | public_mae | `3.345595945037222` | True | True | True |
| tm_priv | private_mae | `3.15007534338225` | True | True | True |
| hic_cv | model_id | `HIC_HYDRO_TITRATION__LASSO` | True | True | True |
| hic_cv | cv_primary_mae | `0.4872225781431885` | True | True | True |
| hic_cv | cv_shadow_mae | `0.4833719610538652` | True | True | True |
| hic_cv | public_mae | `0.4702379457310565` | True | True | True |
| hic_cv | private_mae | `0.4641712708181718` | True | True | True |
| hic_pub | model_id | `HIC__SIMPLE_blend_seq_surf__HIST` | True | True | True |
| hic_pub | cv_primary_mae | `0.4257721922584362` | True | True | True |
| hic_pub | cv_shadow_mae | `0.4328749669956691` | True | True | True |
| hic_pub | public_mae | `0.4148392100594437` | True | True | True |
| hic_pub | private_mae | `0.4318145768709808` | True | True | True |
| hic_priv | model_id | `HIC__SIMPLE_blend_esm2_surf__HIST` | True | True | True |
| hic_priv | cv_primary_mae | `0.4324716368936545` | True | True | True |
| hic_priv | cv_shadow_mae | `nan` | True | True | True |
| hic_priv | public_mae | `0.4237480344059232` | True | True | True |
| hic_priv | private_mae | `0.4180226409189143` | True | True | True |

Scores/rankings were not modified by the language split.
