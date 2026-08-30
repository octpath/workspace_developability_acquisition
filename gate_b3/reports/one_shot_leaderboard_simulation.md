# One-shot Public/Private simulation

Finalists frozen from Train-CV only; labels opened once.

target                             tag        model            slot  cv_spearman    cv_sd  public_spearman  private_spearman
   HIC                      SEQ_SIMPLE   Ridge_grid simple_sequence     0.336465 0.164499         0.535599          0.294456
   HIC                     IMGT_POS_HL   ElasticNet imgt_positional     0.441544 0.109867         0.225922          0.132440
   HIC                    GERMLINE_REL   ElasticNet    bio_germline     0.403123 0.055540         0.205754          0.473886
   HIC                       PLM_ESM1B   ElasticNet      plm_linear     0.508423 0.160761         0.481791          0.334579
   HIC                 PLM_ESM1B_PCA64        Ridge      plm_latent     0.517099 0.201755         0.481430          0.261459
   HIC                   PLM_ESM2_rank   Ridge_grid   rank_oriented     0.473146 0.142478         0.422304          0.271103
   HIC                 ESMFN_STRUCTURE   ElasticNet       structure     0.579581 0.142946         0.476382          0.541879
   HIC                FUSION_ESM2_IMGT   ElasticNet          fusion     0.516181 0.211288         0.381212          0.251646
   HIC RESID_PLM_ESM2__ESMFN_STRUCTURE  Ridge_stack        residual     0.450453      NaN         0.312940          0.369711
   HIC                   ENSEMBLE_NNLS nonneg_ridge        ensemble     0.570978      NaN         0.312940          0.369711
   HIC                      NGRAM_HL_3   Ridge_grid           ngram     0.336251 0.151597         0.334259          0.236017
   HIC               ENSEMBLE_RANKMEAN    rank_mean        top_fill     0.550364      NaN         0.312940          0.369711
 TmApp                      SEQ_SIMPLE   ElasticNet simple_sequence     0.311370 0.084675         0.429642          0.318680
 TmApp                     IMGT_POS_HL   Ridge_grid imgt_positional     0.455919 0.138356         0.117461          0.418748
 TmApp                     PLM_ABLANG2   ElasticNet      plm_linear     0.476385 0.110718         0.356656          0.603210
 TmApp               PLM_ABLANG2_PCA64      SVR_RBF      plm_latent     0.480014 0.090305         0.384569          0.575647
 TmApp                   SEQ_CDR_gauss   Ridge_grid   rank_oriented     0.387398 0.091516         0.461693          0.313663
 TmApp                 ESMFN_STRUCTURE   Ridge_grid       structure     0.357463 0.094761         0.448262          0.244073
 TmApp                FUSION_ESM2_IMGT   Ridge_grid          fusion     0.361945 0.198864         0.391680          0.466765
 TmApp    RESID_BIO_SHORTCUT__PLM_ESM2  Ridge_stack        residual     0.390721      NaN         0.492601          0.372528
 TmApp                   ENSEMBLE_NNLS nonneg_ridge        ensemble     0.446147      NaN         0.492601          0.372528
 TmApp                      NGRAM_HL_3   Ridge_grid           ngram     0.359560 0.182342         0.281703          0.413052
 TmApp               PLM_ABLANG2_PCA32      SVR_RBF        top_fill     0.476858 0.104471         0.384569          0.575647

Bootstrap CIs: see `metrics/bootstrap_results.csv`.
**No post-Public changes applied.**
