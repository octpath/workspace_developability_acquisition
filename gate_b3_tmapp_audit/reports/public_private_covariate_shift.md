# Public vs Private covariate shift

                       feature        type  smd_pub_priv     wass       ks  js_pub_priv  mean_public  mean_private
              A2_HL_charge_ph7  continuous     -0.267491 0.711111 0.123457          NaN          NaN           NaN
                   A2_HL_gravy  continuous      0.128282 0.013798 0.123457          NaN          NaN           NaN
                      A2_HL_pI  continuous     -0.264930 0.340489 0.148148          NaN          NaN           NaN
                    H_CDR3_len  continuous     -0.015431 0.629630 0.098765          NaN          NaN           NaN
                    L_CDR3_len  continuous     -0.162505 0.296296 0.086420          NaN          NaN           NaN
              ORG_kappa_lambda categorical           NaN      NaN      NaN     0.000765          NaN           NaN
PL_anarci_vh_germline_distance  continuous      0.100507 0.008748 0.098765          NaN          NaN           NaN
 PL_combined_germline_distance  continuous      0.020675 0.006420 0.123457          NaN          NaN           NaN
       PL_vh_germline_distance  continuous      0.100507 0.008748 0.098765          NaN          NaN           NaN
       PL_vl_germline_distance  continuous     -0.102616 0.009149 0.111111          NaN          NaN           NaN
                 b_cell_subset categorical           NaN      NaN      NaN     0.009912          NaN           NaN
                  cluster_size  continuous     -0.488280      NaN      NaN          NaN          NaN           NaN
                         donor categorical           NaN      NaN      NaN     0.000000          NaN           NaN
     nearest_train_VH_identity  continuous      0.099083      NaN      NaN          NaN     0.720854      0.713076
                     vh_family categorical           NaN      NaN      NaN     0.013458          NaN           NaN
                        vh_len  continuous     -0.042443 0.641975 0.074074          NaN          NaN           NaN
                     vl_family categorical           NaN      NaN      NaN     0.103439          NaN           NaN
                        vl_len  continuous     -0.108421 0.481481 0.074074          NaN          NaN           NaN

## Membership classifier AUC (diagnostic)

            block  roc_auc
        BIO_basic 0.436519
       SEQ_SIMPLE 0.604176
PLM_ABLANG2_PCA32 0.526749
