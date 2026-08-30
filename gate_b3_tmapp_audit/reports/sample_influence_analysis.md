# Sample influence analysis

Base Public→Private model-rank ρ (clean) = **-0.433**

## Top Public influencers (by |Δ transfer ρ|)

  role        id  delta_transfer_rho  transfer_rho_loo  max_abs_delta_model_score  abs_delta
Public ADI-47237            0.216667         -0.216667                   0.016980   0.216667
Public ADI-47248            0.216667         -0.216667                   0.034048   0.216667
Public ADI-47112            0.183333         -0.250000                   0.021656   0.183333
Public ADI-47057            0.116667         -0.316667                   0.013825   0.116667
Public ADI-47179            0.116667         -0.316667                   0.020548   0.116667
Public ADI-45368            0.116667         -0.316667                   0.024299   0.116667
Public ADI-47077            0.116667         -0.316667                   0.010133   0.116667
Public ADI-47093            0.116667         -0.316667                   0.004359   0.116667
Public ADI-47232            0.116667         -0.316667                   0.006008   0.116667
Public ADI-45373            0.116667         -0.316667                   0.014495   0.116667

## Remove top-k Public influencers

 k_removed_public  transfer_rho  delta_vs_base
                1     -0.216667       0.216667
                3      0.000000       0.433333
                5      0.033333       0.466667

Max |Δρ| from single Public deletion: **0.217**
