# T075–T079 Architecture Comparison (DL_FOLDLOCAL_COSINE_V3)

| Model | H/L base | H-L communication | Summary/readout | Params | VAL_P | VAL_S | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|------|------|------|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| T075 | separate | none | REG_H||REG_L | 501633 | 2.9722 | 3.1202 | 3.5121 | 3.2919 | 3.4020 | 3.5121 | 3.4788 | 3.2874 | 3.3831 |
| T076 | joint | full | single REG | 485121 | 2.9871 | 3.0122 | 3.2431 | 3.4057 | 3.3244 | 3.4057 | 3.5227 | 3.1587 | 3.3407 |
| T077 | joint | full | unrestricted REG_H||REG_L | 501633 | 2.9868 | 3.0242 | 3.3728 | 3.3694 | 3.3711 | 3.3728 | 3.4870 | 3.2859 | 3.3865 |
| T078 | joint | residue full; REG restricted | chain-specific REG_H||REG_L | 501633 | 2.9665 | 3.1036 | 3.4257 | 3.3591 | 3.3924 | 3.4257 | 3.4570 | 3.2416 | 3.3493 |
| T079 | separate | cross-attn bridge | chain-specific REG_H||REG_L | 567683 | 2.9143 | 3.0048 | 3.2363 | 3.2671 | 3.2517 | 3.2671 | 3.5364 | 3.3285 | 3.4324 |

## Paired bootstrap (delta = MAE_first - MAE_second; negative => first better)

- T076_vs_T077 PRIMARY TEST: Δ=-0.1297 [-0.4092, 0.1467]
- T076_vs_T077 SHADOW TEST: Δ=0.0363 [-0.1912, 0.2642]
- T077_vs_T075 PRIMARY TEST: Δ=-0.1394 [-0.3686, 0.0828]
- T077_vs_T075 SHADOW TEST: Δ=0.0775 [-0.1370, 0.2857]
- T078_vs_T077 PRIMARY TEST: Δ=0.0529 [-0.1796, 0.2814]
- T078_vs_T077 SHADOW TEST: Δ=-0.0104 [-0.2388, 0.2152]
- T079_vs_T075 PRIMARY TEST: Δ=-0.2758 [-0.4584, -0.1067]
- T079_vs_T075 SHADOW TEST: Δ=-0.0249 [-0.1610, 0.1169]
- T079_vs_T077 PRIMARY TEST: Δ=-0.1365 [-0.3844, 0.1044]
- T079_vs_T077 SHADOW TEST: Δ=-0.1024 [-0.3114, 0.1154]
- T079_vs_T078 PRIMARY TEST: Δ=-0.1894 [-0.3949, 0.0128]
- T079_vs_T078 SHADOW TEST: Δ=-0.0920 [-0.2637, 0.0662]

## T079 cross-attention gates

- mean(g_H)=0.00270014 median=0.00353664 mean(|g_H|)=0.00491498
- mean(g_L)=-0.00421107 median=-0.0039641 mean(|g_L|)=0.00514899
- sign(g_H)>0 fraction=0.60; sign(g_L)>0 fraction=0.10

## Evaluation policy

- Primary internal metric: TEST_mean
- Robustness: TEST_worst
- External columns above: Primary mean
- Public/Private are post-competition diagnostics; do not override internal verdict.

