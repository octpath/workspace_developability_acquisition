# OpenMM MD Family Ablation — 最終レポート

全36特徴ブロックは有害だったため、事前凍結した family 単位で最終確認。新規 MD / FeNNix 再開なし。

## 冒頭 Q&A

1. **凍結 family？** GLOBAL_DYNAMICS, DOMAIN_FLEXIBILITY, INTERFACE_DYNAMICS, AROMATIC_EXPOSURE, HYDROPHOBIC_EXPOSURE
2. **TmApp BASE で Primary∩Shadow 改善？** YES
3. **BioEmu+MPNN で Primary∩Shadow 改善？** NO
4. **AROMATIC_EXPOSURE が HIC で両方改善？** NO（P=-0.0088 / S=-0.0054）
5. **HYDROPHOBIC_EXPOSURE が HIC で両方改善？** NO（P=-0.0030 / S=-0.0008）
6. **正の結果が BASE_FIXED_ALPHA 生存？** YES
7. **Test に進める family？** organizer確認待ち: [{"family": "GLOBAL_DYNAMICS", "setting": "TmApp_BASE", "primary_delta": 0.002735881899964, "shadow_delta": 0.0063020058051979, "verdict": "FAMILY_WEAK"}, {"family": "DOMAIN_FLEXIBILITY", "setting": "TmApp_BASE", "primary_delta": 0.0095537959172449, "shadow_delta": 0.018980159438529, "verdict": "FAMILY_WEAK"}]
8. **最終 OpenMM 判定？** **OPENMM_ONE_FAMILY_SURVIVES**

## Summary

              family  n_features  tm_base_P  tm_base_S  tm_base_fixed_P  tm_base_fixed_S tm_base_verdict  tm_recipe_P  tm_recipe_S  tm_recipe_fixed_P  tm_recipe_fixed_S tm_recipe_verdict     hic_P     hic_S  hic_fixed_P  hic_fixed_S hic_verdict
     GLOBAL_DYNAMICS           5   0.002736   0.006302         0.002736         0.006302     FAMILY_WEAK    -0.008167    -0.001208          -0.008167          -0.001208       FAMILY_DROP -0.001402 -0.003970    -0.001402    -0.003970 FAMILY_DROP
  DOMAIN_FLEXIBILITY           8   0.009554   0.018980         0.009554         0.018980     FAMILY_WEAK    -0.017467    -0.005066          -0.017467          -0.005066       FAMILY_DROP -0.000462 -0.001445    -0.000462    -0.001445 FAMILY_DROP
  INTERFACE_DYNAMICS          12  -0.087554  -0.040785        -0.087554        -0.040785     FAMILY_DROP    -0.082359    -0.044885          -0.082359          -0.044885       FAMILY_DROP -0.001041 -0.001632    -0.001041    -0.001632 FAMILY_DROP
   AROMATIC_EXPOSURE           7   0.007204  -0.003823         0.007204        -0.003823    FAMILY_MIXED     0.007116    -0.010730           0.007116          -0.010730      FAMILY_MIXED -0.008845 -0.005443    -0.008845    -0.005443 FAMILY_DROP
HYDROPHOBIC_EXPOSURE           4  -0.009120  -0.024714        -0.009120        -0.024714     FAMILY_DROP    -0.004791    -0.025414          -0.004791          -0.025414       FAMILY_DROP -0.003018 -0.000780    -0.003018    -0.000780 FAMILY_DROP

## Survivors / bootstrap

{
  "survivors": [
    {
      "family": "GLOBAL_DYNAMICS",
      "setting": "TmApp_BASE",
      "primary_delta": 0.002735881899964,
      "shadow_delta": 0.0063020058051979,
      "verdict": "FAMILY_WEAK"
    },
    {
      "family": "DOMAIN_FLEXIBILITY",
      "setting": "TmApp_BASE",
      "primary_delta": 0.0095537959172449,
      "shadow_delta": 0.018980159438529,
      "verdict": "FAMILY_WEAK"
    }
  ],
  "bootstrap": {
    "GLOBAL_DYNAMICS|TmApp_BASE|Primary": {
      "ci95": [
        -0.04302775939064853,
        0.04683387548763023
      ],
      "p_improve": 0.4534
    },
    "GLOBAL_DYNAMICS|TmApp_BASE|Shadow": {
      "ci95": [
        -0.039635053049323694,
        0.05213073371203777
      ],
      "p_improve": 0.3915
    },
    "DOMAIN_FLEXIBILITY|TmApp_BASE|Primary": {
      "ci95": [
        -0.06679383568779132,
        0.08651941572579343
      ],
      "p_improve": 0.403
    },
    "DOMAIN_FLEXIBILITY|TmApp_BASE|Shadow": {
      "ci95": [
        -0.056246200876510465,
        0.0959042043251932
      ],
      "p_improve": 0.3184
    }
  }
}

