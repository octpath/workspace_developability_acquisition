# OpenMM Fab MD — 最終レポート（DEV）

解釈: short implicit-solvent classical-MD dynamic descriptors（厳密な溶液MD / Tm直接シミュレーションではない）。

## 冒頭 Q&A

1. **実測 complete wall / Ab？** 91.8 s
2. **最終 production 長？** 0.5 ns
3. **DEV 完走 N？** 158 / 161（fail=3: ['ADI-47103', 'ADI-47163', 'ADI-47316']）
4. **数値安定性？** finite=158/158
5. **TmApp BASE 増分 P/S？** -0.0878 / -0.0390（どちらも悪化）
6. **BioEmu+MPNN 上乗せ P/S？** -0.1041 / -0.0687
7. **HIC 増分 P/S？** -0.0117 / -0.0065（CONT+TITR: -0.0097 / -0.0039）
8. **aromatic/hydrophobic 動的記述子？** HIC でも改善なし（COMP_DROP）
9. **BASE_FIXED_ALPHA？** P=-0.0878 / S=-0.0390（FREEと同じ）
10. **TmApp GO_TO_TEST/STOP？** **STOP**
11. **HIC GO_TO_TEST/STOP？** **STOP**
12. **overall COMP？** TmApp=COMP_DROP / HIC=COMP_DROP

Test 生成は行わない。FeNNix thermal microprobe 37/161 は未再開・保全。

## Simple TVT 表

target                            comparison    fold             mode   N  struct_dim  base_mae  cand_mae     delta status
 TmApp                        BASE+OPENMM_MD Primary       FREE_ALPHA 158          36  2.786507  2.874346 -0.087839     OK
 TmApp                        BASE+OPENMM_MD Primary BASE_FIXED_ALPHA 158          36  2.786507  2.874346 -0.087839     OK
 TmApp                        BASE+OPENMM_MD  Shadow       FREE_ALPHA 158          36  2.916743  2.955769 -0.039026     OK
 TmApp                        BASE+OPENMM_MD  Shadow BASE_FIXED_ALPHA 158          36  2.916743  2.955769 -0.039026     OK
 TmApp BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD Primary       FREE_ALPHA 158          36  2.739172  2.843272 -0.104100     OK
 TmApp BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD Primary BASE_FIXED_ALPHA 158          36  2.739172  2.843272 -0.104100     OK
 TmApp BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD  Shadow       FREE_ALPHA 158          36  2.866889  2.935554 -0.068664     OK
 TmApp BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD  Shadow BASE_FIXED_ALPHA 158          36  2.866889  2.935554 -0.068664     OK
   HIC                 HIC_ARO+OPENMM_MD_HIC Primary       FREE_ALPHA 158          11  0.481176  0.492909 -0.011733     OK
   HIC                 HIC_ARO+OPENMM_MD_HIC Primary BASE_FIXED_ALPHA 158          11  0.481176  0.492909 -0.011733     OK
   HIC                 HIC_ARO+OPENMM_MD_HIC  Shadow       FREE_ALPHA 158          11  0.493563  0.500089 -0.006526     OK
   HIC                 HIC_ARO+OPENMM_MD_HIC  Shadow BASE_FIXED_ALPHA 158          11  0.493563  0.500089 -0.006526     OK
   HIC       HIC_ARO+CONT+TITR+OPENMM_MD_HIC Primary       FREE_ALPHA 158          11  0.465077  0.474739 -0.009662     OK
   HIC       HIC_ARO+CONT+TITR+OPENMM_MD_HIC Primary BASE_FIXED_ALPHA 158          11  0.465077  0.474739 -0.009662     OK
   HIC       HIC_ARO+CONT+TITR+OPENMM_MD_HIC  Shadow       FREE_ALPHA 158          11  0.512495  0.516423 -0.003928     OK
   HIC       HIC_ARO+CONT+TITR+OPENMM_MD_HIC  Shadow BASE_FIXED_ALPHA 158          11  0.512495  0.516423 -0.003928     OK
