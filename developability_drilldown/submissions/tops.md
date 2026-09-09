# T045
ベースの [A] FS_TM_BIOEMU_MPNN は、以前話していたT003そのものの特徴構成です。つまり AbLang2 paired + SEQ_BASIC + BioEmu + ProteinMPNN。これは資料でも明示されています。

そのT003ベースに、classical refinementで得た AbLinguaのCDR3/全CDR pooling と、AbLang2のRASA-aware CDR pooling を追加したのがT045です。したがって「TmAppに効いた既存のglobal情報をかなり広く残しつつ、residue PLMから局所領域を明示的に集約したモデル」と見るのがよいです。

スコアは P=2.7060 / S=2.7289 / mean=2.7175 / worst=2.7289。この worst=2.7289 が非常に強かったため、LINEAR系のrobust CV選択でT045が残った、という理解です。

```
EXP-T045
LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE

Target:
    TmApp

Regressor:
    Ridge regression

Input: 6,649 fixed-length features

    [A] FS_TM_BIOEMU_MPNN
        ├─ AbLang2 paired H+L embedding
        ├─ SEQ_BASIC
        │    ├─ sequence length
        │    ├─ amino-acid composition
        │    └─ CDR-length summaries 等
        ├─ BioEmu NEW_PAIRWISE
        │    └─ VH/VLを別々にBioEmuで生成したensembleからの
        │       Cα RMSD / conformational-variation summary
        └─ ProteinMPNN
             └─ structureに対するnative sequence compatibility score

    [B] FB_AL_CDR3
        └─ AbLingua residue embeddingを
           CDR3領域でpoolした固定長表現

    [C] FB_AL2_RASA_CDR
        └─ AbLang2 residue embeddingを
           CDR領域 + RASA情報を用いてpoolした固定長表現

    [D] FB_AL_CDR_ALL
        └─ AbLingua residue embeddingを
           全CDR領域でpoolした固定長表現

    [A] + [B] + [C] + [D]
              ↓
       fold-local preprocessing
              ↓
            Ridge
              ↓
           TmApp
```

# T048
つまりT048の特徴量はT045と同一です。 feature setだけでなくraw feature contentも同一で、違うのはモデル側です。T045では線形なRidge、T048ではkernel SVRを使って、同じ6,649次元情報から非線形な関係を拾わせています。

結果は P=2.7624 / S=2.9545 / mean=2.8585 / worst=2.9545。Testでは Public=3.0731 / Private=3.0712 と非常に揃っていますが、CVではT045より特にShadowが悪い、という特徴があります。

したがってT045/T048の比較自体が、

同じ特徴量をLinear Ridgeで読むか、SVRで非線形に読むか

というかなり純粋な比較になっています。

```
EXP-T048
SVR_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_SVR

Target:
    TmApp

Regressor:
    Support Vector Regression (SVR)

Input: 6,649 fixed-length features

    [A] FS_TM_BIOEMU_MPNN
        ├─ AbLang2 paired H+L embedding
        ├─ SEQ_BASIC
        ├─ BioEmu NEW_PAIRWISE
        └─ ProteinMPNN compatibility

    [B] FB_AL_CDR3
        └─ AbLingua CDR3 pooled residue representation

    [C] FB_AL2_RASA_CDR
        └─ AbLang2 RASA-aware CDR pooled representation

    [D] FB_AL_CDR_ALL
        └─ AbLingua all-CDR pooled residue representation

    [A] + [B] + [C] + [D]
              ↓
       fold-local preprocessing
              ↓
             SVR
              ↓
           TmApp
```

# H047
元の FS_HIC_HYDRO_TITRATION は、名前から想像するよりかなり盛りだくさんです。ESM-2 Heavy + SEQ_ALL + exposed-aromatic topology + HYDRO_FIELD + TITRATION_SHAPE が全部入っています。過去のclosure資料でも、この完全な構成が明示されています。

さらにH047では FB_ESM2_RASA_CDR3 を追加しています。元blockが1,448特徴、追加blockが1,280特徴で、合計 2,728特徴です。registry上のH047は P=0.45288 / S=0.44398 / mean=0.44843 / worst=0.45288 です。

このモデルの考え方はかなり明快で、

HICのglobalなsequence/PLM情報
＋ 表面疎水性・芳香族性・電荷状態
＋ CDR3で実際に露出している部分のESM-2 residue representation

をSVRに渡している、と整理できます。

```
EXP-H047
SVR_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_SVR

Target:
    HIC retention time

Regressor:
    Support Vector Regression (SVR)

Input: 2,728 fixed-length features

    [A] FS_HIC_HYDRO_TITRATION

        ├─ ESM-2 Heavy-chain embedding
        │    └─ sequence-level PLM representation
        │
        ├─ SEQ_ALL
        │    └─ expanded antibody sequence descriptors
        │
        ├─ AROMATIC_TOPO
        │    └─ ESMFold Fvからの
        │       exposed aromatic / aromatic topology descriptors
        │
        ├─ HYDRO_FIELD
        │    └─ continuous hydrophobic-field
        │       surface descriptors
        │
        └─ TITRATION_SHAPE
             └─ electrostatic / protonation-related
                titration-shape descriptors

    [B] FB_ESM2_RASA_CDR3
        └─ ESM-2 residue representationを
           RASA + CDR3に着目してpoolした
           1,280-dimensional local representation

    [A] + [B]
          ↓
    fold-local preprocessing
          ↓
         SVR
          ↓
    HIC retention time
```

# 3つをまとめると
```
T045:
    [AbLang2 + Seq + BioEmu + ProteinMPNN]
    + [AbLingua CDR3]
    + [AbLang2 RASA×CDR]
    + [AbLingua all-CDR]
    → Ridge
    → TmApp


T048:
    [T045と全く同じ特徴量]
    → SVR
    → TmApp


H047:
    [ESM-2 Heavy + SEQ_ALL
     + aromatic surface topology
     + hydrophobic surface field
     + titration/electrostatics]
    + [ESM-2 RASA×CDR3]
    → SVR
    → HIC
```