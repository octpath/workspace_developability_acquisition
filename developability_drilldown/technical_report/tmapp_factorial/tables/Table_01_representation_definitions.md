# Table 1 — Representation definitions

| Representation           | Domain              | Context        |   Raw dimension | representation_raw   | Notes                                                                            |
|:-------------------------|:--------------------|:---------------|----------------:|:---------------------|:---------------------------------------------------------------------------------|
| Scratch                  | Scratch             | LEARNED_AA     |               0 | scratch              | Learned amino-acid embedding; no frozen PLM.                                     |
| AbLingua                 | Antibody PLM        | SEPARATE_CHAIN |            1280 | ablingua             | Frozen AbLingua 600M residue embeddings.                                         |
| AbLang1                  | Antibody PLM        | SEPARATE_CHAIN |             768 | ablang1              | Official AbLang heavy/light models.                                              |
| AbLang2 (separate-chain) | Antibody PLM        | SEPARATE_CHAIN |             480 | ablang2_paired       | Historical allocation label retains "paired"; actual inference context is SEPARA |
| AbLang2 (paired H/L)     | Antibody PLM        | PAIRED_NATIVE  |             480 | ablang2_unpaired     | Historical allocation label retains "unpaired"; actual inference context is PAIR |
| ESM-1b                   | General protein PLM | SEPARATE_CHAIN |            1280 | esm1b                | facebook/esm1b_t33_650M_UR50S.                                                   |
| ESM-2                    | General protein PLM | SEPARATE_CHAIN |            1280 | esm2                 | facebook/esm2_t33_650M_UR50D.                                                    |
| ESM-C 600M               | General protein PLM | SEPARATE_CHAIN |            1152 | esmc600m             | biohub/ESMC-600M.                                                                |
| CurrAb (separate-chain)  | Antibody PLM        | SEPARATE_CHAIN |            1280 | currab_unpaired      | Same CurrAb revision; SEPARATE_CHAIN H-only / L-only inference.                  |
| CurrAb (paired H/L)      | Antibody PLM        | PAIRED_NATIVE  |            1280 | currab_paired        | brineylab/CurrAb PAIRED_NATIVE; revision 92e2853...                              |
