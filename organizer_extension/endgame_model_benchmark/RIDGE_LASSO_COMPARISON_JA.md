# Ridge vs Lasso

- 同一レシピ比較: Ridge勝ち **12** / Lasso勝ち **7** / 引分 **2**

- 高次元 (raw_dim>=200): Lasso勝ち 7 / Ridge勝ち 9（n=16）

- 低次元 (raw_dim<200): Lasso勝ち 0 / Ridge勝ち 3（n=5）


## パターン

- **TmApp**: 主要コンペレシピでは Ridge が CV worst で優勢（BioEmu+MPNN / AbLingua PARENT / CDR3）。

- **HIC**: 高次元 concat（ESM2+SEQ+ARO±surface）では **Lasso（PCAなし）** が Ridge を明確に上回る。

- Lasso は疎な係数選択が効く高次元 HIC で有利。コンパクトな TmApp 構造増分は Ridge のまま強い。


## 代表比較

| recipe | dim | Ridge worst | Lasso worst | winner | Lasso nz med |
|--------|-----|-------------|-------------|--------|--------------|
| HIC_CONSTANT | 0 | 0.5182 | 0.5182 | TIE | 0.0 |
| HIC_SEQ | 115 | 0.5294 | 0.5540 | RIDGE | 8.0 |
| HIC_ESM2_H | 1280 | 0.5308 | 0.5145 | LASSO | 26.0 |
| HIC_AROMATIC | 19 | 0.5328 | 0.5536 | RIDGE | 9.0 |
| HIC_HYDRO_TITRATION | 1448 | 0.5504 | 0.4872 | LASSO | 26.0 |
| HIC_ARO_TITRATION | 1432 | 0.5504 | 0.4894 | LASSO | 27.0 |
| HIC_ARO_CONT_TITR | 1477 | 0.5524 | 0.4912 | LASSO | 25.0 |
| HIC_ARO_CONTINUOUS_SURFACE | 1459 | 0.5573 | 0.4898 | LASSO | 24.0 |
| HIC_ESM2_SEQ_AROMATIC | 1414 | 0.5591 | 0.4927 | LASSO | 27.0 |
| HIC_ESM2_SEQ | 1395 | 0.5591 | 0.5142 | LASSO | 26.0 |
| TM_PARENT_ABLINGUA_CDR3 | 5689 | 2.7850 | 3.3875 | RIDGE | 63.0 |
| TM_PARENT_ABLINGUA_GLOBAL | 3129 | 2.8227 | 3.2437 | RIDGE | 52.0 |
| TM_BASE_BIOEMU_MPNN | 569 | 2.8459 | 2.8752 | RIDGE | 36.0 |
| TM_BASE_BIOEMU | 568 | 2.8644 | 2.8941 | RIDGE | 36.0 |
| TM_BASE_MPNN | 559 | 2.8781 | 2.9511 | RIDGE | 35.0 |
| TM_ABLANG2_SEQ | 558 | 2.8974 | 2.9748 | RIDGE | 36.0 |
| TM_OPENMM_DOMAIN_FLEX | 566 | 2.8978 | 3.0020 | RIDGE | 36.0 |
| TM_FENNIX_CONTEXT | 650 | 2.9922 | 3.0068 | RIDGE | 43.0 |
| TM_ABLANG2 | 480 | 3.1108 | 3.4365 | RIDGE | 25.0 |
| TM_SEQ_BASIC | 78 | 3.1971 | 3.3479 | RIDGE | 21.0 |
| TM_CONSTANT | 0 | 3.4383 | 3.4383 | TIE | 0.0 |
