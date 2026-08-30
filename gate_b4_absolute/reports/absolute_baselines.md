# Absolute-value baselines (Train outer CV MAE)

## HIC

| tag          | model      |     mean |       std |   median |
|:-------------|:-----------|---------:|----------:|---------:|
| CONST_MEDIAN | constant   | 0.517388 | 0.0639243 | 0.496818 |
| SEQ_SIMPLE   | Ridge_grid | 0.524712 | 0.0579605 | 0.509093 |
| IMGT_POS_HL  | Ridge_grid | 0.528358 | 0.0717262 | 0.525671 |
| BIO          | Ridge_grid | 0.535055 | 0.0803306 | 0.529831 |
| SEQ_CDR      | Ridge_grid | 0.542063 | 0.0766495 | 0.554689 |
| GERMLINE_REL | Ridge_grid | 0.550427 | 0.0565646 | 0.544957 |
| CONST_MEAN   | constant   | 0.581084 | 0.0559965 | 0.576088 |

## TmApp

| tag          | model      |    mean |      std |   median |
|:-------------|:-----------|--------:|---------:|---------:|
| GERMLINE_REL | Ridge_grid | 3.12631 | 0.389988 |  3.16108 |
| BIO          | Ridge_grid | 3.13065 | 0.455211 |  3.13833 |
| SEQ_CDR      | Ridge_grid | 3.19574 | 0.365707 |  3.15463 |
| IMGT_POS_HL  | Ridge_grid | 3.19887 | 0.443398 |  3.24127 |
| SEQ_SIMPLE   | Ridge_grid | 3.26501 | 0.313975 |  3.2328  |
| CONST_MEDIAN | constant   | 3.43605 | 0.341312 |  3.34848 |
| CONST_MEAN   | constant   | 3.44697 | 0.280988 |  3.41377 |

