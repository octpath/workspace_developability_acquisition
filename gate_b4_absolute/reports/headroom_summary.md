# Headroom summary (Train-CV MAE)

## HIC (minutes)
| Stage | MAE |
|---|---|
| Constant median | 0.5174 |
| SEQ_SIMPLE | 0.5225 |
| Best PLM linear | 0.5120 |
| Best PLM nonlinear | 0.4785 |
| Best structure | 0.4781 |
| Best fusion | 0.4892 |
| Best GBDT | 0.4851 |
| Best ensemble | 0.4691 |
| Learned/FT PLM | NOT RUN |

## TmApp (°C)
| Stage | MAE |
|---|---|
| Constant median | 3.4360 |
| SEQ_SIMPLE | 3.2650 |
| BIO | 3.1295 |
| IMGT | 3.2275 |
| Best frozen PLM | 3.0035 |
| Best nonlinear PLM | 2.9995 |
| Best fusion | 2.8172 |
| Best GBDT | 3.1932 |
| Best ensemble | 2.6148 |
| Learned/FT PLM | NOT RUN |

Full table: `metrics/headroom_table.csv`
