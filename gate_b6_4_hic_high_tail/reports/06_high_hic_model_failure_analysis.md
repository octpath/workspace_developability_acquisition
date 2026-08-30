# 06 Model failures

| id        | role    |    HIC |   pred_NESTED |   pred_ESMFN |   pred_PLM | band_NESTED   | band_ESMFN   | band_PLM   | structure_closer   |   nested_under |   esmfn_under | consensus_severe_failure   |   n_adv_pred_LOW |
|:----------|:--------|-------:|--------------:|-------------:|-----------:|:--------------|:-------------|:-----------|:-------------------|---------------:|--------------:|:---------------------------|-----------------:|
| ADI-45438 | Private | 11.643 |      10.6664  |     10.6734  |   10.6598  | MEDIUM        | MEDIUM       | MEDIUM     | True               |       0.976638 |      0.969626 | False                      |                1 |
| ADI-47161 | Public  | 11.775 |      10.0359  |     10.3936  |    9.69943 | LOW           | LOW          | LOW        | True               |       1.73907  |      1.38137  | True                       |                4 |
| ADI-45499 | Train   | 12.216 |      10.1064  |      9.88819 |   10.4207  | LOW           | LOW          | LOW        | False              |       2.10962  |      2.32781  | True                       |                5 |
| ADI-45497 | Train   | 12.74  |      10.666   |     10.8175  |   10.6029  | MEDIUM        | MEDIUM       | MEDIUM     | True               |       2.07395  |      1.92249  | False                      |                2 |
| ADI-47160 | Private | 11.745 |      10.5149  |     10.886   |   10.1659  | MEDIUM        | MEDIUM       | LOW        | True               |       1.23005  |      0.859005 | True                       |                3 |
| ADI-47265 | Train   | 11.886 |      10.2804  |     10.5644  |    9.88801 | LOW           | MEDIUM       | LOW        | True               |       1.60562  |      1.32156  | True                       |                4 |
| ADI-47163 | Train   | 12.148 |       8.84191 |      8.74294 |    9.01528 | LOW           | LOW          | LOW        | False              |       3.30609  |      3.40506  | True                       |                5 |
| ADI-47126 | Public  | 12.905 |       9.51947 |      9.7831  |    9.27147 | LOW           | LOW          | LOW        | True               |       3.38553  |      3.1219   | True                       |                5 |
| ADI-47162 | Train   | 11.866 |      10.1677  |      9.95113 |   10.206   | LOW           | LOW          | LOW        | False              |       1.69834  |      1.91487  | True                       |                4 |
| ADI-45435 | Train   | 11.905 |      10.5954  |     10.5268  |   10.7977  | MEDIUM        | MEDIUM       | MEDIUM     | False              |       1.30964  |      1.37818  | False                      |                1 |
| ADI-45433 | Private | 12.08  |       9.91276 |      9.43401 |   10.3631  | LOW           | LOW          | LOW        | False              |       2.16724  |      2.64599  | True                       |                5 |
| ADI-45498 | Private | 12.449 |      10.4017  |     10.4159  |   10.3883  | LOW           | LOW          | LOW        | True               |       2.04732  |      2.03308  | True                       |                5 |
| ADI-45486 | Public  | 13.856 |       9.76611 |      9.9623  |    9.58156 | LOW           | LOW          | LOW        | True               |       4.08989  |      3.8937   | True                       |                5 |
