# MAE leaderboard-transfer audit

| target   | comparison     |   spearman |   kendall |   top3_overlap |   n_models |
|:---------|:---------------|-----------:|----------:|---------------:|-----------:|
| HIC      | CV→Public      |   0.716667 |  0.5      |              1 |          9 |
| HIC      | Public→Private |   0.9      |  0.722222 |              3 |          9 |
| HIC      | CV→Private     |   0.816667 |  0.666667 |              1 |          9 |
| TmApp    | CV→Public      |   0.335664 |  0.242424 |              0 |         12 |
| TmApp    | Public→Private |   0.027972 |  0        |              0 |         12 |
| TmApp    | CV→Private     |   0.72028  |  0.515152 |              2 |         12 |

Lower MAE is better. TmApp Public→Private under MAE is near-zero on the expanded finalist set; CV→Private remains solid. Do not change the frozen split.
