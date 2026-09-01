# Round 1 Error Analysis

## TmApp
- worst antibodies: `round1_test_error_diagnostics.csv` 参照（abs_err_TmApp 降順）
- Test MAE 3.2211; Primary CV との差 0.5076

## HIC high-tail
{
  "threshold": 10.5372,
  "N_tail_test": 13,
  "tail_MAE": 2.131598701592744,
  "nontail_MAE": 0.2730411114891062,
  "tail_bias": -2.131598701592744,
  "underpred_count": 13,
  "pred_sd": 0.28545603230708105,
  "true_sd": 0.856929296871939,
  "slope": 0.2035719469186179
}

Dev で観測された 17/17 underprediction パターンが Test tail (13 件) でどの程度再現したかを上記 bias / underpred count で確認。
