# AbLang2 residue embedding audit

## Model identity

- Package: `ablang2==0.2.1`
- Checkpoint: `ablang2-paired` (`random_init=False`)
- Same family as Stage-2 / bundle AbLang2 seqcoding

## Residue API

Official `rescoding` uses `AbRep(tokens).last_hidden_states` then
`res_to_list(state, formatted_seq)` which keeps **all** formatted tokens
including `<`, `>`, `|`.

This asset keeps **amino-acid positions only**, via exact character-index
mapping of the formatted string (1:1 AA↔token). Special tokens are removed
by index selection, not heuristic pooling.

## QC summary

```json
{
  "n_antibodies": 324,
  "n_heavy": 324,
  "n_light": 324,
  "duplicate_ids": 1,
  "all_residue_eq_seq_len": true,
  "finite_100pct": true,
  "hidden_dim_constant": true,
  "hidden_dim": 480,
  "max_len_H": 140,
  "max_len_L": 120,
  "mapping_statuses": [
    "EXACT_AA_1TO1_SPECIAL_STRIPPED"
  ],
  "unknown_mapping_count": 0,
  "silent_truncation": false,
  "semantic_vs_pooled": {
    "aa_residue_mean_vs_seqcoding": "NOT_SEMANTICALLY_COMPARABLE",
    "reason": "Official seqcoding averages all formatted tokens including <, >, |; this asset stores AA-only residue vectors after exact special-token strip.",
    "fmt_token_mean_vs_seqcoding_max_abs_sample5": 2.0520967719539485e-07,
    "aa_mean_vs_seqcoding_max_abs_sample5": 0.019993249059857887,
    "pool_parquet_feature_cols": 480
  },
  "extraction_wall_time_sec": 4.1766581535339355,
  "device": "cuda:0"
}
```

## Semantic comparison to pooled AbLang2

NOT_SEMANTICALLY_COMPARABLE: Official seqcoding averages all formatted tokens including <, >, |; this asset stores AA-only residue vectors after exact special-token strip.

Fmt-token mean vs seqcoding (sample5 max abs): 2.052e-07

## License

`REVIEW_MODEL_OUTPUT` — do not claim redistribution OK without organizer review.
