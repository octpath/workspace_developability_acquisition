# AbLingua Token → Residue Mapping Audit

## Sanity (previous HL vs HL+SEQ PCA32)

- Feature matrices **differ**: HL=2560, HL+SEQ=2638; all SEQ_BASIC columns present.
- Fold-local PCA32 + Ridge MAE identical to ~1e-15.
- **Verdict: `SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT`**
  (PCA32 dominated by AbLingua; SEQ adds no measurable OOF change).

## TripleAA algorithm (official BioTokenizer)

1. Append `>` head and `<` tail to AA sequence.
2. Sliding window of length 3 → token strings.
3. Token count = AA length L.
4. Token `t` covers padded positions `[t, t+3)`.
5. Residue `r` (0-based) sits at padded index `r+1`.
6. Residue embedding = mean of hidden states of all **non-special** tokens
   whose span contains `r` (overlapping TripleAA reweighted at residue level).

- Special token IDs excluded: [0, 1, 2, 3, 4]
- Mapping audit: **648/648 chains OK** (324 Abs × H/L).
- Coverage per residue: min=2 max=3 mean=2.983
- CDR source: `cdr_sequence_index_imgt.csv` (IMGT via ANARCI), 324/324 exact AA match.
- RASA: ESMFold Shrake–Rupley + Tien2013 MaxASA; exposed threshold **RASA ≥ 0.2**
  (feature_extension participant default).

If this file exists without UNRESOLVED: mapping is frozen.
