# Data dictionary — participant distribution

Full scientific background: `../../participant/README.md`

## Files

### `dev.csv`

| Field | Value |
|---|---|
| Purpose | Labeled development / training set |
| Rows | 162 antibodies |
| Columns | `id`, `heavy`, `light`, `TmApp`, `HIC` |
| Missing values | None permitted for these columns |

### `test_features.csv`

| Field | Value |
|---|---|
| Purpose | Unlabeled test sequences for prediction |
| Rows | 162 antibodies |
| Columns | `id`, `heavy`, `light` |
| Missing values | None permitted |
| Forbidden | Must not contain `TmApp`, `HIC`, Public/Private flags |

### `sample_submission.csv`

| Field | Value |
|---|---|
| Purpose | Submission template with placeholder predictions |
| Rows | 162 (same IDs / order as `test_features.csv`) |
| Columns | `id`, `TmApp`, `HIC` |
| Placeholders | Train-set medians of each target (not Private-informed) |

---

## Columns

### `id`

- Unique antibody identifier (string).  
- Used to join submissions to the hidden solution.

### `heavy`

- Amino-acid sequence of the antibody **heavy-chain variable region (VH)**.  
- Single-letter IUPAC codes; uppercase; no whitespace.  
- Alphabet: `ACDEFGHIKLMNPQRSTVWY`.

### `light`

- Amino-acid sequence of the antibody **light-chain variable region (VL)**.  
- Same encoding rules as `heavy`.

### `TmApp`

- Apparent melting temperature (Fab, DSF) under the study assay conditions.  
- **Unit:** °C  
- Higher ≈ greater apparent thermal / conformational stability in the assay.  
- Not a universal thermodynamic constant; not an aggregation assay.

### `HIC`

- IgG hydrophobic interaction chromatography **retention time**.  
- **Unit:** minutes  
- Higher ≈ stronger effective hydrophobic interaction under the assay protocol.  
- Hydrophobicity-related developability proxy; **not** a direct aggregation measurement.

Optional interpretive bands (diagnostic only; not competition classes):

- LOW: < 10.5 min  
- MEDIUM: 10.5–11.5 min  
- HIGH: > 11.5 min  
