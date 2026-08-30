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

### `dev_annotations.csv`

| Field | Value |
|---|---|
| Purpose | Optional **sequence-derived antibody annotations** for Dev IDs |
| Rows | 162 (exact same `id` set as `dev.csv`) |
| Join key | `id` |
| Contents | Germline-family assignments, light-chain type, CDR lengths, germline identity |
| Does **not** contain | `heavy`, `light`, `TmApp`, `HIC`, Public/Private, donor, B-cell origin |
| Scorer | **Not used** by the official scorer |

### `test_annotations.csv`

| Field | Value |
|---|---|
| Purpose | Optional **sequence-derived antibody annotations** for Test IDs |
| Rows | 162 (exact same `id` set as `test_features.csv`) |
| Join key | `id` |
| Same column schema as | `dev_annotations.csv` |
| Does **not** contain | target labels or Public/Private membership |
| Scorer | **Not used** by the official scorer |

These annotation files are convenience resources only. They are inferred from the provided VH/VL sequences (and documented reference annotation procedures) and do **not** add experimental TmApp/HIC information.

---

## Columns

### `id`

- Unique antibody identifier (string).  
- Used to join submissions to the hidden solution.
- Also joins `*_annotations.csv` to `dev.csv` / `test_features.csv`.

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

---

## Annotation columns (`dev_annotations.csv` / `test_annotations.csv`)

Missing-value policy for all distributed annotation columns in this package: **no missing values** (every Dev/Test antibody has a complete annotation row). If a future rebuild encounters an unassignable field, organizers would use empty/NA and document it; that does not apply to v1.0.

### `heavy_v_family`

| | |
|---|---|
| Type | categorical string |
| Definition | Inferred heavy-chain **V-gene family** (e.g. `VH3`) |
| Possible values | `VH1`…`VH7` (observed subset) |
| Method | ANARCI IMGT numbering + germline V assignment; family parsed from assigned V gene |
| Missing | none in v1.0 |

### `heavy_j_gene`

| | |
|---|---|
| Type | categorical string |
| Definition | Inferred heavy-chain **J gene** (allele suffix removed), e.g. `JH4` |
| Possible values | `JH1`…`JH6` (observed subset) |
| Method | ANARCI allele-level J call → gene stem (`IGHJ4*01` → `JH4`). This is a **J gene**, not a broad J-gene family. |
| Missing | none in v1.0 |

### `light_v_family`

| | |
|---|---|
| Type | categorical string |
| Definition | Inferred light-chain **V-gene family** (`VK#` or `VL#`) |
| Possible values | e.g. `VK1`, `VK3`, `VL1`, … |
| Method | ANARCI (same procedure as heavy) |
| Missing | none in v1.0 |

### `light_j_gene`

| | |
|---|---|
| Type | categorical string |
| Definition | Inferred light-chain **J gene** (allele suffix removed): `JK#` or `JL#` |
| Possible values | e.g. `JK1`, `JL2`, … |
| Method | ANARCI allele-level J call → gene stem (`IGKJ1*01` → `JK1`, `IGLJ2*01` → `JL2`). This is a **J gene**, not a broad J-gene family. |
| Missing | none in v1.0 |

### `light_chain_type`

| | |
|---|---|
| Type | categorical string |
| Definition | Light-chain isotype class inferred from sequence |
| Possible values | `kappa`, `lambda` |
| Method | ANARCI chain-type / κ–λ assignment |
| Missing | none in v1.0 |
| Note | Descriptor only — **not** a quality score |

### `h_cdr1_length`, `h_cdr2_length`, `h_cdr3_length`

| | |
|---|---|
| Type | integer |
| Definition | Amino-acid length of heavy-chain CDR1 / CDR2 / CDR3 |
| Unit | residue count |
| Method / convention | **`AUTHOR_MMC2_IMGT_SEGMENTS`**: length of the IMGT CDR segment strings from Shehata mmc2 (`VH CDR1`…`VH CDR3`), after removing alignment gap characters (`-`). Gap-stripped FR/CDR segments concatenate exactly to the distributed `heavy` sequence. |
| Missing | none in v1.0 |
| Note | Participants may also recompute CDR lengths with independent IMGT/ANARCI tools; boundaries can differ slightly by convention |

### `l_cdr1_length`, `l_cdr2_length`, `l_cdr3_length`

| | |
|---|---|
| Type | integer |
| Definition | Amino-acid length of light-chain CDR1 / CDR2 / CDR3 |
| Unit | residue count |
| Method / convention | Same `AUTHOR_MMC2_IMGT_SEGMENTS` scheme as heavy (mmc2 `VL CDR*` segments, gap characters removed) |
| Missing | none in v1.0 |

### `heavy_germline_identity`

| | |
|---|---|
| Type | float |
| Definition | Sequence **identity** (similarity) of the observed VH V region to the ANARCI-assigned germline **V allele/reference** |
| Scale | **0–1**; larger = more similar; **1.0** = identical under ANARCI’s identity calculation |
| Method | ANARCI `run_germline_assignment` V-gene identity |
| Optional relation | approximate divergence descriptor ≈ `1 − identity` (not distributed) |
| Missing | none in v1.0 |
| Caution | Not a quality score; not affinity; not developability |

### `light_germline_identity`

| | |
|---|---|
| Type | float |
| Definition | Sequence **identity** (similarity) of the observed VL V region to the ANARCI-assigned germline **V allele/reference** |
| Scale | **0–1**; larger = more similar; **1.0** = identical under ANARCI’s identity calculation |
| Method | Same as heavy |
| Missing | none in v1.0 |
