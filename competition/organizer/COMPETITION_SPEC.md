# Competition specification (organizer)

Competition title: **Antibody Developability — TmApp & HIC**  
Package version: **1.0**  
Status: **production release** — production split frozen.

---

## 1. Source dataset

- Shehata et al., *Cell Reports* 28:3300–3308.e4 (2019)  
- DOI: `10.1016/j.celrep.2019.08.056`  
- Supplement table mmc2 (antibody sequences + assay columns)  
- Frozen competition population: Gate B3 `final_population.csv` (N=324 with both HIC and TmApp)

---

## 2. Tasks

Two independent regression tracks, same antibodies / same Public–Private mask:

| Track | Target column | Unit | Primary metric |
|---|---|---|---|
| 1 | `TmApp` | °C | MAE |
| 2 | `HIC` | min | MAE |

- **No** combined / weighted overall score  
- **No** grand aggregate score  
- Two independent leaderboards  

---

## 3. Distributed schemas

### `data/distribution/dev.csv`

`id,heavy,light,TmApp,HIC` — N=162 Train/Dev.

### `data/distribution/test_features.csv`

`id,heavy,light` — N=162 Test (no labels, no split flags).

### `data/distribution/sample_submission.csv`

`id,TmApp,HIC` — same IDs as `test_features.csv`.

### `data/distribution/dev_annotations.csv` / `test_annotations.csv`

Optional **sequence-derived antibody annotations** (not submission requirements; **not** used by the scorer):

`id,heavy_v_family,heavy_j_gene,light_v_family,light_j_gene,light_chain_type,h_cdr1_length,h_cdr2_length,h_cdr3_length,l_cdr1_length,l_cdr2_length,l_cdr3_length,heavy_germline_identity,light_germline_identity`

- Sequence / reference-annotation derived only  
- Same procedure for Dev and Test  
- **No** target information, Public/Private flags, donor, or B-cell-origin metadata  

---

## 4. Solution schema

### `data/secret/solution.csv`

`id,TmApp,HIC,is_public,is_private` — N=162.

Constraints:

- `is_public XOR is_private` for every row  
- `sum(is_public)=81`, `sum(is_private)=81`  
- identical mask for both targets  

---

## 5. Production split

| Field | Value |
|---|---|
| `split_id` | `GEN_0001_B_20271100` |
| `seed` (metadata) | `20271100` |
| Construction method | `B_simulated_annealing` (Gate B7.3; model-blind statistical balancing prior to model-safety checks) |
| Public N | 81 |
| Private N | 81 |
| Public ID hash | `2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376` |
| Private ID hash | `f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0` |

Authoritative ID lists: `organizer/SPLIT_MANIFEST.json`.  
**Do not** regenerate the mask from seed alone.

HIC interpretive band counts under this split (diagnostic):

- MEDIUM Public/Private = 3/3  
- HIGH Public/Private = 4/3  

---

## 6. Scoring

Join submission to solution by **`id`** (row order ignored).

For each target \(y \in \{\mathrm{TmApp}, \mathrm{HIC}\}\) and each subset \(S \in \{\mathrm{Public}, \mathrm{Private}\}\):

\[
\mathrm{MAE}_S(y) = \frac{1}{|S|}\sum_{i \in S} \lvert y_i - \hat{y}_i\rvert
\]

Primary reported quantities:

- Public TmApp MAE  
- Private TmApp MAE  
- Public HIC MAE  
- Private HIC MAE  

Lower is better.

Optional diagnostics (not primary; never used for ranking or tie-breaking): Pearson, Spearman, RMSE.

---

## 7. Submission validity

A submission is **INVALID** if any of the following hold:

- missing required columns `id`, `TmApp`, `HIC`  
- duplicate IDs  
- missing IDs (ID set ⊂ Test)  
- extra IDs (ID set ⊃ Test)  
- ID set ≠ Test ID set for any other reason  
- non-numeric / NaN / Inf predictions in either target column  

Both prediction columns must exist, be numeric, and be finite—even if a participant focuses on one track.

Duplicate handling: rejected as invalid (do not silently drop).  
Missing predictions: rejected.  
ID mismatch: rejected.  
Row order: ignored (join by ID).

---

## 8. Tie handling (scientific rule)

Within each track independently:

1. Primary ranking key = **Private MAE** (lower is better).  
2. If two submissions have **exactly equal Private MAE**, they receive the **same scientific rank** (tied).  

Exact Private-MAE ties are **not** broken by:

- Pearson  
- Spearman  
- RMSE  
- Public MAE  
- submission timestamp  
- submission ID  

If a hosting platform imposes different mechanical tie rendering for display, document that platform behavior separately. **This specification remains the scientific competition rule.**

---

## 9. Reproducibility

- `build_competition_data.py` rebuilds CSVs from frozen population + `SPLIT_MANIFEST.json`  
- Rebuild must verify Public/Private ID hashes  
- CSV outputs must be byte-stable across rebuilds (no timestamps inside CSVs)  

---

## 10. Secrecy

Participant-visible tree must not contain:

- Test labels  
- `is_public` / `is_private`  
- Public/Private ID lists  
- Organizer benchmark scores / model names  
