# Launch Finalization Audit

Production split:
    GEN_0001_B_20271100

Package version:
    1.0-rc1

Competition rules finalized:
    PASS

Tie policy:
    exact Private MAE tie = shared rank

Scientific TmApp wording:
    VERIFIED

Scientific HIC wording:
    VERIFIED

CC BY 4.0 attribution:
    PASS

Distribution validation:
    PASS

Secret leakage audit:
    PASS

Participant dry-run:
    PASS

Release archive audit:
    PASS

Reproducibility:
    PASS

Overall:
    READY_TO_LAUNCH


---

## Working title

**Antibody Developability — TmApp & HIC**


## Final scoring rules

- Two independent tracks: TmApp, HIC  
- Primary metric: **MAE** (lower better) for each track  
- **No** combined / grand aggregate score  
- Same Public/Private mask for both tracks (81/81 of Test N=162)  
- Join by `id`; both `TmApp` and `HIC` columns required and finite  
- Duplicate / missing / extra IDs → INVALID  
- NaN / Inf → INVALID  
- Exact equal **Private MAE** → **tied scientific rank** (no Pearson/Spearman/RMSE/Public/timestamp/ID breakers)  
- Diagnostics (Pearson/Spearman/RMSE) labeled diagnostic only  


## Split verification

| Check | Result |
|---|---|
| split_id | GEN_0001_B_20271100 |
| Public hash | `2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376` |
| Private hash | `f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0` |
| Public/Private N | 81 / 81 |
| HIC MEDIUM | 3 / 3 |
| HIC HIGH | 4 / 3 |
| Regenerated from seed? | **No** (explicit IDs from SPLIT_MANIFEST) |


## Data counts

| File | N |
|---|---:|
| `dev.csv` | 162 |
| `test_features.csv` | 162 |
| `sample_submission.csv` | 162 |
| `solution.csv` | 162 |


## Attribution status

Crossref VoR metadata (`raw/shehata/a05/crossref.json`):

- Title / authors / journal / year / DOI verified  
- VoR license URL: `http://creativecommons.org/licenses/by/4.0/` (**CC BY 4.0**)  
- Attribution present in participant README and organizer PROVENANCE  
- Explicit non-affiliation / non-endorsement statement included  
- No license conflict detected (TDM entries coexist; packaging uses VoR CC BY 4.0)  


## Scientific wording status

### TmApp — VERIFIED

- Apparent melting temperature  
- Fab fragments  
- DSF; purified Fab heated with fluorescence monitoring  
- Transition from thermal fluorescence curve / derivative  
- “Apparent” caveat retained  
- Simple DSF fluorescence wording (no over-attributed dye mechanism)  

### HIC — VERIFIED

- Exposed hydrophobicity → stronger interaction → self-association tendency → developability risk chain  
- Mandatory: HIC is **not** a direct aggregation assay; high HIC ≠ must aggregate  
- Comparison table lists IgG HIC RT vs Fab DSF TmApp  


## Participant dry-run

Isolated dry-run (`dry_run.py`) and archive-extracted median baseline both **PASS**.

Example median scores (not used for any decision):

| Score | Value |
|---|---:|
| TmApp Public MAE | 3.7840 |
| TmApp Private MAE | 3.7716 |
| HIC Public MAE | 0.5349 |
| HIC Private MAE | 0.5101 |


## Release archive

| Field | Value |
|---|---|
| Filename | `antibody_developability_competition_v1.0-rc1.zip` |
| SHA-256 | `391a73801b4479de5661d6d215c91d7760141371d6527178ec2e51dc1fe8a3f0` |
| Members | `README.md`, `data/dev.csv`, `data/test_features.csv`, `data/sample_submission.csv`, `data/DATA_DICTIONARY.md` |
| Forbidden members | none |
| ZIP timestamps | normalized (1980-01-01); rebuild byte-identical |
| Extracted dry-run | PASS |

Staging tree `release_staging/` mirrors participant contents (byte-identical to competition distribution sources).


## Secret leakage

Post-edit audit of `data/distribution/` + `participant/`: **PASS**  
ZIP member audit: **PASS**  
No organizer benchmarks / Private scores in participant README.


## Reproducibility

- Distribution CSV SHA-256 unchanged from packaging build for data files  
- README / DATA_DICTIONARY / SPEC / PROVENANCE updated for 1.0-rc1 and rehashed in `PACKAGING_MANIFEST.json`  
- ZIP contents deterministic with normalized timestamps  


## Git commit status

Workspace is a Git repository with **no prior commits**.

Launch finalization will attempt a focused first commit of the competition package + release ZIP if cleanly possible.

If a commit SHA cannot be recorded without a circular dirty-state loop on the manifest, `git_commit` may remain documented as post-commit / follow-up.


## Remaining items (non-blocking)

1. Hosting platform may still render ties differently for UI — scientific rule is shared rank.  
2. External publish / upload of the ZIP is **not** performed by this Gate.  
3. Platform-specific attribution footer copy may still be customized at launch.  


## Decision

**READY_TO_LAUNCH**
