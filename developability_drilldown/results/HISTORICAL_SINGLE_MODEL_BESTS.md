# Historical single-model bests (comparable registry)

Comparable = `canonical_simple_tvt` family rows currently in `experiments.csv` (77).
Historical Stage/SVR protocol models are **not** included in these bests (see NONCOMPARABLE in audit).

## TmApp — comparable single-model bests

- **CV Primary**: `2.702681` — `EXP-T003` / `LIN_TM_BIOEMU_MPNN_RIDGE` / LINEAR / `FS_TM_BIOEMU_MPNN`
- **CV Shadow**: `2.772998` — `EXP-T037` / `TRF_TM_ABLINGUA_FULL_CONCAT_FUS_BIOEMU_MPNN` / TRANSFORMER / `FS_TM_BIOEMU_MPNN`
- **CV worst**: `2.772998` — `EXP-T037` / `TRF_TM_ABLINGUA_FULL_CONCAT_FUS_BIOEMU_MPNN` / TRANSFORMER / `FS_TM_BIOEMU_MPNN`
- **Public**: `3.037159` — `EXP-T038` / `TRF_TM_ABLANG2_MIN_CONCAT` / TRANSFORMER / `nan`
- **Private**: `3.090113` — `EXP-T024` / `XGB_TM_ABLINGUA_GLOBAL` / XGBOOST / `FS_TM_ABLINGUA_GLOBAL`
- **Test overall**: `3.129404` — `EXP-T032` / `TRF_TM_SCRATCH_FULL_MEAN_FUS_ABLINGUA_CDR3` / TRANSFORMER / `FS_TM_ABLINGUA_CDR3`

## HIC — comparable single-model bests

- **CV Primary**: `0.434526` — `EXP-H033` / `TRF_HIC_ESM2_FULL_HONLY_FUS_ESM2_SEQ_ARO` / TRANSFORMER / `FS_HIC_ESM2_SEQ_AROMATIC`
- **CV Shadow**: `0.441328` — `EXP-H030` / `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` / TRANSFORMER / `FS_HIC_ESM2_SEQ_AROMATIC`
- **CV worst**: `0.442203` — `EXP-H030` / `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` / TRANSFORMER / `FS_HIC_ESM2_SEQ_AROMATIC`
- **Public**: `0.397606` — `EXP-H030` / `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` / TRANSFORMER / `FS_HIC_ESM2_SEQ_AROMATIC`
- **Private**: `0.452537` — `EXP-H022` / `XGB_HIC_HYDRO_TITRATION` / XGBOOST / `FS_HIC_HYDRO_TITRATION`
- **Test overall**: `0.438820` — `EXP-H030` / `TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO` / TRANSFORMER / `FS_HIC_ESM2_SEQ_AROMATIC`

## Historical ensembles (NOT single models; not registered)

### Notable HIC prediction ensembles

- `HIST__HIC_PRIVATE_WINNER`: Private=0.4180226409189143 Public=0.4237480344059232 CV_P=0.4324716368936545 — **ENSEMBLE**
- `HIST__HIC_PUBLIC_WINNER`: Private=0.4318145768709808 Public=0.4148392100594437 CV_P=0.4257721922584362 — **ENSEMBLE**
- `HIST__HIC_SIMPLE_blend_seq_surf_adv`: Private=0.4222 Public=0.4239 CV_P=0.4252 — **ENSEMBLE**
- `HIC__SIMPLE_blend_esm2_surf__HIST`: Private=0.4180226409189143 Public=0.4237480344059232 CV_P=0.4324716368936545 — **ENSEMBLE**
- `HIC__SIMPLE_blend_seq_surf__HIST`: Private=0.4318145768709808 Public=0.4148392100594437 CV_P=0.4257721922584362 — **ENSEMBLE**
- `LINEAR_ENSEMBLE__H1+H2`: Private=0.4685091035883209 Public=0.4737108787659602 CV_P=0.4812980454095626 — **ENSEMBLE**
- `CROSS_FAMILY_EQUAL_MEAN__HIC__2m`: Private=0.4560640679012345 Public=0.406664422839506 CV_P=0.4272819598005632 — **ENSEMBLE**

### Notable TmApp ensembles

- `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m`: Private=3.260507031387536 Public=3.170291217166945 CV_P=2.673505808625724 — **ENSEMBLE**
- `CROSS_FAMILY_EQUAL_MEAN__TmApp__4m__FOLLOWUP`: Private=3.145526548916667 Public=3.1586887697329566 CV_P=2.653689958872814 — **ENSEMBLE**
- `LINEAR_ENSEMBLE_EQUAL_MEAN__T1+T3`: Private=nan Public=nan CV_P=2.7039045829014308 — **ENSEMBLE**
- `HIST__TmApp_PUBLIC_WINNER`: Private=3.159989439410928 Public=3.0853249349711853 CV_P=2.9375619787158387 — **ENSEMBLE**
- `HIST__TmApp_PRIVATE_WINNER`: Private=3.15007534338225 Public=3.345595945037222 CV_P=2.813262920374202 — **ENSEMBLE**

## Was a stronger historical single HIC model missing from the comparable catalog?

**NO** — for the **comparable** Simple-TVT registry.

### Evidence

- Current comparable HIC best Private: **0.452537** (`EXP-H022` `XGB_HIC_HYDRO_TITRATION`).
- Remembered ~0.433 Private corresponds to prediction **ensembles**, not a single model:
  - `HIC__SIMPLE_blend_seq_surf` / `HIST__HIC_PUBLIC_WINNER`: Private **0.431815** (equal-mean of FUSION esm2+SEQ_ALL SVR + ESMFold SURFACE_CHEM SVR).
  - `HIC__SIMPLE_blend_esm2_surf` / `HIST__HIC_PRIVATE_WINNER`: Private **0.418023** (equal-mean including ESMFold SURFACE_CHEM SVR).
- A genuine ESMFold **single** model exists historically: `HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt` with Private **≈0.438**,
  but it is classified **NONCOMPARABLE** (`HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE` / Stage OOF not proven identical to `canonical_simple_tvt_v1`).
- Therefore it was **not** issued an EXP-H code in this cleanup (per comparability policy).
- No missing **comparable** single-model backfill candidates were found (`MISSING_BACKFILLABLE_*` count = 0).
