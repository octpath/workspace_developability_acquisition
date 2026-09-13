# HIC H140–H339 technical smoke — frozen evidence

**STATUS: SMOKE_PASS (technical wiring only)**

| Field | Value |
|-------|--------|
| Formal prereg commit | `4e175ab4f98864ba7f7c6496f832e06ea9730519` |
| Evaluation freeze | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |
| Implementation patch (bundle loader) | `cc305d4b4e06209fb15281e3ae6431c2664a1318` |
| Environment | `.venv_b1` |
| CUDA | available (RTX 3090 via `CUDA_VISIBLE_DEVICES=0`) |
| Mode | `--quick` (reduced epochs/patience) |
| Smoke log SHA256 | `9fbee4405cd31aeae229a29a2e0a3cc21dc2bb11e2f8d875e4bad358c49913a0` |
| Summary artifact | `developability_drilldown/results/h140_h339_technical_smoke/SMOKE_SUMMARY.yaml` |

## Exact smoke cells (4)

| Code | Representation | Topology | Annotation |
|------|----------------|----------|------------|
| EXP-H140 | Scratch | SEP | BASE |
| EXP-H234 | AbLang2 PAIRED_NATIVE (`ablang2_unpaired`) | XREG | REGION |
| EXP-H317 | CurrAb SEPARATE_CHAIN (`currab_unpaired`) | FUSE | IMGT |
| EXP-H287 | ESM-C 600M | JOINT | FULL |

## PASS criteria verified

- Primary path: **PASS** (all 4)
- Shadow path: **PASS** (all 4)
- Predictions finite: **PASS**
- Nonzero prediction variance: **PASS**
- `share_hl_encoder=True` confirmed at runtime: **PASS**
- Topology flags logged and match frozen semantics: **PASS**
- Surface / SASA / HSP features: **not activated**
- TmApp target/path contamination: **none**
- Public/Private: **not inspected**

## Implementation fix during smoke

`load_bundle_for_rows` initially passed invalid `need_annotations=` to `load_residue_bundle`. Fixed to match the TmApp factorial loader (PLM need-flags only). Patch SHA above.

## Scientific non-claims (binding)

- Smoke **MAE is not scientific evidence** and is irrelevant to PASS/FAIL.
- Smoke used `--quick` reduced training; results **must never enter** the 200-cell factorial scientific table / COMPLETE registry as production cells.
- Raw smoke directories (`EXP-H140/`, `EXP-H234/`, `EXP-H287/`, `EXP-H317/`, `smoke_console.log`) are operational evidence only and are gitignored from version control; they remain outside scientific factorial prediction paths.

## After this freeze

Full production execution of EXP-H140–H339 (no `--quick`) is separately authorized and must use `.venv_b1`.
