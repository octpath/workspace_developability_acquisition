# HIC H140–H339 technical smoke report

**STATUS: SMOKE_PASS**  
**Date context:** post formal prereg commit `4e175ab4f98864ba7f7c6496f832e06ea9730519`  
**MAE is not a PASS criterion.** Public/Private not inspected.

## Configurations tested (`--quick`)

| Code | Representation | Topology | Annotation | Coverage |
|------|----------------|----------|------------|----------|
| EXP-H140 | Scratch | SEP | BASE | Scratch + SEP + BASE + Primary/Shadow |
| EXP-H234 | AbLang2 PAIRED_NATIVE (`ablang2_unpaired`) | XREG | REGION | PLM + PAIRED_NATIVE + XREG + non-BASE |
| EXP-H317 | CurrAb SEPARATE_CHAIN (`currab_unpaired`) | FUSE | IMGT | PLM + SEPARATE_CHAIN + FUSE + non-BASE |
| EXP-H287 | ESM-C 600M | JOINT | FULL | PLM + JOINT + FULL |

## PASS checks observed

- Config load; residue/annotation assets load
- Topology flags match frozen semantics (`share_hl_encoder=True` explicit at runtime)
- Forward/backward; Primary and Shadow paths succeed
- Predictions finite; variance > 0
- Smoke artifacts under `developability_drilldown/results/h140_h339_technical_smoke/`
- No TmApp target/path contamination; no surface features
- Resume/status path exercised via runner (`status` / smoke isolation from scientific COMPLETE)

## Implementation patch

- Fixed `load_bundle_for_rows` (`need_annotations` was invalid for `load_residue_bundle`).
- Smoke executed with `.venv_b1` (CUDA). Default `.venv` is CPU-only torch.

## Not done

- Full EXP-H140–H339 batch **not** launched (awaits next human approval).
- Quick smoke results are **not** scientific factorial COMPLETE cells.
