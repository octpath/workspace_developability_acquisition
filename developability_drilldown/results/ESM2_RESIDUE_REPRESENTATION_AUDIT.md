# ESM-2 residue representation audit (drilldown) — stub

## Model identity

| Field | Value |
|---|---|
| HuggingFace id | `facebook/esm2_t33_650M_UR50D` |
| Layer used | **33** (final; residue hidden states) |
| Hidden dim | **1280** |
| Chain encoding | **H and L separate** (independent forwards) |

## Bundle / cache status

| Asset | Status |
|---|---|
| Heavy residue pack | Present under `top_models_feature_bundle/residue_level/esm2/` (`heavy_embeddings.npy`, `heavy_mask.npy`, `ids.npy`) |
| Light residue pack | Materialized via `developability_drilldown/scripts/build_esm2_light_residue_bundle.py` → `light_embeddings.npy`, `light_mask.npy` (max_len_L=120) |
| Organizer cache | `organizer_extension/feature_prospecting/structure_marathon/cache/esm2_residue/` (324 npz files with H+L) |

## Notes

- Checkout historically shipped **Heavy-only** ESM-2 residue tensors; Light is built from the same organizer cache (`npz` key `L`), id-aligned to `ids.npy`.
- HL experiments with `plm_source=esm2` load `ResidueBundle.esm2_l` when light files exist; ARCH-H0 / `chain_mode=H_ONLY` uses Heavy only.
- Full semantic QC (Light pool parquet parity, part-split packaging, license) can be expanded when the Light pack is frozen for release.

## Cross-chain precontextualization

| Flag | Value |
|---|---|
| `PRECONTEXTUALIZED_ACROSS_CHAINS` | **NO** (separate H/L encoding) |
