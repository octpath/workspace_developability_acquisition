# ESMFold native multimer audit

## Package / API

| Item | Value |
|---|---|
| Package | `facebookresearch/esm` via `fair-esm` **2.0.1** |
| Loader | `esm.pretrained.esmfold_v1()` |
| Input | `sequence = f"{VH}:{VL}"` (official multimer form) |
| Forward | `model.infer(seq)` → `model.output_to_pdb(out)` (single pass; pLDDT from same call) |
| Chunk size | `model.set_chunk_size(128)` |
| GPU | RTX 3090 (physical id 0); ~9–10 GB during inference |
| Recycles | default training max (`cfg.trunk.max_recycles` = **4**) when `num_recycles=None` |
| `residue_index_offset` | default **512** |
| `chain_linker` | default **`"G"*25`** (internal; not supplied as external input string) |
| Chain order | VH then VL (`A` then `B` in PDB output) |

## Semantics verification (n=15)

- Distinct chains: **A / B**
- Length match to VH/VL: **15/15**
- No Gly25 linker residues kept as biological residues in feature extraction (linker masked internally via `linker_mask`)
- COM distance ~21.7–22.7 Å (physically associated; no pathological separation)

## Full union run

- Antibodies: **370** (HIC∪TmApp)
- Success: **370/370**
- Wall time (production pass after validation): **~1686 s** (~4.6 s/structure single-pass)
- Cache: `gate_b2/cache/structures/esmfold_native/`
- Manifest: `gate_b2/cache/structures/esmfold_native_manifest.csv`

## Precise language

ESMFold supports multimer inputs through colon-separated chains and chain-aware inference machinery
(internal poly-G linker + residue-index offset). It is **not** a true AlphaFold-Multimer model.

Contrast with B1: HuggingFace `EsmForProteinFolding` + **external** Gly25 concatenation (see `esmfold_b1_implementation_audit.md`).
