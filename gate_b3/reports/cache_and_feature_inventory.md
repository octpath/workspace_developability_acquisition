# Cache and feature inventory (Gate B3)

Final competition population target: antibodies with **both** HIC and TmApp (N≈324 from `triple_core.csv`).

All listed representations below cover the 324-overlap set and are **target-independent** (safe to reuse without recompute if sequence/model hashes unchanged).

| feature | source | N | dim | overlap coverage | safe | path |
|---|---|---:|---|---:|---|---|
| stage_A_simple | B1 | 400 | 99 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/features/stage_A_simple.csv` |
| stage_B_cdr | B1 | 400 | 260 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/features/stage_B_cdr.csv` |
| stage_C_shortcut | B1 | 400 | 11 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/features/stage_C_shortcut.csv` |
| AbLang2 | B1 | 400 | 480 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/plm/manifest_ablang2_default.csv` |
| ESM-1b | B1 | 400 | 2560 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/plm/manifest_esm1b_t33_650M_UR50S.csv` |
| ESM-2 | B1 | 400 | 2560 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/plm/manifest_esm2_t33_650M_UR50D.csv` |
| ESM-2_CDR6 | B1 | 400 | 7680 | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/plm/manifest_esm2_t33_650M_UR50D_CDR6.csv` |
| ABodyBuilder2_PDB | B1 | 400 | PDB | 324 | True | `/workspace_developability_acquisition/gate_b1/cache/structures/abb2_manifest.csv` |
| ESMFold_native_VHVL | B2 | 370 | PDB | 324 | True | `/workspace_developability_acquisition/gate_b2/cache/structures/esmfold_native_manifest.csv` |
| ABB_SASA_RASA_PATCH | B2 | 400 | 247 | 324 | True | `/workspace_developability_acquisition/gate_b2/cache/structure_features/abb_sasa_rasa_patch.csv` |
| ESMFN_SASA_RASA_PATCH | B2 | 370 | 189 | 324 | True | `/workspace_developability_acquisition/gate_b2/cache/structure_features/esmfold_native_sasa_rasa_patch.csv` |
| IMGT_numbering_germline | B1 | 400 | regions+germline | 324 | True | `/workspace_developability_acquisition/gate_b1/data/numbering_germline.csv` |

## Reuse policy

- Do **not** recompute PLM embeddings, ABB/ESMFold structures, or B2 SASA/RASA/patch tables.
- New B3 branches: IMGT positional one-hot, germline-relative mutations, n-grams, extended structure geometry, Train-only CV features.
- Organizer-only columns (`b_cell_subset`, donor) stay out of participant files.

