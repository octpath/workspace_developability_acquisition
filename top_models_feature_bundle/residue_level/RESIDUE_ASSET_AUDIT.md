# Residue asset audit

## Large embedding files (GitHub size limit)

The three residue embedding tensors exceed GitHub's comfortable / hard size
limits, so the repo stores binary halves (`.npy.part0` / `.npy.part1`).

Before advanced-model runs, reassemble locally:

```bash
bash top_models_feature_bundle/residue_level/assemble_embeddings.sh
```

Assembled outputs (`*.npy`) are gitignored; parts are tracked.

## Annotations
- Rows: 75057
- Antibodies: 324
- See `ANNOTATION_QC.md` (H/L numbering 324/324, UNKNOWN=0)

## ESM-2 Heavy (`residue_level/esm2/`)
- Model: `facebook/esm2_t33_650M_UR50D`
- Hidden dim: **1280**
- Max len: 140
- dtype disk/memory: float16 / float32
- N IDs: 324
- Mean-pool vs bundled `ESM2_H` max abs diff (float32 source): **5.584e-06**
- License status: **REVIEW_MODEL_OUTPUT**

## AbLingua-600M H+L (`residue_level/ablingua600m/`)
- Model: `IDEA-AI4S/AbLingua` @ `4d1272df61a32805695f8789664e2716bb78bb60`
- Hidden dim: **1280**
- Mapping: TripleAA token_to_residue_spans + residue_embeddings_from_tokens (validated guided-pooling)
- Max len H/L: 140 / 120
- Extraction wall time: 14.5s
- License status: **REVIEW_MODEL_OUTPUT**
- Note: residue overlap pooling need not equal historical TripleAA token MASKED_MEAN GLOBAL.

## Integrity checks performed
- Unique IDs, no duplicates
- Mask lengths match sequence lengths
- Finite values, no all-zero antibodies
- ESM-2 mean-pool matches pooled feature block within numerical tolerance
