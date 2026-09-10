# AbLang2 residue representation audit (drilldown)

## Model identity

| Field | Value |
|---|---|
| Package | `ablang2==0.2.1` |
| Checkpoint | `ablang2-paired` (`random_init=False`) |
| Hidden dim | **480** |
| Representation source | `AbLang.AbRep(tokens).last_hidden_states` |
| Residue selection | Exact AA character indices; special tokens `<`, `>`, `\|` stripped |

## Chain encoding

- **H and L are encoded SEPARATELY** with an **empty partner** sequence:
  - Heavy: `pair = [heavy, ""]` → formatted `<heavy>|`
  - Light: `pair = ["", light]` → formatted `\|<light>`
- Extraction lives in `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`
  (`extract_chain_batch`).

## Cross-chain precontextualization

| Flag | Value |
|---|---|
| `PRECONTEXTUALIZED_ACROSS_CHAINS` | **NO** |

Residue vectors for H and L are **not** produced by a single paired forward pass.
Each chain is contextualized only within its own (empty-partner) formatted string.
Any H↔L interaction in downstream models (e.g. ARCH-5/6/6G cross-attention) is
learned separately from these frozen residue embeddings.

## Bundle assets

- `top_models_feature_bundle/residue_level/ablang2/{heavy,light}_{embeddings,mask}.npy`
- See also `top_models_feature_bundle/residue_level/ablang2/ABLANG2_RESIDUE_AUDIT.md`
  and `metadata.json` for QC / license notes (`REVIEW_MODEL_OUTPUT`).
