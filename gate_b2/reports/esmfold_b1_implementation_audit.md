# ESMFold B1 implementation audit

Source inspected: `/workspace_developability_acquisition/gate_b1/scripts/08_structures_esmfold.py`

Also compared to read-only reference: `/workspace/external_envs/esmfold/batch_fold.py`

## Package used in B1

| Item | Value |
|------|-------|
| Package | **Hugging Face `transformers`** |
| Class | `transformers.AutoTokenizer` + `transformers.EsmForProteinFolding` |
| Checkpoint | `facebook/esmfold_v1` |
| **Not used** | `facebookresearch/esm` `esm.pretrained.esmfold_v1()` |

Confirmed by imports in `08_structures_esmfold.py` lines 45–48:

```python
from transformers import AutoTokenizer, EsmForProteinFolding
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = EsmForProteinFolding.from_pretrained(MODEL_ID, low_cpu_mem_usage=True).to(device)
```

## Exact input representation

B1 concatenated:

```text
seq = heavy + ("G" * 25) + light
```

- Colon `:` separator: **NOT used**
- Gly25 linker: **manually inserted** as external input (`LINKER = "G" * 25`)
- Tokenizer: `add_special_tokens=False`

## Chain handling

| Mechanism | B1 behavior |
|-----------|-------------|
| Native multimer API | No |
| Residue-index chain offset (model-native) | No (HF path) |
| Manual linker masking in features | Yes — linker residues dropped when rewriting PDB |
| Output chain IDs preserved by model | No — HF `output_to_pdb` emits single chain; B1 **rewrote** chain IDs by residue index (`H` for VH, skip linker, `L` for VL) |

Relevant rewrite logic (lines 109–143): residue index along polymer; residues in `[hlen, hlen+25)` discarded; remaining labeled `H` / `L`.

## Comparison to workspace C3a ESMFold helper

`/workspace/external_envs/esmfold/batch_fold.py` also uses HF `EsmForProteinFolding`, but concatenates **`heavy + light` with no linker** and only stores pLDDT (no PDB chain rewrite). B1 improved on that by inserting Gly25 and excluding linker from the PDB used for SASA.

## Implications for Gate B2

B1 did **not** exercise the official `facebookresearch/esm` multimer path (`VH:VL` → `infer_pdb`). Gate B2 must run that native branch separately and compare.

## Precise language

B1 used HF ESMFold as a **single-polymer fold with a manual Gly linker and post-hoc chain bookkeeping**, not the official colon-separated multimer inference path.
