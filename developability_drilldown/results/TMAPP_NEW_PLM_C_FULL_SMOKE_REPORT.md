# TmApp NEW PLM C+FULL SMOKE REPORT

**Platform:** `DL_FOLDLOCAL_COSINE_V3`  
**Control:** EXP-T121 (AbLang2 / C / FULL)  
**Seed:** 101  
**Topology:** C (REG-only cross, mean merge, share_hl_encoder=True, FULL)  

Integration / validation only. **Do not rank or drop PLMs by these smoke scores.**

| Code | PLM | representation context | raw dim |  P |  S | mean | status |
| ---- | --- | ---------------------- | ------: | -: | -: | ---: | ------ |
| EXP-T157 | ablang1 | SEPARATE_CHAIN | 768 | 3.4671 | 3.2599 | 3.3635 | PASS |
| EXP-T158 | esm1b | SEPARATE_CHAIN | 1280 | 3.3180 | 3.2040 | 3.2610 | PASS |
| EXP-T159 | esmc600m | SEPARATE_CHAIN | 1152 | 3.3240 | 3.3664 | 3.3452 | PASS |
| EXP-T160 | currab | PAIRED_NATIVE | 1280 | 3.3158 | 3.1833 | 3.2496 | PASS |

## Per-experiment protocol details

### EXP-T157 (ablang1)

- raw PLM hidden dim: `768`
- Linear(768→128) projection params: `98432`
- total trainable params: `476545`
- selected LR / best epoch per fold:

| scheme | fold | selected_lr | best_epoch | best_val_mae |
| ------ | ---: | ----------: | ---------: | -----------: |
| primary | 0 | 0.001 | 37 | 2.7338 |
| primary | 1 | 0.001 | 14 | 3.3653 |
| primary | 2 | 0.001 | 29 | 2.6867 |
| primary | 3 | 0.0001 | 18 | 2.7027 |
| primary | 4 | 0.001 | 28 | 2.4236 |
| shadow | 0 | 0.001 | 7 | 2.6122 |
| shadow | 1 | 0.0001 | 34 | 2.8772 |
| shadow | 2 | 0.001 | 10 | 3.2406 |
| shadow | 3 | 0.001 | 3 | 3.3276 |
| shadow | 4 | 0.001 | 31 | 2.7392 |

### EXP-T158 (esm1b)

- raw PLM hidden dim: `1280`
- Linear(1280→128) projection params: `163968`
- total trainable params: `542081`
- selected LR / best epoch per fold:

| scheme | fold | selected_lr | best_epoch | best_val_mae |
| ------ | ---: | ----------: | ---------: | -----------: |
| primary | 0 | 0.001 | 22 | 2.6779 |
| primary | 1 | 0.001 | 10 | 3.3001 |
| primary | 2 | 0.0001 | 88 | 3.0793 |
| primary | 3 | 0.001 | 28 | 2.6839 |
| primary | 4 | 0.001 | 61 | 2.8743 |
| shadow | 0 | 0.001 | 32 | 2.4275 |
| shadow | 1 | 0.001 | 23 | 3.1379 |
| shadow | 2 | 0.001 | 24 | 3.0713 |
| shadow | 3 | 0.001 | 10 | 3.4700 |
| shadow | 4 | 0.001 | 18 | 3.0566 |

### EXP-T159 (esmc600m)

- raw PLM hidden dim: `1152`
- Linear(1152→128) projection params: `147584`
- total trainable params: `525697`
- selected LR / best epoch per fold:

| scheme | fold | selected_lr | best_epoch | best_val_mae |
| ------ | ---: | ----------: | ---------: | -----------: |
| primary | 0 | 0.001 | 73 | 2.8118 |
| primary | 1 | 0.001 | 11 | 3.2973 |
| primary | 2 | 0.001 | 34 | 3.4025 |
| primary | 3 | 0.0001 | 103 | 2.6836 |
| primary | 4 | 0.001 | 34 | 2.9903 |
| shadow | 0 | 0.0001 | 86 | 2.4939 |
| shadow | 1 | 0.0001 | 94 | 3.1336 |
| shadow | 2 | 0.001 | 25 | 3.0966 |
| shadow | 3 | 0.001 | 28 | 3.5042 |
| shadow | 4 | 0.001 | 50 | 3.0043 |

### EXP-T160 (currab)

- raw PLM hidden dim: `1280`
- Linear(1280→128) projection params: `163968`
- total trainable params: `542081`
- selected LR / best epoch per fold:

| scheme | fold | selected_lr | best_epoch | best_val_mae |
| ------ | ---: | ----------: | ---------: | -----------: |
| primary | 0 | 0.0001 | 22 | 2.7899 |
| primary | 1 | 0.0001 | 20 | 3.5078 |
| primary | 2 | 0.001 | 20 | 2.9400 |
| primary | 3 | 0.0001 | 17 | 2.4199 |
| primary | 4 | 0.001 | 63 | 3.0543 |
| shadow | 0 | 0.001 | 5 | 2.7207 |
| shadow | 1 | 0.0001 | 14 | 3.1085 |
| shadow | 2 | 0.001 | 7 | 3.0880 |
| shadow | 3 | 0.001 | 7 | 3.3276 |
| shadow | 4 | 0.001 | 16 | 2.7206 |

## Embedding diagnostics (QC only; no normalization applied)

- **ablang1**: mean L2=27.6739, std L2=0.3311, zero_frac=0.0, nonfinite=0
- **esm1b**: mean L2=20.6381, std L2=0.5280, zero_frac=0.0, nonfinite=0
- **esmc600m**: mean L2=1.4475, std L2=0.0881, zero_frac=0.0, nonfinite=0
- **currab**: mean L2=35.0110, std L2=0.3722, zero_frac=0.0, nonfinite=0

## Extraction provenance

### ablang1

```json
{
  "model_name": "ablang-heavy / ablang-light",
  "package": "ablang",
  "package_version": "unknown",
  "representation_source": "ablang.pretrained(...).rescoding (res-codings)",
  "raw_hidden_dimension": 768,
  "representation_context": "SEPARATE_CHAIN",
  "paired_vs_separate": "separate",
  "special_token_mapping_rule": "rescoding returns AA-only vectors; no BOS/EOS retained",
  "mapping": "EXACT_AA_1TO1",
  "license_status": "REVIEW_MODEL_OUTPUT",
  "device": "cuda:0",
  "extraction_wall_time_sec": 3.405036449432373
}
```

### esm1b

```json
{
  "model_name": "facebook/esm1b_t33_650M_UR50S",
  "package": "transformers (EsmModel; fair-esm shadowed by biohub esm)",
  "package_version": "4.57.6",
  "representation_source": "EsmModel.last_hidden_state (final layer 33)",
  "raw_hidden_dimension": 1280,
  "representation_context": "SEPARATE_CHAIN",
  "paired_vs_separate": "separate",
  "special_token_mapping_rule": "strip <cls> and <eos>; keep exactly len(seq) AA vectors",
  "mapping": "EXACT_AA_1TO1_BOS_EOS_STRIPPED",
  "license_status": "REVIEW_MODEL_OUTPUT",
  "device": "cuda:0",
  "extraction_wall_time_sec": 12.396049499511719,
  "final_layer": 33
}
```

### esmc600m

```json
{
  "model_name": "biohub/ESMC-600M",
  "package": "esm (Biohub)",
  "package_version": "3.4.1",
  "representation_source": "ESMC.logits(..., return_embeddings=True) final embeddings",
  "raw_hidden_dimension": 1152,
  "representation_context": "SEPARATE_CHAIN",
  "paired_vs_separate": "separate",
  "special_token_mapping_rule": "encode adds BOS/EOS; strip indices [0] and [-1]",
  "mapping": "EXACT_AA_1TO1_BOS_EOS_STRIPPED",
  "license_status": "REVIEW_MODEL_OUTPUT",
  "device": "cuda:0",
  "extraction_wall_time_sec": 9.694927453994751,
  "model_revision": null,
  "inferred_from_checkpoint": true,
  "note_te_fallback": "Transformer Engine / flash-attn may be unavailable; numerical fallback noted by esm package"
}
```

### currab

```json
{
  "model_name": "brineylab/CurrAb",
  "package": "transformers",
  "package_version": "4.57.6",
  "representation_source": "EsmForMaskedLM last_hidden_state",
  "raw_hidden_dimension": 1280,
  "representation_context": "PAIRED_NATIVE",
  "paired_vs_separate": "paired",
  "special_token_mapping_rule": "exclude single <cls> separator; no BOS/EOS when add_special_tokens=False",
  "mapping": "EXACT_AA_1TO1_CLS_SEPARATOR_STRIPPED",
  "license_status": "REVIEW_MODEL_OUTPUT",
  "device": "cuda:0",
  "extraction_wall_time_sec": 10.799379110336304,
  "model_revision": "92e28534663e163f1b398f773b3fe041085737d9"
}
```

## Implementation notes

- ESM-1b extracted via HuggingFace `EsmModel` (`facebook/esm1b_t33_650M_UR50S`) because biohub `esm` 3.x shadows fair-esm on `import esm`.
- AbLang1 light weights were missing locally (empty `tmp.tar.gz` from a prior failed download); restored from the official OPIG tarball before extraction.
- Re-extract QC compares after float16 disk round-trip (bundles stored as float16).
- ESM-C 600M uses Biohub `ESMC_600M_202412` + `LogitsConfig(return_embeddings=True)`; inferred hidden dim **1152**.
- CurrAb uses native paired `Heavy<cls>Light` with `add_special_tokens=False`; `representation_context: PAIRED_NATIVE`.
- Manifest audit: `esm2` entry corrected from historical `chains: [H]` to `[H, L]` after light pack materialization.

## STOP

Do **not** run A/B1/B2/D or annotation ablations for these PLMs until smoke audit.
