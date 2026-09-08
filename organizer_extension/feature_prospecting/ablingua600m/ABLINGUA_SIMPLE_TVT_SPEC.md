# AbLingua-600M — Simple TVT Spec (FROZEN before label evaluation)

Date: 2026-09-09  
Scope: **TmApp only**. No HIC / Public / Private / Optuna / fine-tuning.

## Model (fixed)

- Repo: `IDEA-AI4S/AbLingua` (~600M, BERT, 30 layers, hidden=1280, max_pos=256)
- Tokenizer: official `baysicx/AbLingua` BioTokenizer 3-gram (`tokens.txt`) — HF hub has no tokenizer files
- Representation: `outputs.hidden_states[-1]`
- Chains: Heavy and Light **separately**, then concat for paired candidates
- Inference: `torch.inference_mode()`, FP32, RTX 3090

## Pooling (only these two interpretations)

| Code | Status | Definition |
|------|--------|------------|
| MASKED_MEAN | **USED** | Mean over `attention_mask==1` tokens; excludes PAD. Official path does not insert CLS/SEP. |
| CLS | **CLS_UNSUPPORTED** | Official `BioTokenizer.tokenize` / `Simple_Collator` never inserts `[CLS]`. Do not invent. |

## Frozen candidates

### DIAGNOSTIC
- `H_mean` (1280)
- `L_mean` (1280)

### PRIMARY
- `HL_mean_concat` (2560) = concat(H_mean, L_mean)

## Evaluation protocol

Exact existing Simple TVT:

- Folds: `virtual_participant/stage0_cv/cv_primary.csv` / `cv_shadow.csv`
- Role: TEST=k, VAL=(k+1)%5, TRAIN=remaining 3
- DEV N=162
- Downstream: StandardScaler → Ridge, alphas `{0.1,1,10,100}`, fold-local alpha on VAL
- Modes: FREE_ALPHA and BASE_FIXED_ALPHA
- High-dim structure block: fold-local PCA if dim > 200, cap 32 (same as `run_simple_tvt_rescreen.py`)
- Standalone PLM: also report **raw** and **PCA32** (Stage2 convention for PLM)

## Predeclared comparisons (no fishing)

1. Standalone: H_mean / L_mean / HL_mean_concat (raw + PCA32)
2. Fair vs AbLang2 `ablang2__HL_paired` under same Simple TVT
3. `HL_mean_concat + SEQ_BASIC`
4. Competition recipe REPLACEMENT: AbLingua HL_mean instead of AbLang2 in `AbLang2+SEQ_BASIC+BIOEMU_NEW_PAIRWISE+M1_PROTEINMPNN`
5. Competition recipe ADDITION: current recipe + HL_mean_concat
6. Direct fusion: `SEQ_BASIC + AbLang2_HL_paired + AbLingua_HL_mean_concat`

SVR: skip new Optuna; replay Stage2 fixed SVR (C=3, gamma=0.01, epsilon=0.1) + fold-local PCA32 **only** if time permits as secondary; primary evaluator is Ridge.

## Do not

- New pooling, layer averaging, Optuna, HIC, Public/Private, redefine embeddings after CV
