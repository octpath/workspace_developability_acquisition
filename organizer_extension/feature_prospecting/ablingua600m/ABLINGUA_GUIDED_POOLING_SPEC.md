# AbLingua Structure-Guided Pooling — SPEC (FROZEN)

Frozen before TmApp label evaluation.

## Sanity
`SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT` — HL vs HL+SEQ matrices differ; PCA32 MAE identical ~1e-15.

## Mapping
Official TripleAA: `>seq<` sliding 3-mers; residue = mean of covering non-special token hiddens.

## CDR
IMGT regions from `cdr_sequence_index_imgt.csv` (324/324). Masks: CDR_ALL, FRAMEWORK, CDR3.

## RASA
ESMFold + ShrakeRupley + Tien2013; **exposed ≥ 0.2** (feature_extension default).

## Families (exact)
CDR_ALL, CDR3, EXPOSED, BURIED, RASA_WEIGHTED, BURIED_WEIGHTED, EXPOSED_CDR, CDR_FR_SPLIT

## Parent
CURRENT_RECIPE (AbLang2+SEQ_BASIC+BIOEMU_NEW_PAIRWISE+M1) + AbLingua GLOBAL MASKED_MEAN

## Combos
COMBO_1/2/3 only as listed in FREEZE.json

## Dimensionality
Fold-local PCA32 per AbLingua block (same as prior sprint).

## No
New thresholds, Optuna, Public/Private, layer search, combinatorial fishing.
