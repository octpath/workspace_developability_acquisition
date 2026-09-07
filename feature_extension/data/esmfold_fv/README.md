# esmfold_fv/

## What is this?
Predicted **Fv** structures from **ESMFold**, one PDB per antibody.

## Molecular scope
**Fv** (variable heavy + variable light only). Not Fab. Not IgG.

## Number of antibodies
324 PDB files named `{id}.pdb`.

## Generation method
Organizer ESMFold native Fv generation used for structure-marathon / gap-closure inputs.
Model version: see RELEASE_NOTES (recorded as available from project provenance; otherwise `NOT_RECORDED`).

## Chain identities
Typically heavy = chain H (or A) and light = chain L (or B) depending on generator convention.
Inspect each PDB header/ATOM chain IDs before region-specific analysis.

## Sequence source
Competition antibody VH/VL sequences (target-blind structure generation).

## Confidence
ESMFold pLDDT is commonly stored in the PDB B-factor field when present. Treat as model confidence, not experimental B-factors.

## QC / limitations
- Predicted structures, not crystal structures.
- Fv-only: constant domains absent.
- No experimental refinement in this package.

## Target labels used during generation
**NO**
