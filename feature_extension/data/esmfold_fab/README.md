# esmfold_fab/

## What is this?
**Predicted / reconstructed Fab** structures built by attaching organizer **surrogate constant domains**
to competition variable sequences, then folding with ESMFold (organizer Fab reconstruction pipeline).

## Molecular scope
**Fab** (VH–CH1 + VL–CL). **Not** a full IgG. **Not** an experimentally solved Fab structure.

## Important scientific caveats
- Experimental **TmApp** assays were performed on **Fab** molecules.
- These files are **reconstructed** Fab models for feature extraction, not experimental coordinates.
- Variable sequences come from competition antibodies.
- Constant domains use **common surrogate** CH1 / Cκ / Cλ sequences (organizer POLICY B / UniProt-derived),
  not the exact historical allele/junction/papain terminus of each experimental Fab.
- Exact CH1–hinge / papain cleavage details are **not fully known** for the competition reagents.

See organizer provenance (copied conceptually into RELEASE_NOTES):
`fab_reconstruction/sequences/CONSTANT_DOMAIN_POLICY.md`.

## Number of antibodies
324 PDB files named `{id}.pdb`.

## Target labels used during generation
**NO**

## Known issues
Any antibody-specific prep failures for later physics (e.g. OpenMM/FeNNix) are **out of scope** for this structure release.
Raw ESMFold Fab availability here is separate from later simulation readiness.
