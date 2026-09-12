# F1_SURFACE residue reconstruction — PASS criteria (frozen before results)

Defined before inspecting full reconstruction outcomes.

## PASS-EXACT
For every antibody with successful structure mapping, and for every of the 35
F1 columns (ARO19 + HYDRO16):

- max |recon − historical| ≤ 1e-10
  OR exact integer equality for count features

AND residue↔sequence mapping succeeds for all H and L residues used by the
historical extractors (sequence exact-match chain map).

## PASS-NUMERIC
Not PASS-EXACT, but for all 35 columns across all antibodies:

- max abs error ≤ 1e-6 relative to feature scale, OR
- for continuous SASA/H features: max abs ≤ 1e-4 and Pearson r ≥ 0.999999
- no antibody with any column abs error > 1e-3
- no systematic signed bias (|mean error| < 1e-4 per column)

AND mapping coverage ≥ 99% of antibodies with both chains OK.

## PARTIAL
Exactly one of {ARO19, HYDRO16} meets PASS-EXACT or PASS-NUMERIC;
the other does not. Do not claim a full F1 residue source.

## FAIL
Neither block meets PASS-NUMERIC, or mapping coverage < 95%, or
reconstruction requires forbidden methods (aggregate→residue inversion,
RASA/SAP/SCM substitution, arbitrary nearest-residue without documenting
originating-atom preservation).

## Scope notes
- `phi_finite_frac` is a QC column; if historical values are identically 1.0
  and reconstructed vertices match historical coordinates exactly, attaching
  historical per-vertex phi (or reporting finite fraction 1.0) is allowed as
  provenance-consistent QC, not feature invention.
- Target labels must not influence mapping or tolerances.
