# Generalized spatial property aggregation (design note only)

Do **not** implement in H094–H101.

General form:

```
P_i(R) = Σ_j I[d_ij <= R] * rSASA_j * property_j
```

Future property channels:

- A. HYDROPHOBICITY — STATIC_SAP_KD (this batch)
- B. POSITIVE CHARGE — max(q_j, 0)
- C. NEGATIVE CHARGE — max(-q_j, 0)
- D. ABSOLUTE CHARGE — |q_j|
- E. AROMATICITY
- F. H-BOND DONOR / ACCEPTOR propensity

Charge note: do not rely only on signed charge sum; adjacent +/− can cancel
despite a strongly charged patch. Prefer (B)/(C)/(D) separately.
