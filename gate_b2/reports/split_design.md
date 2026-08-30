# Split design (pre-model)

Splits selected **only** from sequence/target/germline/subset diagnostics — no model scores.

Clustering: paired VH/VL connected components at 90% identity (either chain).

Candidate pool: 120 seeds × geometries approximating 60/20/20, 60/15/25, 55/20/25.

Quality score (lower better): size balance + mean/SD/quantile target balance + germline/subset TV distance + NN similarity penalty + min group counts.

## HIC

- Candidates scored: 120
- **canonical**: score=1.171 counts={'Dev': 208, 'Public': 70, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20262616
- **shadow_1**: score=1.221 counts={'Dev': 207, 'Public': 71, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20262605
- **shadow_2**: score=1.256 counts={'Dev': 201, 'Public': 77, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20262593
- **shadow_3**: score=1.265 counts={'Dev': 204, 'Public': 74, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20262586
- **shadow_4**: score=1.302 counts={'Dev': 208, 'Public': 70, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20262591
- **shadow_5**: score=1.340 counts={'Dev': 208, 'Public': 53, 'Private': 87} geometry≈0.60/0.15/0.25 seed=20263600

## TmApp

- Candidates scored: 120
- **canonical**: score=1.313 counts={'Dev': 197, 'Public': 79, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20269280
- **shadow_1**: score=1.342 counts={'Dev': 189, 'Public': 70, 'Private': 87} geometry≈0.55/0.20/0.25 seed=20271268
- **shadow_2**: score=1.351 counts={'Dev': 205, 'Public': 70, 'Private': 71} geometry≈0.60/0.20/0.20 seed=20269288
- **shadow_3**: score=1.417 counts={'Dev': 202, 'Public': 74, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20269279
- **shadow_4**: score=1.426 counts={'Dev': 206, 'Public': 70, 'Private': 70} geometry≈0.60/0.20/0.20 seed=20269286
- **shadow_5**: score=1.447 counts={'Dev': 187, 'Public': 88, 'Private': 71} geometry≈0.60/0.20/0.20 seed=20269270

## Rationale

Canonical = best pre-model score. Shadows = next-best distinct seeds for robustness.
Larger Private fractions included among candidates because N≈350 makes 20% Public noisy for ranking.
Final geometry of each frozen split is whatever the selected seed produced under group integrity.

