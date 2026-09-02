# OPENMM-STRAIN_v1

Protocol: pdb2pqr AMBER pH7 → OpenMM 8.6 `amber14-all` NoCutoff → `LocalEnergyMinimizer` tol=10 kJ/mol/nm, maxIterations=500（no MD）  
Paper: https://doi.org/10.1371/journal.pcbi.1005659 · Repo: https://github.com/openmm/openmm

- Robustness: **FRAGILE**
- Artifact: **NOT** GENERATOR_GEOMETRY_ARTIFACT_DOMINATED（clash Spearman 低）
- Caveat: vacuum 初期力/ΔE が重尾で CV Ridge が数値不安定
- TmApp: **MIXED**（実用 signal なし）· HIC: NO_EVIDENCE

Master: [FEATURE_PROSPECTING_POSTMORTEM_V1_JA.md](../FEATURE_PROSPECTING_POSTMORTEM_V1_JA.md)
