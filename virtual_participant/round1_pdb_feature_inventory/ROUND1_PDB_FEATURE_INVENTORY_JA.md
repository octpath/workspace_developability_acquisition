# Round1 PDB由来特徴量 Inventory

**状態:** `ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE`

新規 training / Optuna / feature engineering / DeepResearch なし。repository 実装・artifact の監査のみ。

**重要:** incremental signal による feature 選別は行っていない。computed / used / selected を区別。

## 1. 目的

Round1 Stage3/Stage4 で PDB（ESMFold 予測構造）から **実際に計算した物理量** を feature-level で棚卸しし、DeepResearch の baseline とする。

## 2. 調査対象コード / artifact

- `gate_b2/scripts/03_structure_features.py` — SASA/RASA/patch/interface
- `virtual_participant/stage3_structure/scripts/run_stage3.py` — packing + bundles
- `virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py` — PROPKA/APBS/patch/interactions
- Manifests: `stage3_feature_manifest.csv`, `stage4_feature_manifest.csv`
- Feature tables: `cache/features_ESMFold.csv`, `stage4_all_features.csv`

## 3. 構造入力

| Source | Path | Region mapping | Primary? |
|--------|------|----------------|----------|
| ESMFold native | `esmfold_native/{id}.pdb` | author segment lengths | **Yes** |
| ABodyBuilder2 | `gate_b1/cache/structures/abodybuilder2/` | IMGT resseq | ablation only |

Preprocessing: BioPython PDB parse; no relaxation; no OpenMM.

### Success / failure coverage

| Pipeline | Attempted | Successful | Failed | Notes |
|----------|----------:|-------------:|-------:|-------|
| Stage3 ESMFold features | 162 | 162 | 0 | gate_b2 + packing merged |
| Stage3 ABodyBuilder2 features | 162 | 162 | 0 | ablation; same 218 cols |
| Stage4 PROPKA | 162 | 162 | 0 | extraction_status.csv |
| Stage4 PDB2PQR | 162 | 162 | 0 | extraction_status.csv |
| Stage4 APBS | 162 | 162 | 0 | reparse_apbs_features.py |
| Stage4 ESM-IF1 | 162 | 162 | 0 | invfold_ok per antibody |
| Stage3 iface_frac (ESMFold) | — | 63 | 99 NaN | median impute in modeling |

## 4. PDB-derived feature family 一覧

| feature_family         |   n_features | bundles                                   | primary_software                                                   | TmApp_used   | HIC_used   |
|:-----------------------|-------------:|:------------------------------------------|:-------------------------------------------------------------------|:-------------|:-----------|
| CAVITY_PROXY           |            3 | ADV_CAVITY                                | custom Python (Stage4)                                             | True         | False      |
| ELECTROSTATICS         |           14 | ADV_ELECTROSTATICS                        | APBS                                                               | False        | True       |
| GLOBAL_GEOMETRY        |           14 | STRUCT_GLOBAL                             | ShrakeRupley                                                       | True         | True       |
| INVERSE_FOLDING        |           10 | ADV_INVFOLD                               | ESM-IF1                                                            | True         | True       |
| PACKING_CONTACT        |           42 | ADV_INTERACTIONS;ADV_OTHER;STRUCT_PACKING | ShrakeRupley;custom Python (Stage3 packing);custom Python (Stage4) | True         | True       |
| PKA_PROTONATION        |           15 | ADV_PQR_CHARGE;ADV_PROPKA                 | PROPKA 3.5.1 + PDB2PQR 3.7.1                                       | False        | True       |
| SASA_RASA              |           84 | STRUCT_RASA;STRUCT_SASA                   | ShrakeRupley                                                       | True         | True       |
| SURFACE_CHEMISTRY      |           82 | STRUCT_SURFACE_CHEM                       | ShrakeRupley;custom Python (Stage3 packing)                        | True         | True       |
| SURFACE_PATCH_ADVANCED |           19 | ADV_SURFACE_PATCH                         | custom Python (Stage4)                                             | True         | True       |
| SURFACE_PATCH_SIMPLE   |            5 | STRUCT_PATCH                              | ShrakeRupley;custom Python (Stage3 packing)                        | True         | True       |
| UNSAT_POLAR_PROXY      |            2 | ADV_UNSAT_POLAR                           | custom Python (Stage4)                                             | True         | False      |
| VH_VL_INTERFACE        |           10 | STRUCT_INTERFACE                          | ShrakeRupley;custom Python (Stage3 packing)                        | True         | True       |

**Summary:** - Semantic families: **12**
- Semantic features (ESMFold primary): **300**
- Largest family: **SASA_RASA** (84 features)

### SURFACE_CHEM exact summary

Per region (Fv, VH, VL, H_CDR1/2/3, all_CDR, FR) — 10 columns each:
1. `sasa_hydrophobic/aromatic/positive/negative/polar` — sum SASA where aa ∈ fixed class
2. `rasa_w_hydrophobicity_sum/mean` — Σ(RASA × Kyte-Doolittle KD)
3. `rasa_w_charge_sum` — Σ(RASA × charge); D/E=−1, K/R=+1, H=+0.1
4. `exposed_pos/neg` — Σ RASA for positive/negative aa
Plus global: `cdr_hydrophobic_sasa_exposed`, `h3_hydrophobic_sasa_exposed`.

### Simple PATCH vs ADV_SURFACE_PATCH

| Aspect | STRUCT_PATCH (Stage3) | ADV_SURFACE_PATCH (Stage4) |
|--------|----------------------|----------------------------|
| Graph | Exposed CA, edge if ≤8Å | Exposed CA, edge if ≤8Å |
| Patch | Union-find on hydrophobic/charge/aromatic | CC + local 10Å neighborhood SASA |
| Extra | CDR/H3 hydrophobic exposed SASA | compactness, extent, mixed arom-hydro patches |
| SAP | No | Code: **not canonical SAP formula** |

## 5. Global geometry

STRUCT_GLOBAL: n_res, mean_plddt, Fv_buried_frac

| feature_family   |   n_features |   n_bundles | bundles       | structure_sources   | stages   | raw_quantities                                               | primary_software   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:--------------|:--------------------|:---------|:-------------------------------------------------------------|:-------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| GLOBAL_GEOMETRY  |           14 |           1 | STRUCT_GLOBAL | ESMFold             | Stage3   | ESMFold per-residue pLDDT;structure-derived scalar aggregate | ShrakeRupley       | True       | True                | True         | True       | False                     | supplementary only — not used for inventory filtering |                     nan |                       nan |

## 6. SASA / RASA

ShrakeRupley probe=1.4Å; RASA=SASA/MaxASA(Tien2013); thresholds 0.20/0.25/0.50/1.0

| feature_family   |   n_features |   n_bundles | bundles                 | structure_sources   | stages   | raw_quantities                                                                         | primary_software   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:------------------------|:--------------------|:---------|:---------------------------------------------------------------------------------------|:-------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| SASA_RASA        |           84 |           2 | STRUCT_RASA;STRUCT_SASA | ESMFold             | Stage3   | residue-level RASA (SASA/MaxASA);residue-level SASA;structure-derived scalar aggregate | ShrakeRupley       | True       | True                | True         | True       | True                      | supplementary only — not used for inventory filtering |                0.494268 |                   3.47229 |

## 7. Surface chemistry

AA-class SASA sums + RASA×KD + RASA×charge; 7 regions × ~10 cols

| feature_family    |   n_features |   n_bundles | bundles             | structure_sources   | stages   | raw_quantities                                                                               | primary_software                            | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:------------------|-------------:|------------:|:--------------------|:--------------------|:---------|:---------------------------------------------------------------------------------------------|:--------------------------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| SURFACE_CHEMISTRY |           82 |           1 | STRUCT_SURFACE_CHEM | ESMFold             | Stage3   | residue-level RASA × KD hydrophobicity;residue-level RASA × charge weight;residue-level SASA | ShrakeRupley;custom Python (Stage3 packing) | True       | True                | True         | True       | True                      | supplementary only — not used for inventory filtering |                0.478125 |                   3.37285 |

## 8. Surface patch

Stage3: exposed CA graph 8Å CC; 5 global patch cols + CDR/H3 hydrophobic exposed SASA

| feature_family       |   n_features |   n_bundles | bundles      | structure_sources   | stages   | raw_quantities                        | primary_software                            | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:---------------------|-------------:|------------:|:-------------|:--------------------|:---------|:--------------------------------------|:--------------------------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| SURFACE_PATCH_SIMPLE |            5 |           1 | STRUCT_PATCH | ESMFold             | Stage3   | exposed residue CA graph + local SASA | ShrakeRupley;custom Python (Stage3 packing) | True       | True                | True         | True       | False                     | supplementary only — not used for inventory filtering |                0.478125 |                   3.37285 |

## 9. Packing / contacts

Stage3 CA<8Å contacts, Rg, compactness; Stage4 heavy contacts/clash/packing_degree

| feature_family   |   n_features |   n_bundles | bundles                                   | structure_sources   | stages        | raw_quantities                                                                                                                                       | primary_software                                                   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:------------------------------------------|:--------------------|:--------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| PACKING_CONTACT  |           42 |           3 | ADV_INTERACTIONS;ADV_OTHER;STRUCT_PACKING | ESMFold             | Stage3;Stage4 | CA/heavy-atom distances;N/O heavy atom distance proxy;heavy-atom distance donor/acceptor pairs;residue-level SASA;structure-derived scalar aggregate | ShrakeRupley;custom Python (Stage3 packing);custom Python (Stage4) | True       | True                | True         | True       | True                      | supplementary only — not used for inventory filtering |                0.538954 |                   2.88811 |

## 10. VH–VL interface

BSA, n_interface_res (CA<=5Å), iface composition; no orientation angle

| feature_family   |   n_features |   n_bundles | bundles          | structure_sources   | stages   | raw_quantities                                                                                             | primary_software                            | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:-----------------|:--------------------|:---------|:-----------------------------------------------------------------------------------------------------------|:--------------------------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| VH_VL_INTERFACE  |           10 |           1 | STRUCT_INTERFACE | ESMFold             | Stage3   | CA/heavy-atom distances;chain SASA difference + CA cross-chain distance;structure-derived scalar aggregate | ShrakeRupley;custom Python (Stage3 packing) | True       | True                | True         | True       | False                     | supplementary only — not used for inventory filtering |                0.541927 |                   2.95713 |

## 11. H-bond / salt bridge / polar

salt_bridge 4Å; hbond_proxy 3.5Å N-O; no angle criterion

| feature_family   |   n_features |   n_bundles | bundles                                   | structure_sources   | stages        | raw_quantities                                                                                                                                       | primary_software                                                   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:------------------------------------------|:--------------------|:--------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| PACKING_CONTACT  |           42 |           3 | ADV_INTERACTIONS;ADV_OTHER;STRUCT_PACKING | ESMFold             | Stage3;Stage4 | CA/heavy-atom distances;N/O heavy atom distance proxy;heavy-atom distance donor/acceptor pairs;residue-level SASA;structure-derived scalar aggregate | ShrakeRupley;custom Python (Stage3 packing);custom Python (Stage4) | True       | True                | True         | True       | True                      | supplementary only — not used for inventory filtering |                0.538954 |                   2.88811 |

## 12. Cavity / buried-unsatisfied-polar

cavity_proxy: buried + low CA degree; UNSAT: buried polar + no N/O within 6Å — **both PROXY**

| feature_family   |   n_features |   n_bundles | bundles    | structure_sources   | stages   | raw_quantities                              | primary_software       | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:-----------|:--------------------|:---------|:--------------------------------------------|:-----------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| CAVITY_PROXY     |            3 |           1 | ADV_CAVITY | ESMFold             | Stage4   | buried CA neighbor degree (packing-derived) | custom Python (Stage4) | True       | True                | True         | False      | True                      | supplementary only — not used for inventory filtering |                     nan |                   3.18339 |

## 13. Electrostatics

APBS lpbe mg-auto; 14 potential stats + patch CC on exposed CA

| feature_family   |   n_features |   n_bundles | bundles            | structure_sources   | stages   | raw_quantities                                                                                   | primary_software   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:-------------------|:--------------------|:---------|:-------------------------------------------------------------------------------------------------|:-------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| ELECTROSTATICS   |           14 |           1 | ADV_ELECTROSTATICS | ESMFold             | Stage4   | APBS electrostatic potential sampled at exposed residue CA;exposed residue CA graph + local SASA | APBS               | True       | True                | False        | True       | False                     | supplementary only — not used for inventory filtering |                0.568183 |                       nan |

## 14. pKa / protonation

PROPKA 9 cols + PQR 6 charge cols; no residue-level pKa vector in ML

| feature_family   |   n_features |   n_bundles | bundles                   | structure_sources   | stages   | raw_quantities                                                                                                                        | primary_software             | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:--------------------------|:--------------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| PKA_PROTONATION  |           15 |           2 | ADV_PQR_CHARGE;ADV_PROPKA | ESMFold             | Stage4   | PDB2PQR atom partial charges at pH 6.5;PROPKA residue pKa → Henderson-Hasselbalch charge at pH 6.5;residue-level RASA × charge weight | PROPKA 3.5.1 + PDB2PQR 3.7.1 | True       | True                | False        | True       | False                     | supplementary only — not used for inventory filtering |                0.566634 |                       nan |

## 15. Energy / force / mechanics

OpenMM/MM energy/normal modes: **NOT_FOUND**

## 16. Inverse folding

ESM-IF1 log-likelihood per chain + Fv mean; **scalar score not embedding**

| feature_family   |   n_features |   n_bundles | bundles     | structure_sources   | stages   | raw_quantities                                  | primary_software   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:------------|:--------------------|:---------|:------------------------------------------------|:-------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| INVERSE_FOLDING  |           10 |           1 | ADV_INVFOLD | ESMFold             | Stage4   | ESM-IF1 sequence log-likelihood given structure | ESM-IF1            | True       | True                | True         | True       | True                      | supplementary only — not used for inventory filtering |                0.538969 |                   3.16947 |

## 17. Structure confidence

mean_plddt, mean_confidence from ESMFold B-factor

| feature_family   |   n_features |   n_bundles | bundles       | structure_sources   | stages   | raw_quantities                                               | primary_software   | computed   | used_in_any_model   | TmApp_used   | HIC_used   | in_final_representative   | performance_note                                      |   median_Primary_CV_HIC |   median_Primary_CV_TmApp |
|:-----------------|-------------:|------------:|:--------------|:--------------------|:---------|:-------------------------------------------------------------|:-------------------|:-----------|:--------------------|:-------------|:-----------|:--------------------------|:------------------------------------------------------|------------------------:|--------------------------:|
| GLOBAL_GEOMETRY  |           14 |           1 | STRUCT_GLOBAL | ESMFold             | Stage3   | ESMFold per-residue pLDDT;structure-derived scalar aggregate | ShrakeRupley       | True       | True                | True         | True       | False                     | supplementary only — not used for inventory filtering |                     nan |                       nan |

## 18. ESMFold vs ABodyBuilder2 differences

- Same pipeline code; ABB uses IMGT CDR numbering, ESMFold uses author segment lengths.
- ESMFold: L_CDR1/2/3_empty flags; ABB has full L-CDR region features.
- Stage4 advanced features: **ESMFold only** (162/162).

## 19. Feature lineage

See `round1_pdb_feature_lineage.csv` (15 traced paths).

## 20. Round1内のsemantic overlap

|                   |   total_SASA |   RASA |   SURFACE_CHEM |   simple_PATCH |   ADV_SURFACE_PATCH |   PACKING |   INTERFACE |   ADV_INTERACTIONS |   UNSAT_POLAR |   cavity_proxy |   PROPKA |   APBS |   ESM-IF |
|:------------------|-------------:|-------:|---------------:|---------------:|--------------------:|----------:|------------:|-------------------:|--------------:|---------------:|---------:|-------:|---------:|
| total_SASA        |            3 |      3 |              2 |              2 |                   0 |         0 |           0 |                  0 |             0 |              0 |        0 |      0 |        0 |
| RASA              |            3 |      3 |              2 |              0 |                   0 |         0 |           0 |                  0 |             0 |              0 |        0 |      0 |        0 |
| SURFACE_CHEM      |            2 |      2 |              3 |              2 |                   0 |         0 |           0 |                  0 |             0 |              0 |        0 |      0 |        0 |
| simple_PATCH      |            2 |      0 |              2 |              3 |                   2 |         0 |           0 |                  0 |             0 |              0 |        0 |      0 |        0 |
| ADV_SURFACE_PATCH |            0 |      0 |              0 |              2 |                   3 |         0 |           0 |                  0 |             0 |              0 |        0 |      1 |        0 |
| PACKING           |            0 |      0 |              0 |              0 |                   0 |         3 |           0 |                  2 |             0 |              2 |        0 |      0 |        0 |
| INTERFACE         |            0 |      0 |              0 |              0 |                   0 |         0 |           3 |                  2 |             0 |              0 |        0 |      0 |        0 |
| ADV_INTERACTIONS  |            0 |      0 |              0 |              0 |                   0 |         2 |           2 |                  3 |             1 |              0 |        0 |      0 |        0 |
| UNSAT_POLAR       |            0 |      0 |              0 |              0 |                   0 |         0 |           0 |                  1 |             3 |              1 |        0 |      0 |        0 |
| cavity_proxy      |            0 |      0 |              0 |              0 |                   0 |         2 |           0 |                  0 |             1 |              3 |        0 |      0 |        0 |
| PROPKA            |            0 |      0 |              0 |              0 |                   0 |         0 |           0 |                  0 |             0 |              0 |        3 |      2 |        0 |
| APBS              |            0 |      0 |              0 |              0 |                   1 |         0 |           0 |                  0 |             0 |              0 |        2 |      3 |        0 |
| ESM-IF            |            0 |      0 |              0 |              0 |                   0 |         0 |           0 |                  0 |             0 |              0 |        0 |      0 |        3 |

## 21. Round1で計算していない候補群

| Candidate | Status | Evidence |
|-----------|--------|----------|
| detailed electrostatic surface topology | PARTIAL | apbs_pos/neg_patch_count only; no full surface mesh topology |
| dipole / multipole | NOT_FOUND | no repository implementation |
| explicit pKa-shift descriptors | PARTIAL | propka_mean_pka aggregate only; no per-residue pKa distribution features |
| pH titration curves | NOT_FOUND | fixed pH=6.5 only |
| OpenMM minimization | NOT_FOUND | no OpenMM import or usage in repo |
| force-field energy decomposition | NOT_FOUND |  |
| per-atom / per-residue forces | NOT_FOUND |  |
| relaxation ΔE | NOT_FOUND |  |
| minimization RMSD | NOT_FOUND |  |
| elastic network | NOT_FOUND |  |
| ANM | NOT_FOUND |  |
| GNM | NOT_FOUND |  |
| normal modes | NOT_FOUND |  |
| rigidity analysis | NOT_FOUND |  |
| MaSIF | NOT_FOUND |  |
| dMaSIF | NOT_FOUND |  |
| pretrained geometric GNN | NOT_FOUND |  |
| structure-aware pretrained embedding | PARTIAL | ESM-IF1 scalar log-likelihood only; no 3D GNN embedding stored |
| ML force field | NOT_FOUND |  |
| molecular dynamics | NOT_FOUND |  |
| conformer ensemble | NOT_FOUND | single static ESMFold structure per antibody |
| canonical SAP | NOT_FOUND | adv_spatial_hydrophobicity explicitly not SAP formula |
| continuous molecular hydrophobic potential | NOT_FOUND | discrete KD-weighted SASA only |
| VH/VL orientation angle | NOT_FOUND | vh_vl_center_dist only; no angle |
| explicit cavity detection | NOT_FOUND | cavity_proxy is packing-derived; no fpocket/pyKVFinder |

## 22. DeepResearch handoff

| Candidate concept | Round1 status | Closest Round1 feature | What Round1 actually computed | Remaining difference to investigate |
|-------------------|---------------|--------------------------|-------------------------------|-------------------------------------|
| OpenMM minimization | NOT_FOUND | ADV_INTERACTIONS | static geometric contacts/clashes only | force-field relaxation/strain not computed |
| pKa-shift descriptors | PARTIAL | ADV_PROPKA | propka_mean_pka, net charge at fixed pH; not residue-level pKa vector | abnormal pKa distribution / burial shift not featurized |
| electrostatic patch topology | PARTIAL | ADV_ELECTROSTATICS | APBS potential sampled at CA; CC patch count/size on ±0.5 kT/e | full surface mesh topology / multipole not computed |
| normal modes / elastic network | NOT_FOUND | STRUCT_PACKING / ADV_INTERACTIONS | static contact density and Rg only | mechanical flexibility absent |
| pretrained geometric DL | PARTIAL | ADV_INVFOLD | ESM-IF1 per-chain log-likelihood scalars | no MaSIF/dMaSIF/GNN embedding |
| canonical SAP | NOT_FOUND | ADV_SURFACE_PATCH | local hydrophobic SASA + CA patches; code comment: not SAP formula | continuous hydrophobic potential / SAP weighting absent |
| explicit cavity detection | NOT_FOUND | ADV_CAVITY | buried low CA degree count × sphere volume proxy | fpocket/pyKVFinder-style void detection absent |
| VH/VL orientation angle | NOT_FOUND | STRUCT_INTERFACE | VH_VL_center_dist and interface contacts only | no dihedral/angle between domain axes |


## Q1–Q20 回答要約

- **Q1 SASA/RASA features?** 218 Stage3 cols: SASA(20)+RASA(64)+ratios; see inventory
- **Q2 SURFACE_CHEM exactly?** Per-region sum SASA by AA class + RASA×KD + RASA×charge + exposed_pos/neg
- **Q3 simple vs ADV patch?** Simple: global exposed CA CC 8Å on hydrophobic/charge. ADV: local 10Å SASA + CC + compactness/extent + mixed patches
- **Q4 patch uses surface geometry?** Yes: CA Euclidean graph on exposed residues; not pure sequence aggregation
- **Q5 PACKING?** CA contact count/density, Rg, compactness, clash proxy; Stage4 adds heavy-atom contacts
- **Q6 cavity explicit?** NO — packing-derived proxy (buried + low CA degree)
- **Q7 UNSAT_POLAR explicit?** NO — proxy (buried polar + zero N/O within 6Å)
- **Q8 PROPKA residue pKa as feature?** PARTIAL — aggregates only (mean_pka, net_charge); not per-residue pKa vector
- **Q9 APBS statistics?** 14 cols: mean/median/std/quantiles/abs mean, pos/neg/strong frac, pos/neg patch count/max
- **Q10 electrostatic patch topology?** PARTIAL — APBS CC patch count/size on thresholded exposed CA only
- **Q11 dipole/multipole?** NOT_FOUND
- **Q12 OpenMM/MM energy?** NOT_FOUND
- **Q13 energy minimization?** NOT_FOUND
- **Q14 normal modes/ENM?** NOT_FOUND
- **Q15 ESM-IF features?** Per-chain ll/nll/len + fv mean/var/worst — sequence consistency scalars
- **Q16 geometric DL embedding?** NOT_FOUND (ESM-IF scalars only)
- **Q17 canonical SAP?** NOT_FOUND — code explicitly disclaims SAP formula
- **Q18 ESMFold vs ABB?** Stage3 both; Stage4 ESMFold only
- **Q19 definition differs by source?** Region mapping differs (IMGT vs author segments); same formulas
- **Q20 DeepResearch overlap?** See overlap matrix + handoff table

---

**状態:** `ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE`
