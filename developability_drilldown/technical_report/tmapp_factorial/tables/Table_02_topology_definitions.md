# Table 2 — Topology definitions

| topology_id   | topology_display              | legacy_id   | residue_level_HL_communication   | REG_cross_chain_communication   | point_of_HL_interaction             |
|:--------------|:------------------------------|:------------|:---------------------------------|:--------------------------------|:------------------------------------|
| SEP           | Separate H/L                  | A           | False                            | False                           | none_before_merge                   |
| JOINT         | Full Joint                    | B1          | True                             | True                            | joint_transformer                   |
| REG-SEP       | Joint Residues, Separate REGs | B2          | True                             | False                           | joint_residues_chain_specific_regs  |
| XREG          | Cross-REG Read                | C           | False                            | True                            | reg_queries_opposite_chain_residues |
| FUSE          | REG Fusion                    | D           | False                            | True                            | post_encoding_reg_reg_d3            |
