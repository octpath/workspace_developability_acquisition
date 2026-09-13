# TmApp H/L Topology Freeze V2 (five topologies)

**Status:** FROZEN for PLM × topology matrix (T151–T156).  
**Supersedes for matrix work:** five-topology definitions below.  
**Historical ABCD freeze remains unchanged:** `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.*`

| Topology | Concept | Key flags | AbLang2 source | AbLingua source |
|----------|---------|-----------|----------------|-----------------|
| **A** | Independent H/L → dual REG → mean merge | no cross / no joint | EXP-T110 | EXP-T080 |
| **B1** | ARCH-3 unrestricted joint dual REG | `joint_hl_dual_reg=True` | EXP-T113 | EXP-T081 |
| **B2** | ARCH-4 residue-joint, chain-specific REG | `joint_hl_chain_specific_dual_reg=True` | EXP-T115 | EXP-T082 |
| **C** | ARCH-7 REG-only opposite-chain read (C0) | `use_reg_only_cross_attention=True` | EXP-T121 | EXP-T087 |
| **D** | D3 symmetric post-summary pair MLP | `pair_interaction_mode=d3_symmetric_mlp` | EXP-T149 | NEW T151 |

Common: FULL annotation, mean merge, `share_hl_encoder=True`, `d_model=128`, seed 101, DL_FOLDLOCAL_COSINE_V3.

Machine-readable: `results/TMAPP_HL_TOPOLOGY_FREEZE_V2.yaml`
