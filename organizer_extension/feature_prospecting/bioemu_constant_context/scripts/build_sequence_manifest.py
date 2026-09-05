#!/usr/bin/env python3
"""Build BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv from frozen fab_reconstruction sequences."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FAB = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
OUT = ROOT / "organizer_extension/feature_prospecting/bioemu_constant_context/BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv"


def main():
    df = pd.read_csv(FAB)
    rows = []
    for r in df.itertuples(index=False):
        vh_len = int(r.VH_len_used)
        vl_len = int(r.VL_len_used)
        h_len = int(r.heavy_length)
        l_len = int(r.light_length)
        rows.append(
            {
                "id": r.id,
                "light_locus": r.light_locus,
                "CL_sequence_id": r.CL_sequence_id,
                "CH1_sequence_id": r.CH1_sequence_id,
                "VH_len": vh_len,
                "CH1_len": h_len - vh_len,
                "VL_len": vl_len,
                "CL_len": l_len - vl_len,
                "VH_start": 0,
                "VH_end": vh_len,
                "CH1_start": vh_len,
                "CH1_end": h_len,
                "VL_start": 0,
                "VL_end": vl_len,
                "CL_start": vl_len,
                "CL_end": l_len,
                "heavy_fab_seq": r.heavy_fab_seq,
                "light_fab_seq": r.light_fab_seq,
                "heavy_length": h_len,
                "light_length": l_len,
                "heavy_V_gene": getattr(r, "heavy_V_gene", ""),
                "light_V_gene": getattr(r, "light_V_gene", ""),
                "reconstruction_class": r.reconstruction_class,
                "boundary_indexing": "0-based_half_open",
                "heavy_arm_label": "UNPAIRED_HEAVY_CONTEXT",
                "light_arm_label": "LIGHT_PRIMARY_VL_PLUS_CL",
            }
        )
    out = pd.DataFrame(rows)
    assert len(out) == 324
    out.to_csv(OUT, index=False)
    print("wrote", OUT, "n=", len(out), "kappa=", (out.light_locus == "kappa").sum(), "lambda=", (out.light_locus == "lambda").sum())


if __name__ == "__main__":
    main()
