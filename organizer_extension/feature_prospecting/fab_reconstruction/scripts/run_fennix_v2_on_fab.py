#!/usr/bin/env python3
"""
Wrapper to run FeNNix-v2 protocol on reconstructed Fab PDBs.

Does NOT execute TmApp scoring. Reuses foundation_stability_v2 relaxation /
curvature protocol by importing its helpers where possible, with Fab domain maps:

  VH, VL, CH1, CL, variable region, constant region

Usage (future session):
  python run_fennix_v2_on_fab.py --pdb structures/esmfold_fab/ADI-xxxxx.pdb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
FAB_ROOT = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
V2 = ROOT / "organizer_extension/feature_prospecting/foundation_stability_v2"
SEQ = FAB_ROOT / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"


def domain_mask_from_lengths(vh: int, ch1: int, vl: int, cl: int):
    """Residue index ranges on chains A (heavy) and B (light), 0-based."""
    return {
        "VH": {"chain": "A", "start": 0, "end": vh},
        "CH1": {"chain": "A", "start": vh, "end": vh + ch1},
        "VL": {"chain": "B", "start": 0, "end": vl},
        "CL": {"chain": "B", "start": vl, "end": vl + cl},
        "variable_region": [("A", 0, vh), ("B", 0, vl)],
        "constant_region": [("A", vh, vh + ch1), ("B", vl, vl + cl)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Only emit domain map JSON")
    args = ap.parse_args()

    import pandas as pd

    fab = pd.read_csv(SEQ)
    aid = args.id or args.pdb.stem
    row = fab[fab["id"] == aid].iloc[0]
    vh, vl = int(row["VH_len_used"]), int(row["VL_len_used"])
    ch1 = int(row["heavy_length"]) - vh
    cl = int(row["light_length"]) - vl
    mapping = domain_mask_from_lengths(vh, ch1, vl, cl)
    mapping["antibody_id"] = aid
    mapping["pdb"] = str(args.pdb)
    mapping["reconstruction_class"] = row["reconstruction_class"]
    mapping["fennix_v2_spec"] = str(V2 / "FOUNDATION_STABILITY_V2_SPEC.json")
    mapping["instruction"] = (
        "Import/reuse foundation_stability_v2/scripts/fennix_v2_relax_curvature.py "
        "protocol without duplicating physics; do not run TmApp scoring here."
    )
    out = FAB_ROOT / "cache" / "fennix_fab_domain_maps" / f"{aid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2))
    print(json.dumps(mapping, indent=2))
    if args.dry_run:
        return
    print(
        "READY: domain map written. Execute FeNNix-v2 in a dedicated session "
        "reusing V2 codepaths; this wrapper intentionally does not score TmApp."
    )


if __name__ == "__main__":
    main()
