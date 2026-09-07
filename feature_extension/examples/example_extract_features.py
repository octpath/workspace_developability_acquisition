#!/usr/bin/env python3
"""Example: extract SASA + aromatic features from one bundled ESMFold Fv PDB."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from feature_extension.extractors import extract_aromatic, extract_sasa  # noqa: E402


def main() -> None:
    pdb_dir = ROOT / "data" / "esmfold_fv"
    pdbs = sorted(pdb_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDBs found in {pdb_dir}")
    pdb = pdbs[0]
    sasa = extract_sasa(pdb)
    arom = extract_aromatic(pdb)
    row = {"id": pdb.stem, **sasa, **arom}
    df = pd.DataFrame([row])
    print(df.T.to_string())


if __name__ == "__main__":
    main()
