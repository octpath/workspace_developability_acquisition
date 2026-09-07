#!/usr/bin/env python3
"""Build Fab-geometry Fv extracts with hydrogens (OpenMM/PDBFixer; .venv_b1)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from openmm.app import PDBFile
from pdbfixer import PDBFixer

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
SEQ = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    fab = pd.read_csv(SEQ)
    if args.ids:
        fab = fab[fab.id.isin(args.ids)]
    for _, row in fab.iterrows():
        ab = row.id
        vh, vl = int(row.VH_len_used), int(row.VL_len_used)
        ren = CTX / "cache/renumbered_fab" / f"{ab}.pdb"
        prepared = CTX / "cache/prepared_fab" / f"{ab}_prepared.pdb"
        src = ren if ren.exists() else prepared
        if not src.exists():
            print("SKIP", ab)
            continue
        out_h = CTX / "cache/fabgeom_fv" / f"{ab}_fv_h.pdb"
        if out_h.exists():
            continue
        raw = CTX / "cache/fabgeom_fv" / f"{ab}_fv.pdb"
        raw.parent.mkdir(parents=True, exist_ok=True)
        class FV(Select):
            def accept_residue(self, residue):
                ch = residue.get_parent().id
                rs = residue.id[1]
                return 1 if (ch == "A" and rs <= vh) or (ch == "B" and rs <= vl) else 0
        st = PDBParser(QUIET=True).get_structure("x", str(src))
        io = PDBIO(); io.set_structure(st); io.save(str(raw), FV())
        fixer = PDBFixer(filename=str(raw))
        fixer.findMissingResidues(); fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms(); fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.0)
        with open(out_h, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
        print("OK", ab, flush=True)

if __name__ == "__main__":
    main()
