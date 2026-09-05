#!/usr/bin/env python3
"""Audit v1-prepared PDBs before FeNNix-v2 R1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
PREP = V2 / "cache/prepared_v1"
OUT = V2 / "STRUCTURE_PREPARATION_V2_AUDIT.csv"
BB = {"N", "CA", "C", "O"}


def audit_one(pdb: Path):
    p = PDBParser(QUIET=True)
    try:
        s = p.get_structure("x", str(pdb))
    except Exception as e:
        return {"ok": False, "error": f"PARSE:{e}"}
    atoms = list(s.get_atoms())
    chains = sorted({a.get_parent().get_parent().id for a in atoms})
    elems = {}
    names = []
    coords = []
    missing_bb = 0
    cys_sg_h = 0
    for res in s.get_residues():
        present = {a.get_name() for a in res}
        if res.get_resname() in (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ):
            for b in BB:
                if b == "O" and "OXT" in present:
                    continue
                if b not in present and not (b == "CA" and res.get_resname() == "GLY" and False):
                    if b not in present:
                        missing_bb += 1
        if res.get_resname() == "CYS":
            # protonated SG if HG present
            if "HG" in present or "HG1" in present:
                cys_sg_h += 1
        for a in res:
            el = (a.element or "").strip().upper() or a.get_name()[0]
            elems[el] = elems.get(el, 0) + 1
            names.append((a.get_parent().get_parent().id, a.get_parent().id[1], a.get_name()))
            coords.append(a.coord)
    coords = np.asarray(coords, float)
    # duplicate atom names within residue
    from collections import Counter

    dup = sum(1 for _, c in Counter(names).items() if c > 1)
    # severe clash: heavy-heavy < 1.5 A (non-bonded approx: skip 1-2 within 2.2 of covalent)
    heavy_idx = [i for i, a in enumerate(atoms) if (a.element or "X").upper() != "H"]
    hc = coords[heavy_idx]
    severe = 0
    all_clash = 0
    min_nb = np.inf
    n = len(hc)
    # O(n^2) ok for ~2k
    for i in range(n):
        d = np.linalg.norm(hc[i + 1 :] - hc[i], axis=1)
        if len(d):
            min_nb = min(min_nb, float(d.min()))
            severe += int((d < 1.5).sum())
            all_clash += int((d < 2.0).sum())
    unsupported = [e for e in elems if e not in {"H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE"}]
    return {
        "ok": True,
        "n_atoms": len(atoms),
        "n_heavy": len(heavy_idx),
        "chains": "".join(chains),
        "n_chains": len(chains),
        "missing_bb_atom_slots": missing_bb,
        "duplicate_atom_name_slots": dup,
        "cys_with_SG_H": cys_sg_h,
        "severe_clash_count": severe,
        "all_atom_clash_lt2": all_clash,
        "min_heavy_distance": min_nb if min_nb < np.inf else np.nan,
        "unsupported_elements": ",".join(unsupported) if unsupported else "",
        "has_H": int(elems.get("H", 0) > 0),
        "error": "",
    }


def main():
    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    rows = []
    for gen in ["esmfold", "abodybuilder2"]:
        for _, r in cw.iterrows():
            pdb = PREP / f"{r.id}_{gen}_h.pdb"
            row = {"id": r.id, "generator": gen, "pdb": str(pdb), "exists": pdb.exists()}
            if not pdb.exists():
                row.update({"ok": False, "error": "MISSING"})
            else:
                row.update(audit_one(pdb))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby(["generator", "ok"]).size())
    print("severe clash median", df.groupby("generator")["severe_clash_count"].median())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
