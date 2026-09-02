#!/usr/bin/env python3
"""Shared structure helpers for Physical Batch1 (target-blind)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CDR_CSV = FP / "cdr_sequence_index_imgt.csv"

PROBE = 1.4
N_POINTS = 100
RASA_EXPOSED = 0.20
RASA_STRONG = 0.50
IFACE_CA = 5.0
PATCH_R = 8.0
LOCAL_R = 10.0

MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# Black & Mould 1991 (Anal. Biochem. 193:72-82) 0–1 tabulated values
BLACK_MOULD_01 = {
    "A": 0.616, "R": 0.000, "N": 0.236, "D": 0.028, "C": 0.680,
    "Q": 0.251, "E": 0.043, "G": 0.501, "H": 0.165, "I": 0.943,
    "L": 0.943, "K": 0.283, "M": 0.738, "F": 1.000, "P": 0.711,
    "S": 0.359, "T": 0.450, "W": 0.878, "Y": 0.880, "V": 0.825,
}
# SAP literature: center at Gly so Gly=0
BLACK_MOULD_SAP = {k: float(v - BLACK_MOULD_01["G"]) for k, v in BLACK_MOULD_01.items()}

AROMATIC = set("FWY")
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_sequences() -> pd.DataFrame:
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def load_cdr_map() -> dict:
    cdr = pd.read_csv(CDR_CSV)
    return {aid: g.to_dict("records") for aid, g in cdr.groupby("id")}


def build_residue_table(pdb_path: Path, heavy: str, light: str, cdr_rows: list[dict]):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    sr = ShrakeRupley(probe_radius=PROBE, n_points=N_POINTS)
    sr.compute(model, level="R")

    chain_seqs = {}
    for ch in model.get_chains():
        aas = []
        for res in ch.get_residues():
            if is_aa(res, standard=True):
                aas.append(aa1(res.get_resname()))
        chain_seqs[ch.id] = "".join(aas)

    pdb_to_hl = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            pdb_to_hl[cid] = "H"
        elif seq == light:
            pdb_to_hl[cid] = "L"
    if set(pdb_to_hl.values()) != {"H", "L"}:
        return None, f"chain_map_fail:{pdb_to_hl}"

    cdr_map = {(r["chain"], int(r["sequence_index"])): r for r in cdr_rows}
    rows = []
    for ch in model.get_chains():
        if ch.id not in pdb_to_hl:
            continue
        hl = pdb_to_hl[ch.id]
        seq_i = 0
        for res in ch.get_residues():
            if not is_aa(res, standard=True):
                continue
            a = aa1(res.get_resname())
            sasa = float(getattr(res, "sasa", np.nan))
            maxasa = MAX_ASA.get(a)
            rasa = sasa / maxasa if maxasa else np.nan
            ca = res["CA"].coord.copy() if "CA" in res else None
            atoms = []
            for atom in res.get_atoms():
                atoms.append((atom.get_name().strip(), atom.coord.copy(), atom.element.strip().upper()))
            c = cdr_map.get((hl, seq_i))
            rows.append(
                {
                    "pdb_chain": ch.id,
                    "chain": hl,
                    "sequence_index": seq_i,
                    "amino_acid": a,
                    "resname3": res.get_resname().strip().upper(),
                    "pdb_resseq": int(res.id[1]),
                    "sasa": sasa,
                    "rasa": rasa,
                    "is_exposed": bool(np.isfinite(rasa) and rasa >= RASA_EXPOSED),
                    "is_strongly_exposed": bool(np.isfinite(rasa) and rasa >= RASA_STRONG),
                    "is_buried": bool(np.isfinite(rasa) and rasa < RASA_EXPOSED),
                    "is_cdr": bool(c["is_cdr"]) if c else False,
                    "is_framework": bool(c["is_framework"]) if c else False,
                    "region": c["region"] if c else "UNMAPPED",
                    "ca": ca,
                    "atoms": atoms,
                    "res_obj": res,
                }
            )
            seq_i += 1

    for hl, expected in [("H", heavy), ("L", light)]:
        got = "".join(r["amino_acid"] for r in rows if r["chain"] == hl)
        if got != expected:
            return None, f"seq_mismatch_{hl}"

    h = [r for r in rows if r["chain"] == "H" and r["ca"] is not None]
    l = [r for r in rows if r["chain"] == "L" and r["ca"] is not None]
    iface = set()
    for ri in h:
        for rj in l:
            if np.linalg.norm(ri["ca"] - rj["ca"]) <= IFACE_CA:
                iface.add(("H", ri["sequence_index"]))
                iface.add(("L", rj["sequence_index"]))
    for r in rows:
        r["is_vh_vl_interface"] = (r["chain"], r["sequence_index"]) in iface
    return rows, None


def connected_components(indices: list[int], coords: np.ndarray, radius: float) -> list[list[int]]:
    n = len(indices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= radius:
                union(i, j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(indices[i])
    return list(comps.values())
