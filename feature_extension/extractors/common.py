"""Shared helpers for participant-facing PDB feature extractors."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

# Tien et al. / Stage4-style max ASA (Å²) for RASA
MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

HYDROPHOBIC = set("AILMFVWY")
POLAR = set("STNQDEKRH")
AROMATIC = set("FWY")

# AROMATIC-TOPO frozen thresholds (FEATURE_SPEC.json)
RASA_EXPOSED = 0.20
RASA_STRONG = 0.50
PROBE_RADIUS = 1.4
N_POINTS = 100


def load_structure(pdb_path: str | Path, structure_id: str = "struct"):
    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB not found: {path}")
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, str(path))


def compute_sasa(structure) -> None:
    """Attach per-residue SASA via Bio.PDB ShrakeRupley (in-place)."""
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(structure, level="R")


def iter_residues(structure, hetero: bool = False):
    for model in structure:
        for chain in model:
            for res in chain:
                het = res.id[0].strip()
                if not hetero and het not in ("", "W"):
                    # skip hetero; keep standard residues (hetflag ' ')
                    if het != " ":
                        continue
                if res.id[0] != " ":
                    continue
                if res.get_resname() in ("HOH", "WAT"):
                    continue
                yield chain.id, res


def aa1(res) -> str | None:
    from Bio.PDB.Polypeptide import protein_letters_3to1

    return protein_letters_3to1.get(res.get_resname().strip().upper())


def residue_sasa_rows(structure) -> list[dict]:
    rows = []
    for chain_id, res in iter_residues(structure):
        aa = aa1(res)
        if aa is None:
            continue
        sasa = float(getattr(res, "sasa", np.nan))
        max_asa = MAX_ASA.get(aa, np.nan)
        rasa = float(sasa / max_asa) if max_asa and np.isfinite(sasa) else np.nan
        try:
            ca = res["CA"].coord
        except KeyError:
            ca = np.array([np.nan, np.nan, np.nan])
        rows.append(
            {
                "chain": chain_id,
                "resseq": int(res.id[1]),
                "amino_acid": aa,
                "sasa": sasa,
                "rasa": rasa,
                "ca": ca,
                "is_exposed": bool(np.isfinite(rasa) and rasa >= RASA_EXPOSED),
                "is_strongly_exposed": bool(np.isfinite(rasa) and rasa >= RASA_STRONG),
            }
        )
    return rows


def finite_sum(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.sum()) if arr.size else 0.0
