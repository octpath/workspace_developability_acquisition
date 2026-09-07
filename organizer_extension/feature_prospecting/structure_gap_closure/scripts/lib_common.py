#!/usr/bin/env python3
"""Shared target-blind helpers for structure gap closure (≤8 CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "structure_gap_closure"
FAB_DIR = FP / "fab_reconstruction"
SEQ_FAB = FAB_DIR / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
RAW_FAB = FAB_DIR / "structures/esmfold_fab"
PREP_FAB = FP / "fennix_fab_context/cache/prepared_fab"
CROSSWALK = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
CDR_CSV = FP / "cdr_sequence_index_imgt.csv"
CACHE = CTX / "cache"
RESULTS = CTX / "results"

PROBE = 1.4
MIN_SASA = 0.5
DENSITY = 0.35
MAX_POINTS = 2500
LINK = 2.0
RASA_BURIED = 0.20
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}

KYTE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
FAUCHERE = {
    "A": 0.31, "R": -1.01, "N": -0.60, "D": -0.77, "C": 1.54, "Q": -0.22, "E": -0.64,
    "G": 0.00, "H": 0.13, "I": 1.80, "L": 1.70, "K": -0.99, "M": 1.23, "F": 1.79,
    "P": 0.72, "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22,
}
BM01 = {
    "A": 0.616, "R": 0.000, "N": 0.236, "D": 0.028, "C": 0.680, "Q": 0.251, "E": 0.043,
    "G": 0.501, "H": 0.165, "I": 0.943, "L": 0.943, "K": 0.283, "M": 0.738, "F": 1.000,
    "P": 0.711, "S": 0.359, "T": 0.450, "W": 0.878, "Y": 0.880, "V": 0.825,
}
BM_SAP = {k: float(v - BM01["G"]) for k, v in BM01.items()}

SCALES = {
    "KD": (KYTE, 0.0),
    "FP": (FAUCHERE, 0.5),
    "BM": (BM_SAP, 0.0),
}
ARO = set("FWY")
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0, "E": 223.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
    "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


class _HeavySelect(Select):
    def accept_atom(self, atom):
        return atom.element.strip().upper() not in ("H", "")


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_fab_meta() -> pd.DataFrame:
    return pd.read_csv(SEQ_FAB).set_index("id")


def load_cdr() -> dict:
    cdr = pd.read_csv(CDR_CSV)
    return {aid: g.to_dict("records") for aid, g in cdr.groupby("id")}


def load_crosswalk() -> pd.DataFrame:
    return pd.read_csv(CROSSWALK).set_index("id")


def load_hl_seqs() -> pd.DataFrame:
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def map_chains(model, heavy: str, light: str):
    chain_seqs = {}
    for ch in model.get_chains():
        aas = [aa1(r.get_resname()) for r in ch.get_residues() if is_aa(r, standard=True)]
        chain_seqs[ch.id] = "".join(aas)
    pdb_to_hl = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            pdb_to_hl[cid] = "H"
        elif seq == light:
            pdb_to_hl[cid] = "L"
    if set(pdb_to_hl.values()) != {"H", "L"}:
        # Fab: allow prefix match (VH+CH1 / VL+CL)
        for cid, seq in chain_seqs.items():
            if heavy.startswith(seq) or seq.startswith(heavy[: min(80, len(heavy))]):
                pass
        for cid, seq in chain_seqs.items():
            if cid in pdb_to_hl:
                continue
            if len(seq) >= len(heavy) - 5 and seq[: len(heavy) - 5] == heavy[: len(heavy) - 5]:
                pdb_to_hl[cid] = "H"
            elif len(seq) >= len(light) - 5 and seq[: len(light) - 5] == light[: len(light) - 5]:
                pdb_to_hl[cid] = "L"
            elif seq == heavy or (len(heavy) > 50 and seq[:50] == heavy[:50]):
                pdb_to_hl[cid] = "H"
            elif seq == light or (len(light) > 50 and seq[:50] == light[:50]):
                pdb_to_hl[cid] = "L"
    if set(pdb_to_hl.values()) != {"H", "L"}:
        # length heuristic for Fab: longer chain often heavy
        if len(chain_seqs) == 2 and not pdb_to_hl:
            items = sorted(chain_seqs.items(), key=lambda x: -len(x[1]))
            pdb_to_hl = {items[0][0]: "H", items[1][0]: "L"}
        elif len(chain_seqs) >= 2:
            # pick best identity
            best_h, best_l = None, None
            sc_h, sc_l = -1, -1
            for cid, seq in chain_seqs.items():
                n = min(len(seq), len(heavy))
                s = sum(a == b for a, b in zip(seq[:n], heavy[:n])) / max(n, 1)
                if s > sc_h:
                    sc_h, best_h = s, cid
            for cid, seq in chain_seqs.items():
                if cid == best_h:
                    continue
                n = min(len(seq), len(light))
                s = sum(a == b for a, b in zip(seq[:n], light[:n])) / max(n, 1)
                if s > sc_l:
                    sc_l, best_l = s, cid
            if best_h and best_l and best_h != best_l and sc_h > 0.8 and sc_l > 0.8:
                pdb_to_hl = {best_h: "H", best_l: "L"}
    if set(pdb_to_hl.values()) != {"H", "L"}:
        return None, f"chain_map_fail:{pdb_to_hl}|{ {k:len(v) for k,v in chain_seqs.items()} }"
    return pdb_to_hl, None


def domain_of(chain_hl: str, seq_i: int, vh_len: int, vl_len: int) -> str:
    if chain_hl == "H":
        return "VH" if seq_i < vh_len else "CH1"
    return "VL" if seq_i < vl_len else "CL"


def fibonacci_sphere(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1 - (i / max(n - 1, 1)) * 2
    r = np.sqrt(np.clip(1 - y * y, 0, None))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def connected_components(mask: np.ndarray, coords: np.ndarray, link: float) -> list[list[int]]:
    idxs = np.where(mask)[0].tolist()
    if not idxs:
        return []
    parent = {i: i for i in idxs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_i, a in enumerate(idxs):
        for b in idxs[a_i + 1 :]:
            if np.linalg.norm(coords[a] - coords[b]) <= link:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    comps: dict[int, list[int]] = {}
    for i in idxs:
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def write_heavy_tmp(structure, path: Path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), _HeavySelect())
