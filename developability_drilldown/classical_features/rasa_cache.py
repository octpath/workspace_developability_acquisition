#!/usr/bin/env python3
"""Residue RASA cache aligned to competition sequences (ESMFold Fv, Tien2013)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

# Tien2013 / Wilke MaxASA (Å²) — same as feature_extension/extractors/common.py
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
RASA_EXPOSED = 0.20  # historical participant default

REPO = Path(__file__).resolve().parents[2]
PDB_DIR = REPO / "feature_extension" / "data" / "esmfold_fv"
CACHE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "classical_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _three_to_one(resname: str) -> str | None:
    from Bio.Data.IUPACData import protein_letters_3to1

    key = resname.strip().capitalize()
    return protein_letters_3to1.get(key) or protein_letters_3to1.get(resname.upper())


def compute_rasa_pdb(pdb_path: Path, heavy: str, light: str) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("ab", str(pdb_path))
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(struct, level="R")
    chains = {}
    for ch in struct.get_chains():
        rows = []
        for res in ch:
            if res.id[0] != " ":
                continue
            aa = _three_to_one(res.get_resname())
            if aa is None:
                continue
            sasa = float(res.sasa) if hasattr(res, "sasa") else np.nan
            maxasa = MAX_ASA.get(aa, np.nan)
            rasa = (sasa / maxasa) if maxasa and np.isfinite(sasa) else np.nan
            rows.append((aa, rasa))
        if rows:
            chains[ch.id] = rows

    h_rasa = np.full(len(heavy), np.nan)
    l_rasa = np.full(len(light), np.nan)
    mapped_h = mapped_l = False

    # Prefer H/L naming; else match by AA sequence
    for cid, rows in chains.items():
        seq = "".join(a for a, _ in rows)
        rasas = np.array([r for _, r in rows], float)
        if cid in ("H", "A") and len(seq) == len(heavy) and seq == heavy:
            h_rasa, mapped_h = rasas, True
        elif cid in ("L", "B") and len(seq) == len(light) and seq == light:
            l_rasa, mapped_l = rasas, True
    if not mapped_h or not mapped_l:
        for cid, rows in chains.items():
            seq = "".join(a for a, _ in rows)
            rasas = np.array([r for _, r in rows], float)
            if not mapped_h and len(seq) == len(heavy) and seq == heavy:
                h_rasa, mapped_h = rasas, True
            elif not mapped_l and len(seq) == len(light) and seq == light:
                l_rasa, mapped_l = rasas, True
    return h_rasa, l_rasa, mapped_h, mapped_l


def build_or_load_rasa_cache(dev: pd.DataFrame, test: pd.DataFrame) -> dict:
    cache_h = CACHE_DIR / "rasa_heavy.npy"
    cache_l = CACHE_DIR / "rasa_light.npy"
    cache_ids = CACHE_DIR / "rasa_ids.npy"
    meta_path = CACHE_DIR / "rasa_meta.json"
    all_df = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    all_df["id"] = all_df["id"].astype(str)
    ids = all_df["id"].tolist()
    if cache_h.exists() and cache_l.exists() and cache_ids.exists():
        cached_ids = [str(x) for x in np.load(cache_ids, allow_pickle=True)]
        if cached_ids == ids:
            return {
                "ids": ids,
                "H": np.load(cache_h),
                "L": np.load(cache_l),
                "meta": json.loads(meta_path.read_text()) if meta_path.exists() else {},
            }

    max_h = max(len(s) for s in all_df["heavy"])
    max_l = max(len(s) for s in all_df["light"])
    H = np.full((len(ids), max_h), np.nan, np.float32)
    L = np.full((len(ids), max_l), np.nan, np.float32)
    n_mapped = 0
    for i, r in enumerate(all_df.itertuples(index=False)):
        pdb = PDB_DIR / f"{r.id}.pdb"
        if not pdb.exists():
            continue
        h, l, mh, ml = compute_rasa_pdb(pdb, str(r.heavy), str(r.light))
        H[i, : len(h)] = h
        L[i, : len(l)] = l
        if mh and ml:
            n_mapped += 1
        if (i + 1) % 50 == 0:
            print(f"RASA {i+1}/{len(ids)} mapped={n_mapped}", flush=True)
    np.save(cache_h, H)
    np.save(cache_l, L)
    np.save(cache_ids, np.array(ids, dtype=object))
    meta = {
        "n_ids": len(ids),
        "n_mapped_both": n_mapped,
        "source": str(PDB_DIR),
        "method": "Bio.PDB.SASA.ShrakeRupley probe=1.4 n_points=100 MaxASA=Tien2013",
        "exposed_threshold": RASA_EXPOSED,
        "clip_for_weights": "[0,1]",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"ids": ids, "H": H, "L": L, "meta": meta}
