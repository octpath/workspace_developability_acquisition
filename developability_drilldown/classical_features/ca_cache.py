#!/usr/bin/env python3
"""Residue Cα coordinate cache aligned to competition sequences (ESMFold Fv)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from .rasa_cache import _three_to_one  # same PDB AA mapping

REPO = Path(__file__).resolve().parents[2]
PDB_DIR = REPO / "feature_extension" / "data" / "esmfold_fv"
CACHE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "classical_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_ca_chains(pdb_path: Path, heavy: str, light: str) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Return (H_ca[Lh,3], L_ca[Ll,3], mapped_h, mapped_l); missing CA -> NaN rows."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("ab", str(pdb_path))
    chains: dict[str, list[tuple[str, np.ndarray]]] = {}
    for ch in struct.get_chains():
        rows = []
        for res in ch:
            if res.id[0] != " ":
                continue
            aa = _three_to_one(res.get_resname())
            if aa is None:
                continue
            if "CA" not in res:
                rows.append((aa, np.array([np.nan, np.nan, np.nan], dtype=np.float64)))
            else:
                rows.append((aa, np.asarray(res["CA"].get_coord(), dtype=np.float64)))
        if rows:
            chains[ch.id] = rows

    h_ca = np.full((len(heavy), 3), np.nan, dtype=np.float64)
    l_ca = np.full((len(light), 3), np.nan, dtype=np.float64)
    mapped_h = mapped_l = False

    def try_assign(cid: str, rows: list, target: str, out: np.ndarray) -> bool:
        seq = "".join(a for a, _ in rows)
        if len(seq) != len(target) or seq != target:
            return False
        for i, (_, xyz) in enumerate(rows):
            out[i] = xyz
        return True

    for cid, rows in chains.items():
        if cid in ("H", "A") and not mapped_h:
            mapped_h = try_assign(cid, rows, heavy, h_ca)
        elif cid in ("L", "B") and not mapped_l:
            mapped_l = try_assign(cid, rows, light, l_ca)
    if not mapped_h or not mapped_l:
        for cid, rows in chains.items():
            if not mapped_h:
                mapped_h = try_assign(cid, rows, heavy, h_ca)
            if not mapped_l:
                mapped_l = try_assign(cid, rows, light, l_ca)
    return h_ca.astype(np.float32), l_ca.astype(np.float32), mapped_h, mapped_l


def build_or_load_ca_cache(dev: pd.DataFrame, test: pd.DataFrame) -> dict:
    cache_h = CACHE_DIR / "ca_heavy.npy"
    cache_l = CACHE_DIR / "ca_light.npy"
    cache_ids = CACHE_DIR / "ca_ids.npy"
    meta_path = CACHE_DIR / "ca_meta.json"
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
    H = np.full((len(ids), max_h, 3), np.nan, np.float32)
    L = np.full((len(ids), max_l, 3), np.nan, np.float32)
    n_mapped = 0
    n_missing_ca = 0
    for i, r in enumerate(all_df.itertuples(index=False)):
        pdb = PDB_DIR / f"{r.id}.pdb"
        if not pdb.exists():
            continue
        h, l, mh, ml = extract_ca_chains(pdb, str(r.heavy), str(r.light))
        H[i, : len(h)] = h
        L[i, : len(l)] = l
        n_missing_ca += int(np.sum(~np.isfinite(h))) + int(np.sum(~np.isfinite(l)))
        if mh and ml:
            n_mapped += 1
        if (i + 1) % 50 == 0:
            print(f"CA {i+1}/{len(ids)} mapped={n_mapped}", flush=True)
    np.save(cache_h, H)
    np.save(cache_l, L)
    np.save(cache_ids, np.array(ids, dtype=object))
    meta = {
        "n_ids": len(ids),
        "n_mapped_both": n_mapped,
        "n_missing_ca_residues": n_missing_ca,
        "source": str(PDB_DIR),
        "method": "Bio.PDB CA coordinates; chain map H/A L/B then sequence match",
        "units": "angstrom",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {"ids": ids, "H": H, "L": L, "meta": meta}


def train_median_pair_distance(
    ca_heavy: np.ndarray,
    ca_light: np.ndarray,
    heavy_mask: np.ndarray,
    light_mask: np.ndarray,
    train_idxs: list[int],
) -> float:
    """Median valid within-chain Cα pairwise distance over TRAIN antibodies only."""
    dists: list[float] = []
    for i in train_idxs:
        for ca, mask in ((ca_heavy[i], heavy_mask[i]), (ca_light[i], light_mask[i])):
            n = int(mask.sum())
            if n < 2:
                continue
            xyz = ca[:n]
            if not np.isfinite(xyz).all():
                # drop non-finite residues
                ok = np.isfinite(xyz).all(axis=1)
                xyz = xyz[ok]
                n = len(xyz)
                if n < 2:
                    continue
            # upper triangle
            d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            iu = np.triu_indices(n, k=1)
            dists.append(d[iu].astype(np.float64))
    if not dists:
        raise RuntimeError("no valid Cα pairs in TRAIN for median ell init")
    all_d = np.concatenate(dists)
    return float(np.median(all_d))
