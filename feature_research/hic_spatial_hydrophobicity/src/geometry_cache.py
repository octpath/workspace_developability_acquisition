#!/usr/bin/env python3
"""Residue geometry + SASA cache for Fv ESMFold structures (target-blind)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path(__file__).resolve().parents[3]
DRILL = ROOT / "developability_drilldown"
FP = ROOT / "organizer_extension/feature_prospecting"
sys.path.insert(0, str(FP))

from common.structure_utils import MAX_ASA, load_cdr_map, load_sequences  # noqa: E402

CROSSWALK = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
PROBE = 1.4
N_POINTS = 100
BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})
CACHE_DIR = Path(__file__).resolve().parents[1] / "features" / "cache"


def _aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _centroid(atoms: list[tuple[str, np.ndarray, str]], aa: str, ca: np.ndarray | None):
    coords = []
    for name, coord, el in atoms:
        n = name.strip().upper()
        if n in BACKBONE:
            continue
        if (el or "").upper() == "H" or n.startswith("H"):
            continue
        coords.append(coord)
    if coords:
        return np.mean(np.stack(coords), axis=0), "sidechain"
    if aa == "G" and ca is not None:
        return ca.copy(), "gly_ca"
    if ca is not None:
        return ca.copy(), "ca_fallback"
    return None, "invalid"


def _sidechain_heavy(atoms, aa, ca):
    pts = []
    for name, coord, el in atoms:
        n = name.strip().upper()
        if n in BACKBONE:
            continue
        if (el or "").upper() == "H" or n.startswith("H"):
            continue
        pts.append(coord)
    if pts:
        return np.stack(pts)
    if ca is not None:
        return ca.reshape(1, 3)
    return np.zeros((0, 3))


def process_pdb(pdb_path: Path, heavy: str, light: str, cdr_rows: list[dict], n_points: int = N_POINTS):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    ShrakeRupley(probe_radius=PROBE, n_points=n_points).compute(model, level="A")

    chain_seqs = {}
    for ch in model.get_chains():
        aas = [_aa1(r.get_resname()) for r in ch.get_residues() if is_aa(r, standard=True)]
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
            aa = _aa1(res.get_resname())
            atoms = []
            total_sasa = 0.0
            bb_sasa = 0.0
            sc_sasa = 0.0
            for atom in res.get_atoms():
                name = atom.get_name().strip()
                el = atom.element.strip().upper()
                sasa = float(getattr(atom, "sasa", 0.0) or 0.0)
                atoms.append((name, atom.coord.copy(), el))
                total_sasa += sasa
                if name.upper() in BACKBONE:
                    bb_sasa += sasa
                elif el != "H" and not name.upper().startswith("H"):
                    sc_sasa += sasa
            ca = res["CA"].coord.copy() if "CA" in res else None
            cent, prov = _centroid(atoms, aa, ca)
            sc_heavy = _sidechain_heavy(atoms, aa, ca)
            c = cdr_map.get((hl, seq_i))
            region = c["region"] if c else "UNMAPPED"
            is_cdr = bool(c["is_cdr"]) if c else False
            tien = MAX_ASA.get(aa, np.nan)
            rows.append(
                {
                    "chain": hl,
                    "residue_index": seq_i,
                    "aa": aa,
                    "region": region,
                    "is_cdr": is_cdr,
                    "is_h_cdr3": hl == "H" and region == "CDR3",
                    "is_l_cdr3": hl == "L" and region == "CDR3",
                    "centroid": cent,
                    "centroid_prov": prov,
                    "sc_heavy": sc_heavy,
                    "total_SASA": total_sasa,
                    "backbone_SASA": bb_sasa,
                    "sidechain_SASA": sc_sasa,
                    "Tien_MaxASA": tien,
                    "total_rASA_Tien": float(np.clip(total_sasa / tien, 0, 1)) if tien else np.nan,
                    "sidechain_over_Tien": float(np.clip(sc_sasa / tien, 0, 1)) if tien else np.nan,
                }
            )
            seq_i += 1
    for hl, expected in [("H", heavy), ("L", light)]:
        got = "".join(r["aa"] for r in rows if r["chain"] == hl)
        if got != expected:
            return None, f"seq_mismatch_{hl}"
    return rows, None


def build_or_load_cache(force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / "residue_geometry_sasa.parquet"
    meta_path = CACHE_DIR / "residue_geometry_meta.json"
    if out.exists() and meta_path.exists() and not force:
        return out

    cw = pd.read_csv(CROSSWALK)
    seqs = load_sequences()
    cdr_map = load_cdr_map()
    flat = []
    errors = []
    for _, crow in cw.iterrows():
        aid = str(crow["id"])
        if aid not in seqs.index:
            errors.append({"id": aid, "error": "no_seq"})
            continue
        pdb = Path(str(crow["esmfold_canonical_path"]))
        if not pdb.exists():
            errors.append({"id": aid, "error": "no_pdb"})
            continue
        rows, err = process_pdb(pdb, str(seqs.loc[aid, "heavy"]), str(seqs.loc[aid, "light"]), cdr_map.get(aid, []))
        if err:
            errors.append({"id": aid, "error": err})
            continue
        sh = _sha_file(pdb)
        for r in rows:
            c = r["centroid"]
            flat.append(
                {
                    "id": aid,
                    "chain": r["chain"],
                    "residue_index": r["residue_index"],
                    "aa": r["aa"],
                    "region": r["region"],
                    "is_cdr": r["is_cdr"],
                    "is_h_cdr3": r["is_h_cdr3"],
                    "is_l_cdr3": r["is_l_cdr3"],
                    "centroid_x": float(c[0]) if c is not None else np.nan,
                    "centroid_y": float(c[1]) if c is not None else np.nan,
                    "centroid_z": float(c[2]) if c is not None else np.nan,
                    "centroid_prov": r["centroid_prov"],
                    "n_sc_heavy": int(len(r["sc_heavy"])),
                    "sc_heavy": r["sc_heavy"].astype(np.float32).tobytes(),
                    "total_SASA": r["total_SASA"],
                    "backbone_SASA": r["backbone_SASA"],
                    "sidechain_SASA": r["sidechain_SASA"],
                    "Tien_MaxASA": r["Tien_MaxASA"],
                    "total_rASA_Tien": r["total_rASA_Tien"],
                    "sidechain_over_Tien": r["sidechain_over_Tien"],
                    "structure_sha256": sh,
                    "structure_path": str(pdb),
                }
            )
        if len(flat) % 5000 < 50:
            print(f"geometry {aid} n_abs={len({x['id'] for x in flat})}", flush=True)

    df = pd.DataFrame(flat)
    assert df["id"].nunique() == 324, (df["id"].nunique(), errors[:5])
    df.to_parquet(out, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "n_antibodies": int(df["id"].nunique()),
                "n_residues": len(df),
                "probe": PROBE,
                "n_points": N_POINTS,
                "structure_scope": "Fv",
                "errors": errors,
                "sha256": _sha_file(out),
            },
            indent=2
        )
    )
    print("Wrote", out, flush=True)
    return out


def load_antibody_arrays(df: pd.DataFrame, aid: str) -> dict:
    sub = df[df["id"] == aid].sort_values(["chain", "residue_index"])
    n = len(sub)
    centroids = sub[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(float)
    sc_list = []
    for b in sub["sc_heavy"].tolist():
        arr = np.frombuffer(b, dtype=np.float32).reshape(-1, 3)
        sc_list.append(arr)
    return {
        "aa": sub["aa"].tolist(),
        "chain": sub["chain"].tolist(),
        "region": sub["region"].tolist(),
        "is_cdr": sub["is_cdr"].to_numpy(bool),
        "is_h_cdr3": sub["is_h_cdr3"].to_numpy(bool),
        "is_l_cdr3": sub["is_l_cdr3"].to_numpy(bool),
        "centroids": centroids,
        "sc_heavy": sc_list,
        "total_rASA_Tien": sub["total_rASA_Tien"].to_numpy(float),
        "sidechain_over_Tien": sub["sidechain_over_Tien"].to_numpy(float),
        "sidechain_SASA": sub["sidechain_SASA"].to_numpy(float),
        "total_SASA": sub["total_SASA"].to_numpy(float),
        "residue_index": sub["residue_index"].to_numpy(int),
    }


if __name__ == "__main__":
    build_or_load_cache(force="--force" in sys.argv)
