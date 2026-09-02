#!/usr/bin/env python3
"""Shared Advanced Batch2 surface + MLP helpers (target-blind)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import freesasa
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa

PROBE = 1.4
DENSITY = 0.35
MAX_POINTS = 2000
MIN_SASA = 0.5
ALPHA = 1.0
CUTOFF = 7.0
LINK = 2.0

FAUCHERE_PI = {
    "A": 0.31, "R": -1.01, "N": -0.60, "D": -0.77, "C": 1.54,
    "Q": -0.22, "E": -0.64, "G": 0.00, "H": 0.13, "I": 1.80,
    "L": 1.70, "K": -0.99, "M": 1.23, "F": 1.79, "P": 0.72,
    "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22,
}
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}


class _HeavySelect(Select):
    def accept_atom(self, atom):
        return atom.element.strip().upper() not in ("H", "")


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def fibonacci_sphere(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1 - (i / max(n - 1, 1)) * 2
    r = np.sqrt(np.clip(1 - y * y, 0, None))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def map_chains(model, heavy: str, light: str):
    chain_seqs = {}
    for ch in model.get_chains():
        aas = [aa1(res.get_resname()) for res in ch.get_residues() if is_aa(res, standard=True)]
        chain_seqs[ch.id] = "".join(aas)
    pdb_to_hl = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            pdb_to_hl[cid] = "H"
        elif seq == light:
            pdb_to_hl[cid] = "L"
    if set(pdb_to_hl.values()) != {"H", "L"}:
        return None, f"chain_map_fail:{pdb_to_hl}"
    return pdb_to_hl, None


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


def cdr_index_set(cdr_rows: list[dict]) -> set[tuple[str, int]]:
    out = set()
    for r in cdr_rows:
        flag = r.get("is_cdr", False)
        if isinstance(flag, str):
            flag = flag.strip().lower() in ("1", "true", "t", "yes")
        if flag:
            out.add((str(r["chain"]), int(r["sequence_index"])))
    return out


def build_surface_and_field(pdb_path: Path, heavy: str, light: str, cdr_rows: list[dict]):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    pdb_to_hl, err = map_chains(model, heavy, light)
    if err:
        return None, err
    cdr_set = cdr_index_set(cdr_rows)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    io = PDBIO()
    io.set_structure(structure)
    io.save(tmp_path, _HeavySelect())
    try:
        # Rebuild meta from heavy-only PDB so FreeSASA atom order matches exactly
        heavy_st = parser.get_structure("h", tmp_path)
        heavy_model = next(heavy_st.get_models())
        pdb_to_hl2, err2 = map_chains(heavy_model, heavy, light)
        if err2:
            return None, err2
        meta = []
        for ch in heavy_model.get_chains():
            if ch.id not in pdb_to_hl2:
                continue
            hl = pdb_to_hl2[ch.id]
            seq_i = 0
            for res in ch.get_residues():
                if not is_aa(res, standard=True):
                    continue
                aa = aa1(res.get_resname())
                pi = FAUCHERE_PI.get(aa, 0.0)
                is_cdr = (hl, seq_i) in cdr_set
                for atom in res.get_atoms():
                    elem = atom.element.strip().upper()
                    if elem in ("H", ""):
                        continue
                    meta.append(
                        {
                            "coord": atom.coord.astype(float).copy(),
                            "element": elem,
                            "pi": float(pi),
                            "chain": hl,
                            "seq_i": seq_i,
                            "is_cdr": is_cdr,
                            "r_vdw": VDW.get(elem, 1.7),
                        }
                    )
                seq_i += 1
        if not meta:
            return None, "no_atoms"
        fs = freesasa.Structure(tmp_path)
        result = freesasa.calc(
            fs, freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": PROBE})
        )
        if fs.nAtoms() != len(meta):
            return None, f"freesasa_atom_mismatch:{fs.nAtoms()}!={len(meta)}"
        for i, m in enumerate(meta):
            m["sasa"] = float(result.atomArea(i))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    atom_coords = np.array([m["coord"] for m in meta], float)
    atom_radii = np.array([m["r_vdw"] + PROBE for m in meta], float)
    atom_pi = np.array([m["pi"] for m in meta], float)

    points, areas, cdr_flags = [], [], []
    for m in meta:
        if m["sasa"] < MIN_SASA:
            continue
        n = max(1, int(round(m["sasa"] * DENSITY)))
        dirs = fibonacci_sphere(n)
        R = m["r_vdw"] + PROBE
        cand = m["coord"][None, :] + dirs * R
        for p in cand:
            d = np.linalg.norm(atom_coords - p, axis=1)
            if np.all(d >= (atom_radii - 0.05 - 1e-6)):
                points.append(p)
                areas.append(m["sasa"] / n)
                cdr_flags.append(m["is_cdr"])
    if not points:
        return None, "no_surface_points"
    pts = np.asarray(points, float)
    areas_a = np.asarray(areas, float)
    cdr_a = np.asarray(cdr_flags, bool)
    if len(pts) > MAX_POINTS:
        idx = np.linspace(0, len(pts) - 1, MAX_POINTS).astype(int)
        pts, areas_a, cdr_a = pts[idx], areas_a[idx], cdr_a[idx]

    # Vectorized neighborhood field with chunking
    H = np.zeros(len(pts), float)
    chunk = 256
    cutoff2 = CUTOFF * CUTOFF
    for i0 in range(0, len(pts), chunk):
        block = pts[i0 : i0 + chunk]
        # (n_block, n_atoms)
        d2 = np.sum((block[:, None, :] - atom_coords[None, :, :]) ** 2, axis=2)
        mask = d2 <= cutoff2
        d = np.sqrt(np.where(mask, d2, 0.0))
        contrib = np.where(mask, atom_pi[None, :] * np.exp(-ALPHA * d), 0.0)
        H[i0 : i0 + chunk] = contrib.sum(axis=1)

    return {
        "points": pts,
        "areas": areas_a,
        "cdr": cdr_a,
        "H": H,
        "n_atoms": len(meta),
        "n_points": len(pts),
    }, None


def hydro_summaries(surf: dict) -> dict:
    H = surf["H"]
    areas = surf["areas"]
    cdr = surf["cdr"]
    w = areas / areas.sum() if areas.sum() > 0 else np.ones_like(areas) / len(areas)
    q75, q90, q95 = np.quantile(H, [0.75, 0.90, 0.95])
    top10_n = max(1, int(math.ceil(0.10 * len(H))))
    top10_idx = np.argsort(-H)[:top10_n]
    high = H >= np.quantile(H, 0.80)
    comps = connected_components(high, surf["points"], LINK)
    sizes = [len(c) for c in comps] if comps else [0]
    largest = comps[int(np.argmax(sizes))] if comps else []
    total_area = float(areas.sum())
    largest_area = float(areas[largest].sum()) if largest else 0.0
    cdr_H = H[cdr] if cdr.any() else np.array([])
    return {
        "mean_H_surface": float(np.average(H, weights=w)),
        "q75_H_surface": float(q75),
        "q90_H_surface": float(q90),
        "q95_H_surface": float(q95),
        "max_H_surface": float(H.max()),
        "positive_H_area_fraction": float(areas[H > 0].sum() / total_area) if total_area else 0.0,
        "top10_H_mean": float(H[top10_idx].mean()),
        "top10_H_area_fraction": float(areas[top10_idx].sum() / total_area) if total_area else 0.0,
        "high_H_patch_count": int(len(comps)),
        "largest_high_H_patch_area_fraction": largest_area / total_area if total_area else 0.0,
        "largest_high_H_patch_n_vertices": int(len(largest)),
        "CDR_mean_H": float(cdr_H.mean()) if len(cdr_H) else 0.0,
        "CDR_q90_H": float(np.quantile(cdr_H, 0.90)) if len(cdr_H) else 0.0,
        "CDR_high_H_area_fraction": (
            float(areas[cdr & high].sum() / areas[cdr].sum()) if cdr.any() and areas[cdr].sum() > 0 else 0.0
        ),
        "n_surface_points": int(surf["n_points"]),
    }


def copatch_summaries(surf: dict, phi: np.ndarray) -> dict:
    H = surf["H"]
    areas = surf["areas"]
    cdr = surf["cdr"]
    pts = surf["points"]
    total = float(areas.sum()) or 1.0
    high_H = H >= np.quantile(H, 0.80)
    high_abs = np.abs(phi) >= np.quantile(np.abs(phi), 0.80)
    pos = phi > 0
    neg = phi < 0

    def frac(mask):
        return float(areas[mask].sum() / total)

    def largest_frac(mask):
        comps = connected_components(mask, pts, LINK)
        if not comps:
            return 0.0
        sizes = [float(areas[c].sum()) for c in comps]
        return float(max(sizes) / total)

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "corr_H_absPhi": safe_corr(H, np.abs(phi)),
        "corr_H_phi": safe_corr(H, phi),
        "area_frac_highH_highAbsPhi": frac(high_H & high_abs),
        "area_frac_highH_positivePhi": frac(high_H & pos),
        "area_frac_highH_negativePhi": frac(high_H & neg),
        "largest_highH_highAbsPhi_patch_area_fraction": largest_frac(high_H & high_abs),
        "largest_highH_positivePhi_patch_area_fraction": largest_frac(high_H & pos),
        "largest_highH_negativePhi_patch_area_fraction": largest_frac(high_H & neg),
        "CDR_area_frac_highH_highAbsPhi": (
            float(areas[cdr & high_H & high_abs].sum() / areas[cdr].sum())
            if cdr.any() and areas[cdr].sum() > 0
            else 0.0
        ),
    }
