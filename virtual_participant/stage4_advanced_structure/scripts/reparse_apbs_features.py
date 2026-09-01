#!/usr/bin/env python3
"""Recompute ADV_ELECTROSTATICS from existing APBS DX outputs; merge into feature table."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/stage4_advanced_structure"
APBS_DIR = OUT / "cache/apbs"
ESMFOLD = ROOT / "esmfold_native"
DEV = ROOT / "competition/data/distribution/dev.csv"
FEAT = OUT / "cache/features"

PROBE = 1.4
RASA_EXPOSED = 0.20
PATCH_R = 8.0
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def aa1(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_dx(path: Path):
    lines = path.read_text().splitlines()
    dims = origin = None
    deltas = []
    data = []
    reading = False
    for line in lines:
        if line.startswith("object 1 class gridpositions"):
            parts = line.split()
            dims = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
        elif line.startswith("origin"):
            origin = np.array(list(map(float, line.split()[1:4])))
        elif line.startswith("delta"):
            deltas.append(list(map(float, line.split()[1:4])))
        elif "data follows" in line:
            reading = True
            continue
        elif reading:
            if line.startswith("attribute") or line.startswith("object") or not line.strip():
                if line.startswith("attribute") or line.startswith("object"):
                    reading = False
                continue
            for tok in line.split():
                try:
                    data.append(float(tok))
                except ValueError:
                    reading = False
                    break
    if dims is None or origin is None or len(deltas) < 3:
        raise ValueError(f"bad dx header {path}")
    arr = np.asarray(data, float)
    expected = math.prod(dims)
    if arr.size < expected:
        raise ValueError(f"short data {arr.size}<{expected}")
    arr = arr[:expected].reshape(dims, order="F")  # APBS often Fortran order
    return origin, np.array(deltas), arr


def sample_vals(origin, dmat, grid, points):
    nx, ny, nz = grid.shape
    dx = dmat[0, 0]
    dy = dmat[1, 1]
    dz = dmat[2, 2]
    vals = []
    for p in points:
        i = (p[0] - origin[0]) / dx
        j = (p[1] - origin[1]) / dy
        k = (p[2] - origin[2]) / dz
        i0, j0, k0 = int(np.floor(i)), int(np.floor(j)), int(np.floor(k))
        if 0 <= i0 < nx and 0 <= j0 < ny and 0 <= k0 < nz:
            vals.append(grid[i0, j0, k0])
    return np.asarray(vals, float)


def residue_rows(pdb):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(pdb))
    sr = ShrakeRupley(probe_radius=PROBE, n_points=60)
    sr.compute(st, level="R")
    rows = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                a = aa1(res.get_resname())
                sasa = float(res.sasa)
                maxasa = MAX_ASA.get(a)
                rasa = sasa / maxasa if maxasa else np.nan
                ca = res["CA"].coord.copy() if "CA" in res else None
                rows.append({"chain": chain.id, "aa": a, "sasa": sasa, "rasa": rasa, "ca": ca,
                             "exposed": bool(np.isfinite(rasa) and rasa >= RASA_EXPOSED)})
        break
    return rows


def apbs_feat(aid):
    feat = {"antibody_id": aid, "apbs_ok": 0}
    cands = list(APBS_DIR.glob(f"{aid}_pot*.dx")) + list(APBS_DIR.glob(f"{aid}*.dx"))
    if not cands:
        feat["apbs_error"] = "missing_dx"
        return feat
    try:
        origin, dmat, grid = load_dx(cands[0])
        rows = residue_rows(ESMFOLD / f"{aid}.pdb")
        exposed = [r for r in rows if r["exposed"] and r["ca"] is not None]
        pts = [r["ca"] + np.array([1.4, 0.0, 0.0]) for r in exposed]
        vals = sample_vals(origin, dmat, grid, pts)
        if len(vals) == 0:
            feat["apbs_error"] = "no_samples"
            return feat
        feat["apbs_ok"] = 1
        feat["apbs_mean_potential"] = float(vals.mean())
        feat["apbs_median_potential"] = float(np.median(vals))
        feat["apbs_std_potential"] = float(vals.std())
        feat["apbs_q10_potential"] = float(np.quantile(vals, 0.1))
        feat["apbs_q90_potential"] = float(np.quantile(vals, 0.9))
        feat["apbs_abs_mean_potential"] = float(np.mean(np.abs(vals)))
        feat["apbs_pos_frac"] = float((vals > 0).mean())
        feat["apbs_neg_frac"] = float((vals < 0).mean())
        feat["apbs_strong_pos_frac"] = float((vals > 1.0).mean())  # kT/e scale from APBS output
        feat["apbs_strong_neg_frac"] = float((vals < -1.0).mean())
        # patches by potential sign at exposed CA
        if len(vals) == len(exposed):
            for name, pred in [("pos", lambda v: v > 0.5), ("neg", lambda v: v < -0.5)]:
                idxs = [i for i, v in enumerate(vals) if pred(v)]
                if not idxs:
                    feat[f"apbs_{name}_patch_count"] = 0
                    feat[f"apbs_{name}_max_patch"] = 0
                    continue
                parent = {i: i for i in idxs}

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for a in idxs:
                    for b in idxs:
                        if a >= b:
                            continue
                        if np.linalg.norm(exposed[a]["ca"] - exposed[b]["ca"]) <= PATCH_R:
                            ra, rb = find(a), find(b)
                            if ra != rb:
                                parent[rb] = ra
                from collections import defaultdict
                comps = defaultdict(list)
                for i in idxs:
                    comps[find(i)].append(i)
                sizes = [len(v) for v in comps.values()]
                feat[f"apbs_{name}_patch_count"] = len(sizes)
                feat[f"apbs_{name}_max_patch"] = int(max(sizes))
    except Exception as e:
        feat["apbs_error"] = f"{type(e).__name__}:{e}"
    return feat


def main():
    # ensure all dx exist; run missing APBS if needed
    import subprocess
    pqr_dir = OUT / "cache/pqr"
    # import write helper from extract script lightly
    from extract_stage4_features import write_apbs_in, APBS_BIN

    dev = pd.read_csv(DEV)
    for aid in dev["id"]:
        dxs = list(APBS_DIR.glob(f"{aid}_pot*.dx"))
        if dxs:
            continue
        pqr = pqr_dir / f"{aid}.pqr"
        if not pqr.exists():
            continue
        prefix = APBS_DIR / aid
        inp = write_apbs_in(pqr, prefix)
        subprocess.run([str(APBS_BIN), str(inp)], cwd=str(APBS_DIR), capture_output=True, text=True, timeout=300)

    rows = []
    for i, aid in enumerate(dev["id"]):
        rows.append(apbs_feat(aid))
        if (i + 1) % 20 == 0:
            print(f"apbs reparse {i+1}/162", flush=True)
    ap = pd.DataFrame(rows)
    ap.to_csv(FEAT / "apbs_features.csv", index=False)
    print("apbs_ok", ap.apbs_ok.value_counts().to_dict())
    print("errors", ap.loc[ap.apbs_ok == 0, "apbs_error"].value_counts().head())

    # merge into advanced features when ready
    adv_path = FEAT / "advanced_features.csv"
    if not adv_path.exists():
        adv_path = FEAT / "advanced_features_partial.csv"
    adv = pd.read_csv(adv_path)
    drop_cols = [c for c in adv.columns if c.startswith("apbs")]
    adv = adv.drop(columns=drop_cols, errors="ignore")
    merged = adv.merge(ap, on="antibody_id", how="left")
    merged.to_csv(FEAT / "advanced_features_with_apbs.csv", index=False)
    print("wrote advanced_features_with_apbs", merged.shape)


if __name__ == "__main__":
    main()
