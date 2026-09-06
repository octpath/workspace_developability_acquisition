#!/usr/bin/env python3
"""S2: Cross-generator structural disagreement (ESMFold / ABB2 / Boltz2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SM = FP / "structure_marathon/generator_disagreement"
SM.mkdir(parents=True, exist_ok=True)
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
CDR = FP / "cdr_sequence_index_imgt.csv"
ARO = {"PHE", "TYR", "TRP"}


def ca_list(pdb: Path, chain_map=None):
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    atoms = []
    meta = []
    for ch in st[0]:
        cid = ch.id
        for res in ch:
            if res.id[0] != " " or not is_aa(res, standard=True):
                continue
            if "CA" not in res:
                continue
            atoms.append(res["CA"])
            meta.append((cid, int(res.id[1]), res.get_resname()))
    return atoms, meta


def rmsd_super(a_atoms, b_atoms):
    n = min(len(a_atoms), len(b_atoms))
    if n < 5:
        return np.nan
    fixed = a_atoms[:n]
    mov = b_atoms[:n]
    # copy coords
    from Bio.PDB import Atom

    # Superimposer needs Atom objects with coords
    try:
        sup = Superimposer()
        sup.set_atoms(fixed, mov)
        return float(sup.rms)
    except Exception:
        A = np.asarray([a.coord for a in fixed], float)
        B = np.asarray([a.coord for a in mov], float)
        A = A - A.mean(0)
        B = B - B.mean(0)
        return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def chain_split(atoms, meta, prefer=("H", "A")):
    # first chain = heavy-like
    chains = []
    for c, _, _ in meta:
        if c not in chains:
            chains.append(c)
    if not chains:
        return [], [], [], []
    h = chains[0]
    l = chains[1] if len(chains) > 1 else chains[0]
    ha = [a for a, m in zip(atoms, meta) if m[0] == h]
    la = [a for a, m in zip(atoms, meta) if m[0] == l]
    hm = [m for m in meta if m[0] == h]
    lm = [m for m in meta if m[0] == l]
    return ha, la, hm, lm


def sasa_proxy(atoms, meta):
    coords = np.asarray([a.coord for a in atoms], float)
    sasa = []
    for i in range(len(coords)):
        d = np.linalg.norm(coords - coords[i], axis=1)
        nn = ((d > 0.1) & (d < 10)).sum()
        sasa.append(max(0.0, 1.0 - nn / 25.0))
    return np.asarray(sasa, float)


def aromatic_exposed(meta, sasa, thr=0.25):
    return float(sum(s for m, s in zip(meta, sasa) if m[2] in ARO and s >= thr))


def main():
    cw = pd.read_csv(CW)
    rows = []
    for _, r in cw.iterrows():
        ab = str(r.id)
        paths = {
            "esmfold": Path(str(r.esmfold_canonical_path)) if pd.notna(r.esmfold_canonical_path) else None,
            "abb2": Path(str(r.abodybuilder2_path)) if pd.notna(r.abodybuilder2_path) else None,
            "boltz2": Path(str(r.boltz2_pdb_path)) if pd.notna(r.boltz2_pdb_path) else None,
        }
        loaded = {}
        for g, p in paths.items():
            if p and p.exists():
                try:
                    atoms, meta = ca_list(p)
                    loaded[g] = (atoms, meta)
                except Exception:
                    pass
        if len(loaded) < 2:
            continue
        gens = list(loaded.keys())
        feat = {"id": ab, "n_generators": len(gens)}
        # pairwise RMSDs
        vh_rms, vl_rms, fv_rms = [], [], []
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                a, ma = loaded[gens[i]]
                b, mb = loaded[gens[j]]
                ha, la, hma, lma = chain_split(a, ma)
                hb, lb, hmb, lmb = chain_split(b, mb)
                rvh = rmsd_super(ha, hb)
                rvl = rmsd_super(la, lb)
                rfv = rmsd_super(a, b)
                vh_rms.append(rvh)
                vl_rms.append(rvl)
                fv_rms.append(rfv)
        for name, arr in [("VH", vh_rms), ("VL", vl_rms), ("Fv", fv_rms)]:
            a = np.asarray(arr, float)
            a = a[np.isfinite(a)]
            feat[f"S2_{name}_RMSD_median"] = float(np.median(a)) if len(a) else np.nan
            feat[f"S2_{name}_RMSD_max"] = float(np.max(a)) if len(a) else np.nan
            feat[f"S2_{name}_RMSD_sd"] = float(np.std(a)) if len(a) else np.nan
        # exposure / aromatic disagreement
        aro = []
        sasa_tot = []
        for g, (atoms, meta) in loaded.items():
            s = sasa_proxy(atoms, meta)
            aro.append(aromatic_exposed(meta, s))
            sasa_tot.append(float(s.sum()))
        aro = np.asarray(aro, float)
        sasa_tot = np.asarray(sasa_tot, float)
        feat["S2_aro_exposed_sd"] = float(np.std(aro))
        feat["S2_aro_exposed_maxdiff"] = float(np.max(aro) - np.min(aro)) if len(aro) else np.nan
        feat["S2_sasa_proxy_sd"] = float(np.std(sasa_tot))
        feat["S2_sasa_proxy_maxdiff"] = float(np.max(sasa_tot) - np.min(sasa_tot)) if len(sasa_tot) else np.nan
        rows.append(feat)
    df = pd.DataFrame(rows)
    df.to_csv(SM / "S2_DISAGREEMENT_FEATURES.csv", index=False)
    (SM / "S2_META.json").write_text(json.dumps({"n": len(df), "dim": len(df.columns) - 1}, indent=2))
    print("DONE S2", len(df), flush=True)


if __name__ == "__main__":
    main()
