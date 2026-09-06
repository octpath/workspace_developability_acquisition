#!/usr/bin/env python3
"""S3 surface-patch graph + S4 contact/packing graph (ESMFold Fv)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SM = FP / "structure_marathon"
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
CDR = FP / "cdr_sequence_index_imgt.csv"
CONTACT = 8.0
PATCH_EDGE = 6.0
EXPOSED = 0.25
HYDRO = {"ALA", "ILE", "LEU", "MET", "PHE", "VAL", "TRP", "TYR"}
ARO = {"PHE", "TYR", "TRP"}
CHG = {"ARG", "LYS", "ASP", "GLU", "HIS"}


def load_res(pdb: Path):
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    out = []
    for ch in st[0]:
        for res in ch:
            if res.id[0] != " " or not is_aa(res, standard=True):
                continue
            if "CA" not in res:
                continue
            out.append(
                {
                    "chain": ch.id,
                    "resseq": int(res.id[1]),
                    "resname": res.get_resname(),
                    "ca": res["CA"].coord.copy(),
                    "cb": (res["CB"].coord.copy() if "CB" in res else res["CA"].coord.copy()),
                }
            )
    return out


def rasa_proxy(res):
    c = np.asarray([r["ca"] for r in res], float)
    out = np.zeros(len(res))
    for i in range(len(res)):
        d = np.linalg.norm(c - c[i], axis=1)
        out[i] = float(np.clip(1.0 - ((d > 0.1) & (d < 10)).sum() / 25.0, 0, 1))
    return out


def largest_component(indices, coords, edge):
    if len(indices) == 0:
        return []
    adj = {i: [] for i in indices}
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            if np.linalg.norm(coords[i] - coords[j]) <= edge:
                adj[i].append(j)
                adj[j].append(i)
    seen, best = set(), []
    for s in indices:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def graph_stats(indices, coords, edge):
    n = len(indices)
    if n == 0:
        return {k: np.nan for k in ["n", "n_edges", "density", "mean_deg", "max_deg", "clustering"]}
    idx = list(indices)
    pos = {i: k for k, i in enumerate(idx)}
    A = np.zeros((n, n), int)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = idx[a], idx[b]
            if np.linalg.norm(coords[i] - coords[j]) <= edge:
                A[a, b] = A[b, a] = 1
    deg = A.sum(axis=1).astype(float)
    n_edges = int(A.sum() // 2)
    dens = n_edges / (n * (n - 1) / 2) if n > 1 else 0.0
    # local clustering
    cl = []
    for i in range(n):
        nbr = np.where(A[i] > 0)[0]
        if len(nbr) < 2:
            cl.append(0.0)
            continue
        sub = A[np.ix_(nbr, nbr)]
        possible = len(nbr) * (len(nbr) - 1) / 2
        cl.append(float(sub.sum() / 2 / possible) if possible else 0.0)
    return {
        "n": n,
        "n_edges": n_edges,
        "density": dens,
        "mean_deg": float(deg.mean()),
        "max_deg": float(deg.max()) if n else np.nan,
        "clustering": float(np.mean(cl)),
    }


def main():
    cw = pd.read_csv(CW)
    s3, s4 = [], []
    for _, r in cw.iterrows():
        ab = str(r.id)
        pdb = Path(str(r.esmfold_canonical_path))
        if not pdb.exists():
            continue
        res = load_res(pdb)
        rasa = rasa_proxy(res)
        coords = np.asarray([x["ca"] for x in res], float)
        chains = sorted({x["chain"] for x in res})
        heavy = chains[0]
        # S3 exposed nodes
        exposed = [i for i, s in enumerate(rasa) if s >= EXPOSED]
        hydro = [i for i in exposed if res[i]["resname"] in HYDRO]
        aro = [i for i in exposed if res[i]["resname"] in ARO]
        chg = [i for i in exposed if res[i]["resname"] in CHG]
        feat3 = {"id": ab}
        for tag, nodes, edge in [
            ("hydro", hydro, PATCH_EDGE),
            ("aromatic", aro, PATCH_EDGE),
            ("charged", chg, PATCH_EDGE),
            ("exposed", exposed, PATCH_EDGE),
        ]:
            st = graph_stats(nodes, coords, edge)
            best = largest_component(nodes, coords, edge)
            for k, v in st.items():
                feat3[f"S3_{tag}_{k}"] = v
            feat3[f"S3_{tag}_largest_n"] = len(best)
            feat3[f"S3_{tag}_largest_sasa"] = float(rasa[best].sum()) if best else np.nan
            if tag == "hydro" and best:
                feat3["S3_aro_in_largest_hydro"] = float(sum(1 for i in best if res[i]["resname"] in ARO))
                feat3["S3_largest_hydro_heavy_frac"] = float(np.mean([res[i]["chain"] == heavy for i in best]))
        s3.append(feat3)

        # S4 contact graph whole / VH / VL / interface
        feat4 = {"id": ab}
        vh = [i for i, x in enumerate(res) if x["chain"] == heavy]
        vl = [i for i, x in enumerate(res) if x["chain"] != heavy]
        for tag, nodes in [("Fv", list(range(len(res)))), ("VH", vh), ("VL", vl)]:
            st = graph_stats(nodes, coords, CONTACT)
            for k, v in st.items():
                feat4[f"S4_{tag}_{k}"] = v
        # interface contacts count
        n_if = 0
        for i in vh:
            for j in vl:
                if np.linalg.norm(coords[i] - coords[j]) <= 4.5:
                    n_if += 1
        feat4["S4_VH_VL_interface_contacts"] = n_if
        # contact chemistry fractions on Fv edges
        hydro_e = aro_e = polar_e = charged_e = tot_e = 0
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                if np.linalg.norm(coords[i] - coords[j]) <= CONTACT:
                    tot_e += 1
                    ri, rj = res[i]["resname"], res[j]["resname"]
                    if ri in HYDRO and rj in HYDRO:
                        hydro_e += 1
                    if ri in ARO or rj in ARO:
                        aro_e += 1
                    if ri in CHG and rj in CHG:
                        charged_e += 1
        feat4["S4_hydro_contact_frac"] = hydro_e / tot_e if tot_e else np.nan
        feat4["S4_aromatic_contact_frac"] = aro_e / tot_e if tot_e else np.nan
        feat4["S4_charged_contact_frac"] = charged_e / tot_e if tot_e else np.nan
        s4.append(feat4)

    pd.DataFrame(s3).to_csv(SM / "surface_patch_graph/S3_FEATURES.csv", index=False)
    pd.DataFrame(s4).to_csv(SM / "contact_graph/S4_FEATURES.csv", index=False)
    print("DONE S3", len(s3), "S4", len(s4), flush=True)


if __name__ == "__main__":
    main()
