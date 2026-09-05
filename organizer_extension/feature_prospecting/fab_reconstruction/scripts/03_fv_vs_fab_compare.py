#!/usr/bin/env python3
"""Target-blind Fv (existing ESMFold) vs Fab VH/VL comparison + Fab descriptors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
SEQ = OUT / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
CROSS = ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"
COMP = OUT / "comparisons"
AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def ca_atoms(structure, chain_id=None):
    atoms = []
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                if "CA" in res:
                    atoms.append(res["CA"])
    return atoms


def backbone_rmsd(atoms_a, atoms_b):
    if len(atoms_a) != len(atoms_b) or len(atoms_a) == 0:
        return float("nan")
    si = Superimposer()
    si.set_atoms(atoms_a, atoms_b)
    return float(si.rms)


def chain_ca_coords(structure, chain_id):
    coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if is_aa(res, standard=True) and "CA" in res:
                    coords.append(res["CA"].coord.copy())
    return np.array(coords)


def interface_contacts(ca_a, ca_b, cutoff=8.0):
    if len(ca_a) == 0 or len(ca_b) == 0:
        return 0
    d = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
    return int((d < cutoff).sum())


def rg(coords):
    if len(coords) == 0:
        return float("nan")
    c = coords - coords.mean(0)
    return float(np.sqrt((c * c).sum(axis=1).mean()))


def load(path):
    return PDBParser(QUIET=True).get_structure("x", str(path))


def main():
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    fab = pd.read_csv(SEQ)
    cross = pd.read_csv(CROSS)
    path_col = "esmfold_canonical_path" if "esmfold_canonical_path" in cross.columns else None
    if path_col is None:
        for c in cross.columns:
            if "esmfold" in c.lower() and "path" in c.lower():
                path_col = c
                break
    id_col = "antibody_id" if "antibody_id" in cross.columns else "id"

    if mode == "pilot":
        ids = json.loads((OUT / "structures/pilot_ids.json").read_text())
        fab_dir = OUT / "structures/pilot"
    else:
        ids = fab["id"].tolist()
        fab_dir = OUT / "structures/esmfold_fab"

    rows = []
    desc = []
    for aid in ids:
        fp = fab_dir / f"{aid}.pdb"
        if not fp.exists():
            continue
        row = fab[fab["id"] == aid].iloc[0]
        vh, vl = int(row["VH_len_used"]), int(row["VL_len_used"])
        cref = cross[cross[id_col] == aid]
        fv_path = None
        if len(cref) and path_col:
            fv_path = Path(str(cref.iloc[0][path_col]))
        if fv_path is None or not fv_path.exists():
            # fallback antibody-id path
            cand = ROOT / "esmfold_native" / f"{aid}.pdb"
            fv_path = cand if cand.exists() else None
        fab_s = load(fp)
        # Fab chains A/B
        h_ca = chain_ca_coords(fab_s, "A")
        l_ca = chain_ca_coords(fab_s, "B")
        if len(h_ca) < vh or len(l_ca) < vl:
            continue
        drow = {
            "id": aid,
            "VH_CH1_com_distance": float(np.linalg.norm(h_ca[:vh].mean(0) - h_ca[vh:].mean(0))) if len(h_ca) > vh else np.nan,
            "VL_CL_com_distance": float(np.linalg.norm(l_ca[:vl].mean(0) - l_ca[vl:].mean(0))) if len(l_ca) > vl else np.nan,
            "VH_VL_contacts_8A": interface_contacts(h_ca[:vh], l_ca[:vl]),
            "CH1_CL_contacts_8A": interface_contacts(h_ca[vh:], l_ca[vl:]),
            "Fab_Rg": rg(np.concatenate([h_ca, l_ca])),
            "elbow_proxy_angle_deg": np.nan,  # optional; left NaN unless defensible
        }
        # simple elbow proxy: angle between Fv COM→elbow and Const COM→elbow using junction CAs
        try:
            fv_com = np.concatenate([h_ca[:vh], l_ca[:vl]]).mean(0)
            ct_com = np.concatenate([h_ca[vh:], l_ca[vl:]]).mean(0)
            elbow = np.array([h_ca[vh - 1], l_ca[vl - 1]]).mean(0)
            v1, v2 = fv_com - elbow, ct_com - elbow
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            drow["elbow_proxy_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        except Exception:
            pass
        desc.append(drow)

        if fv_path is None or not Path(fv_path).exists():
            rows.append({"id": aid, "notes": "missing_fv", "combined_Fv_backbone_RMSD": np.nan})
            continue
        fv_s = load(fv_path)
        # Fv typically A=VH B=VL
        fv_h = ca_atoms(fv_s, "A")
        fv_l = ca_atoms(fv_s, "B")
        fab_h = ca_atoms(fab_s, "A")[:vh]
        fab_l = ca_atoms(fab_s, "B")[:vl]
        # length align
        nH = min(len(fv_h), len(fab_h))
        nL = min(len(fv_l), len(fab_l))
        rms_h = backbone_rmsd(fv_h[:nH], fab_h[:nH])
        rms_l = backbone_rmsd(fv_l[:nL], fab_l[:nL])
        # combined: superimpose jointly
        rms_comb = backbone_rmsd(fv_h[:nH] + fv_l[:nL], fab_h[:nH] + fab_l[:nL])
        # orientation: after aligning heavy, light COM distance change
        si = Superimposer()
        si.set_atoms(fv_h[:nH], fab_h[:nH])
        # apply to copies via transform on fab light relative — use coords
        R, t = si.rotran
        fv_l_c = np.array([a.coord for a in fv_l[:nL]])
        fab_l_c = np.array([a.coord for a in fab_l[:nL]])
        fab_l_al = (R @ fab_l_c.T).T + t
        orient = float(np.linalg.norm(fv_l_c.mean(0) - fab_l_al.mean(0)))
        rows.append(
            {
                "id": aid,
                "fv_path": str(fv_path),
                "fab_path": str(fp),
                "VH_backbone_RMSD": rms_h,
                "VL_backbone_RMSD": rms_l,
                "combined_Fv_backbone_RMSD": rms_comb,
                "VH_VL_orientation_diff_A": orient,
                "all_CDR_backbone_RMSD": np.nan,  # needs CDR index; left for later if CDR map available
                "HCDR3_backbone_RMSD": np.nan,
                "notes": "",
            }
        )

    COMP.mkdir(parents=True, exist_ok=True)
    cdf = pd.DataFrame(rows)
    ddf = pd.DataFrame(desc)
    cdf.to_csv(COMP / "FV_VS_FAB_STRUCTURE_COMPARISON.csv", index=False)
    ddf.to_csv(COMP / "FAB_STRUCTURAL_DESCRIPTORS_TARGET_BLIND.csv", index=False)
    ok = cdf["combined_Fv_backbone_RMSD"].dropna()
    summary = f"""# Fv vs Fab structure comparison ({mode})

N compared: {ok.shape[0]}

| Metric | Mean | Median | Max |
|--------|------|--------|-----|
| VH backbone RMSD (Å) | {cdf['VH_backbone_RMSD'].mean():.2f} | {cdf['VH_backbone_RMSD'].median():.2f} | {cdf['VH_backbone_RMSD'].max():.2f} |
| VL backbone RMSD (Å) | {cdf['VL_backbone_RMSD'].mean():.2f} | {cdf['VL_backbone_RMSD'].median():.2f} | {cdf['VL_backbone_RMSD'].max():.2f} |
| Combined Fv RMSD (Å) | {cdf['combined_Fv_backbone_RMSD'].mean():.2f} | {cdf['combined_Fv_backbone_RMSD'].median():.2f} | {cdf['combined_Fv_backbone_RMSD'].max():.2f} |
| VH/VL orient. diff (Å) | {cdf['VH_VL_orientation_diff_A'].mean():.2f} | {cdf['VH_VL_orientation_diff_A'].median():.2f} | {cdf['VH_VL_orientation_diff_A'].max():.2f} |

**Question:** Does adding CH1/CL materially alter predicted VH/VL conformation?

On this {mode} set, combined Fv RMSD mean ≈ {ok.mean():.2f} Å (target-blind).
"""
    (COMP / "FV_VS_FAB_STRUCTURE_SUMMARY.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
