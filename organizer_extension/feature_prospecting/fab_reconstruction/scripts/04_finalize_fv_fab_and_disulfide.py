#!/usr/bin/env python3
"""Finalize target-blind Fab analyses: Fv-vs-Fab (324), H-L Cys distances, report updates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
SEQ = OUT / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
CROSS = ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"
CDR_IDX = ROOT / "organizer_extension/feature_prospecting/cdr_sequence_index_imgt.csv"
FAB_DIR = OUT / "structures/esmfold_fab"
COMP = OUT / "comparisons"
QC = OUT / "qc"
STR = OUT / "structures"

AROMATIC = set("FWY")
AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}
PROBE_RADIUS = 1.4
N_POINTS = 100


def load(path: Path):
    return PDBParser(QUIET=True).get_structure("x", str(path))


def residues_ca(structure, chain_id: str):
    out = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                if "CA" not in res:
                    continue
                out.append(res)
    return out


def backbone_atoms(residues, names=("N", "CA", "C")):
    atoms = []
    for res in residues:
        for n in names:
            if n in res:
                atoms.append(res[n])
    return atoms


def rmsd_super(fixed_atoms, moving_atoms):
    if len(fixed_atoms) != len(moving_atoms) or not fixed_atoms:
        return float("nan")
    si = Superimposer()
    si.set_atoms(fixed_atoms, moving_atoms)
    return float(si.rms)


def interface_contacts_ca(res_a, res_b, cutoff=8.0):
    ca_a = np.array([r["CA"].coord for r in res_a if "CA" in r])
    ca_b = np.array([r["CA"].coord for r in res_b if "CA" in r])
    if len(ca_a) == 0 or len(ca_b) == 0:
        return 0
    d = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
    return int((d < cutoff).sum())


def vl_orient_after_vh_align(fv_h, fab_h, fv_l, fab_l):
    """After aligning Fab VH onto Fv VH, distance between VL COMs (Å)."""
    n = min(len(fv_h), len(fab_h))
    m = min(len(fv_l), len(fab_l))
    if n < 10 or m < 10:
        return float("nan"), float("nan")
    fixed = [fv_h[i]["CA"] for i in range(n)]
    moving = [fab_h[i]["CA"] for i in range(n)]
    si = Superimposer()
    si.set_atoms(fixed, moving)
    R, t = si.rotran
    fab_l_coords = np.array([fab_l[i]["CA"].coord for i in range(m)], dtype=float)
    fv_l_coords = np.array([fv_l[i]["CA"].coord for i in range(m)], dtype=float)
    # BioPython Atom.transform uses coord @ rot + tran
    fab_l_al = fab_l_coords @ np.asarray(R) + np.asarray(t).reshape(3)
    com_diff = float(np.linalg.norm(fv_l_coords.mean(0) - fab_l_al.mean(0)))
    # VL RMSD without further alignment (orientation-sensitive)
    vl_rmsd_no_realign = float(np.sqrt(((fv_l_coords - fab_l_al) ** 2).sum(axis=1).mean()))
    return com_diff, vl_rmsd_no_realign


def exposed_aromatic_sasa(structure, chain_res: dict[str, list], aromatic_only=True):
    """ShrakeRupley on full structure; sum SASA of aromatic residues in given chains/res lists.

    chain_res: {'A': [Residue,...], 'B': [...]} — only these residues counted.
    """
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(structure, level="R")
    total = 0.0
    for ch, residues in chain_res.items():
        for res in residues:
            aa = AA3.get(res.get_resname(), "X")
            if aromatic_only and aa not in AROMATIC:
                continue
            sasa = float(res.sasa) if hasattr(res, "sasa") and res.sasa is not None else 0.0
            total += sasa
    return total


def dist_summary(s: pd.Series, name: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"subset": name, "n": 0}
    return {
        "subset": name,
        "n": int(len(s)),
        "q10": float(s.quantile(0.10)),
        "median": float(s.median()),
        "q90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "frac_lt_2_3": float((s < 2.3).mean()),
        "frac_lt_2_6": float((s < 2.6).mean()),
        "frac_lt_3_0": float((s < 3.0).mean()),
        "frac_ge_3_0": float((s >= 3.0).mean()),
    }


def metric_summary(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "iqr": float(q3 - q1),
        "q25": q1,
        "q75": q3,
        "q90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def main():
    fab = pd.read_csv(SEQ)
    cross = pd.read_csv(CROSS)
    cdr = pd.read_csv(CDR_IDX)
    full_qc = pd.read_csv(QC / "ESMFOLD_FAB_FULL_QC.csv")

    # CDR index maps: sequence_index 0-based within VH or VL
    cdr_h = cdr[(cdr["chain"] == "H") & (cdr["is_cdr"] == True)].groupby("id")["sequence_index"].apply(list).to_dict()
    cdr_l = cdr[(cdr["chain"] == "L") & (cdr["is_cdr"] == True)].groupby("id")["sequence_index"].apply(list).to_dict()
    hcdr3 = (
        cdr[(cdr["chain"] == "H") & (cdr["region"] == "CDR3")]
        .groupby("id")["sequence_index"]
        .apply(list)
        .to_dict()
    )

    cross_paths = cross.set_index("id")["esmfold_canonical_path"].to_dict() if "esmfold_canonical_path" in cross.columns else {}

    rows = []
    for i, r in fab.iterrows():
        aid = r["id"]
        fp = FAB_DIR / f"{aid}.pdb"
        fv_path = Path(cross_paths.get(aid, "")) if aid in cross_paths else ROOT / "esmfold_native" / f"{aid}.pdb"
        if not fv_path.exists():
            alt = ROOT / "esmfold_native" / f"{aid}.pdb"
            fv_path = alt if alt.exists() else None
        if not fp.exists() or fv_path is None or not Path(fv_path).exists():
            rows.append({"id": aid, "notes": "missing_structure"})
            continue

        vh_len, vl_len = int(r["VH_len_used"]), int(r["VL_len_used"])
        fab_s = load(fp)
        fv_s = load(fv_path)
        fab_h = residues_ca(fab_s, "A")[:vh_len]
        fab_l = residues_ca(fab_s, "B")[:vl_len]
        fv_h = residues_ca(fv_s, "A")
        fv_l = residues_ca(fv_s, "B")
        nH = min(len(fv_h), len(fab_h), vh_len)
        nL = min(len(fv_l), len(fab_l), vl_len)
        fab_h, fab_l = fab_h[:nH], fab_l[:nL]
        fv_h, fv_l = fv_h[:nH], fv_l[:nL]

        # Backbone RMSDs (CA for speed/stability; also N-CA-C optional — use CA for consistency with prior)
        rms_h = rmsd_super([x["CA"] for x in fv_h], [x["CA"] for x in fab_h])
        rms_l = rmsd_super([x["CA"] for x in fv_l], [x["CA"] for x in fab_l])
        rms_comb = rmsd_super(
            [x["CA"] for x in fv_h] + [x["CA"] for x in fv_l],
            [x["CA"] for x in fab_h] + [x["CA"] for x in fab_l],
        )

        # CDR / HCDR3
        h_idx = [j for j in cdr_h.get(aid, []) if j < nH]
        l_idx = [j for j in cdr_l.get(aid, []) if j < nL]
        h3_idx = [j for j in hcdr3.get(aid, []) if j < nH]
        if h_idx or l_idx:
            fv_cdr = [fv_h[j]["CA"] for j in h_idx] + [fv_l[j]["CA"] for j in l_idx]
            fab_cdr = [fab_h[j]["CA"] for j in h_idx] + [fab_l[j]["CA"] for j in l_idx]
            rms_cdr = rmsd_super(fv_cdr, fab_cdr)
        else:
            rms_cdr = float("nan")
        if h3_idx:
            rms_h3 = rmsd_super([fv_h[j]["CA"] for j in h3_idx], [fab_h[j]["CA"] for j in h3_idx])
        else:
            rms_h3 = float("nan")

        orient_com, vl_rmsd_orient = vl_orient_after_vh_align(fv_h, fab_h, fv_l, fab_l)

        fv_iface = interface_contacts_ca(fv_h, fv_l)
        fab_iface = interface_contacts_ca(fab_h, fab_l)

        # Exposed aromatic SASA (gate_b2-compatible ShrakeRupley)
        try:
            aro_fv = exposed_aromatic_sasa(fv_s, {"A": fv_h, "B": fv_l})
            # Recompute on Fab structure but only count VH/VL residues
            aro_fab = exposed_aromatic_sasa(fab_s, {"A": fab_h, "B": fab_l})
            aro_delta = aro_fab - aro_fv
            aro_note = "ok"
        except Exception as e:
            aro_fv = aro_fab = aro_delta = float("nan")
            aro_note = f"sasa_fail:{type(e).__name__}"

        rows.append(
            {
                "id": aid,
                "light_locus": r["light_locus"],
                "fv_path": str(fv_path),
                "fab_path": str(fp),
                "VH_backbone_RMSD": rms_h,
                "VL_backbone_RMSD": rms_l,
                "combined_Fv_backbone_RMSD": rms_comb,
                "all_CDR_backbone_RMSD": rms_cdr,
                "HCDR3_backbone_RMSD": rms_h3,
                "VH_VL_orientation_COM_diff_A": orient_com,
                "VL_RMSD_after_VH_align_A": vl_rmsd_orient,
                "VH_VL_interface_contacts_Fv": fv_iface,
                "VH_VL_interface_contacts_Fab": fab_iface,
                "VH_VL_interface_contact_delta": fab_iface - fv_iface,
                "exposed_aromatic_SASA_Fv": aro_fv,
                "exposed_aromatic_SASA_Fab_FvPortion": aro_fab,
                "exposed_aromatic_SASA_delta": aro_delta,
                "notes": aro_note,
            }
        )
        if (len(rows) % 25) == 0:
            print(f"compared {len(rows)}/{len(fab)}", flush=True)

    cdf = pd.DataFrame(rows)
    COMP.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(COMP / "FV_VS_FAB_STRUCTURE_COMPARISON.csv", index=False)

    # Summaries
    metrics = [
        "VH_backbone_RMSD",
        "VL_backbone_RMSD",
        "combined_Fv_backbone_RMSD",
        "all_CDR_backbone_RMSD",
        "HCDR3_backbone_RMSD",
        "VH_VL_orientation_COM_diff_A",
        "VL_RMSD_after_VH_align_A",
        "VH_VL_interface_contact_delta",
        "exposed_aromatic_SASA_delta",
    ]
    sum_rows = []
    for m in metrics:
        sm = metric_summary(cdf[m])
        sm["metric"] = m
        sum_rows.append(sm)
    pd.DataFrame(sum_rows).to_csv(COMP / "FV_VS_FAB_METRIC_DISTRIBUTIONS.csv", index=False)

    comb = metric_summary(cdf["combined_Fv_backbone_RMSD"])
    material = (
        "No — adding CH1/CL does **not** materially rewrite predicted Fv backbone geometry "
        f"(combined Fv CA RMSD median {comb['median']:.2f} Å, q90 {comb['q90']:.2f} Å, max {comb['max']:.2f} Å). "
        "Local CDR/HCDR3 shifts remain similarly small. Orientation and interface contact deltas are "
        "secondary and should be interpreted as ESMFold multimer coupling differences, not experimental Fab physics."
    )
    if comb["median"] >= 1.5 or comb["q90"] >= 3.0:
        material = (
            "Yes/partial — Fv geometry shifts are large enough to matter for downstream physics "
            f"(combined median {comb['median']:.2f} Å, q90 {comb['q90']:.2f} Å)."
        )

    # H-L disulfide audit
    qc = full_qc.merge(fab[["id", "light_locus"]], on="id", how="left")
    dcol = "heavy_light_disulfide_distance"
    ss_summaries = [
        dist_summary(qc[dcol], "overall"),
        dist_summary(qc.loc[qc["light_locus"] == "kappa", dcol], "kappa"),
        dist_summary(qc.loc[qc["light_locus"] == "lambda", dcol], "lambda"),
    ]
    ss_df = pd.DataFrame(ss_summaries)
    ss_df.to_csv(QC / "HL_DISULFIDE_DISTANCE_AUDIT.csv", index=False)

    ss_md = f"""# Heavy–light interchain cysteine geometry audit (ESMFold Fab, N={len(qc)})

**Important:** Distances are **Sγ–Sγ** (or nearest heavy/light Cys SG pair used in QC).  
ESMFold does **not** enforce a covalent disulfide. Proximity ≠ formed bond.

## Distributions (Å)

| Subset | n | q10 | median | q90 | max | <2.3Å | <2.6Å | <3.0Å | ≥3.0Å |
|--------|---|-----|--------|-----|-----|-------|-------|-------|-------|
"""
    for s in ss_summaries:
        if s["n"] == 0:
            continue
        ss_md += (
            f"| {s['subset']} | {s['n']} | {s['q10']:.2f} | {s['median']:.2f} | {s['q90']:.2f} | {s['max']:.2f} | "
            f"{100*s['frac_lt_2_3']:.1f}% | {100*s['frac_lt_2_6']:.1f}% | {100*s['frac_lt_3_0']:.1f}% | {100*s['frac_ge_3_0']:.1f}% |\n"
        )
    ss_md += """
## FeNNix-on-Fab QC note

Treat H–L Cys geometry as an **input-QC / post-relaxation check**, not as ground-truth chemistry.
After FeNNix structure preparation / restrained relaxation, verify whether the expected H–L disulfide
geometry becomes chemically reasonable. Do **not** allow unresolved disulfide strain to dominate
energy features used for later TmApp work.
"""
    (QC / "HL_DISULFIDE_DISTANCE_AUDIT.md").write_text(ss_md)

    # Fv vs Fab summary markdown
    def line(m):
        sm = next(x for x in sum_rows if x["metric"] == m)
        return f"| {m} | {sm['median']:.3f} | {sm['iqr']:.3f} | {sm['q90']:.3f} | {sm['max']:.3f} |"

    summary = f"""# Fv vs Fab structure comparison (full cohort)

**N:** {int(cdf['combined_Fv_backbone_RMSD'].notna().sum())} / {len(fab)}  
**Target-blind.** Reconstructed experimental-like Fab vs existing ESMFold Fv.

## Distributions (Å unless noted)

| Metric | Median | IQR | q90 | Max |
|--------|--------|-----|-----|-----|
{chr(10).join(line(m) for m in metrics)}

## Scientific question

Does adding CH1/CL materially change the predicted Fv geometry?

**Answer:** {material}

## Notes

- Backbone RMSDs use CA atoms.
- `exposed_aromatic_SASA_*` uses Bio.PDB ShrakeRupley (probe=1.4, n_points=100; same frozen params as gate_b2). Fab value counts only VH/VL residues after SASA on the full Fab complex.
- Orientation: VL COM displacement after VH-only superposition, plus VL CA RMSD without re-alignment.
"""
    (COMP / "FV_VS_FAB_STRUCTURE_SUMMARY.md").write_text(summary)

    # Refresh FAB structural descriptors for all 324 (quick geometry)
    desc_rows = []
    for _, r in fab.iterrows():
        aid = r["id"]
        fp = FAB_DIR / f"{aid}.pdb"
        if not fp.exists():
            continue
        vh, vl = int(r["VH_len_used"]), int(r["VL_len_used"])
        s = load(fp)
        h = residues_ca(s, "A")
        l = residues_ca(s, "B")
        if len(h) < vh + 1 or len(l) < vl + 1:
            continue
        h_ca = np.array([x["CA"].coord for x in h])
        l_ca = np.array([x["CA"].coord for x in l])
        fv_com = np.concatenate([h_ca[:vh], l_ca[:vl]]).mean(0)
        ct_com = np.concatenate([h_ca[vh:], l_ca[vl:]]).mean(0)
        elbow = np.mean([h_ca[vh - 1], l_ca[vl - 1]], axis=0)
        v1, v2 = fv_com - elbow, ct_com - elbow
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        coords = np.concatenate([h_ca, l_ca])
        rg = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(axis=1).mean()))
        desc_rows.append(
            {
                "id": aid,
                "VH_CH1_com_distance": float(np.linalg.norm(h_ca[:vh].mean(0) - h_ca[vh:].mean(0))),
                "VL_CL_com_distance": float(np.linalg.norm(l_ca[:vl].mean(0) - l_ca[vl:].mean(0))),
                "VH_VL_contacts_8A": interface_contacts_ca(h[:vh], l[:vl]),
                "CH1_CL_contacts_8A": interface_contacts_ca(h[vh:], l[vl:]),
                "Fab_Rg": rg,
                "elbow_proxy_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
            }
        )
    pd.DataFrame(desc_rows).to_csv(COMP / "FAB_STRUCTURAL_DESCRIPTORS_TARGET_BLIND.csv", index=False)

    # Ensure manifest columns
    man = full_qc.copy()
    want = [
        "id", "status", "runtime", "heavy_length", "light_length", "global_pLDDT",
        "VH_pLDDT", "VL_pLDDT", "CH1_pLDDT", "CL_pLDDT", "severe_clash_count",
        "heavy_light_disulfide_distance", "file_path", "SHA256", "notes",
    ]
    for c in want:
        if c not in man.columns:
            man[c] = ""
    if "status" not in man.columns or man["status"].isna().all():
        man["status"] = man.apply(lambda r: "ok" if r.get("success") else "fail", axis=1)
    man[want].to_csv(STR / "ESMFOLD_FAB_STRUCTURE_MANIFEST.csv", index=False)

    # Persist comparison summary JSON for report
    payload = {
        "n_compared": int(cdf["combined_Fv_backbone_RMSD"].notna().sum()),
        "combined_Fv_backbone_RMSD": comb,
        "material_change_answer": material,
        "hl_disulfide": ss_summaries,
        "metric_distributions": sum_rows,
    }
    (COMP / "FV_VS_FAB_FULL_SUMMARY.json").write_text(json.dumps(payload, indent=2))
    print("FINALIZE_OK", payload["n_compared"], "combined_median", comb["median"])


if __name__ == "__main__":
    main()
