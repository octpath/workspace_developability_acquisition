#!/usr/bin/env python3
"""Feature extraction + sample-count convergence for BioEmu constant-context arms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "bioemu_constant_context"
V2 = FP / "foundation_stability_v2"
FAB_STR = FP / "fab_reconstruction/structures/esmfold_fab"
SAMPLES = CTX / "cache/bioemu_samples"
ISO = CTX / "cache/isolated_v2_samples"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"
MAN = pd.read_csv(CTX / "BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv").set_index("id")


def load_traj(out_dir: Path, n_max: int | None = None):
    top = out_dir / "topology.pdb"
    xtc = list(out_dir.glob("*.xtc"))
    if top.exists() and xtc:
        traj = md.load(str(xtc[0]), top=str(top))
        for x in xtc[1:]:
            traj = traj.join(md.load(str(x), top=str(top)))
    else:
        pdbs = sorted(p for p in out_dir.glob("*.pdb") if p.name != "topology.pdb")
        if not pdbs:
            raise FileNotFoundError(out_dir)
        traj = md.load(str(pdbs[0]))
        for p in pdbs[1:]:
            traj = traj.join(md.load(str(p)))
    if n_max is not None:
        traj = traj[:n_max]
    return traj


def ca_slice(traj):
    return traj.atom_slice(traj.topology.select("name CA"))


def pairwise_ca_rmsd(ca, max_frames=24):
    n = ca.n_frames
    if n < 2:
        return np.nan, np.nan
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).astype(int)
        ca = ca[idx]
        n = ca.n_frames
    vals = []
    for i in range(n - 1):
        vals.extend(md.rmsd(ca[i + 1 :], ca[i]).tolist())
    vals = np.asarray(vals, float)
    return float(np.median(vals)), float(np.quantile(vals, 0.9))


def contact_stats(ca, i0=0, i1=None, min_sep=4, cutoff_nm=0.8):
    n = ca.n_atoms
    i1 = n if i1 is None else i1
    idxs = list(range(i0, i1))
    pairs = np.array([[a, b] for ia, a in enumerate(idxs) for b in idxs[ia + min_sep :]], dtype=int)
    if len(pairs) == 0:
        return dict(mean_occ=np.nan, frac_persist=np.nan, frac_labile=np.nan, entropy=np.nan)
    dists = md.compute_distances(ca, pairs)
    occ = (dists < cutoff_nm).mean(axis=0)
    p = np.clip(occ, 1e-6, 1 - 1e-6)
    return dict(
        mean_occ=float(occ.mean()),
        frac_persist=float(np.mean(occ >= 0.8)),
        frac_labile=float(np.mean((occ > 0.2) & (occ < 0.8))),
        entropy=float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)))),
    )


def interdomain_contacts(ca, a0, a1, b0, b1, cutoff_nm=0.8):
    pairs = np.array([[i, j] for i in range(a0, a1) for j in range(b0, b1)], dtype=int)
    if len(pairs) == 0:
        return dict(mean_count=np.nan, sd_count=np.nan, frac_persist=np.nan, entropy=np.nan)
    dists = md.compute_distances(ca, pairs)
    contact = dists < cutoff_nm
    counts = contact.sum(axis=1).astype(float)
    occ = contact.mean(axis=0)
    p = np.clip(occ, 1e-6, 1 - 1e-6)
    return dict(
        mean_count=float(counts.mean()),
        sd_count=float(counts.std()),
        frac_persist=float(np.mean(occ >= 0.5)),
        entropy=float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)))),
    )


def com_xyz(ca_frame_xyz, i0, i1):
    return ca_frame_xyz[i0:i1].mean(axis=0)


def orientation_angle(ca_xyz, a0, a1, b0, b1):
    """Angle between domain principal axes (first PC of CA coords)."""
    def axis(xyz):
        x = xyz - xyz.mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return vt[0]

    va, vb = axis(ca_xyz[a0:a1]), axis(ca_xyz[b0:b1])
    c = float(np.clip(np.abs(np.dot(va, vb)), -1, 1))
    return float(np.degrees(np.arccos(c)))


def region_rmsf(rmsf, ab_id, chain_imgt, v_len):
    cdr = pd.read_csv(CDR_IDX)
    sub = cdr[(cdr.id == ab_id) & (cdr.chain.astype(str) == chain_imgt)].sort_values("sequence_index")
    out = dict(rmsf_cdr=np.nan, rmsf_fw=np.nan, rmsf_cdr3=np.nan)
    if len(sub) == 0:
        return out
    # only variable indices
    sub = sub[sub.sequence_index < v_len]
    if len(sub) == 0 or sub.sequence_index.max() >= len(rmsf):
        return out
    is_cdr = sub.region.astype(str).str.upper().str.contains("CDR")
    is_cdr3 = sub.region.astype(str).str.upper().str.contains("CDR3")
    idx = sub.sequence_index.values.astype(int)
    if is_cdr.any():
        out["rmsf_cdr"] = float(rmsf[idx[is_cdr.values]].mean())
    if (~is_cdr).any():
        out["rmsf_fw"] = float(rmsf[idx[~is_cdr.values]].mean())
    if is_cdr3.any():
        out["rmsf_cdr3"] = float(rmsf[idx[is_cdr3.values]].mean())
    return out


def align_and_rmsf(ca, atom_indices_for_align):
    """Superpose using subset of CA indices, return per-residue RMSF on full CA."""
    # mdtraj superpose atom_indices refers to atoms in this traj
    ref = ca
    ca.superpose(ref, 0, atom_indices=atom_indices_for_align)
    return md.rmsf(ca, ca, 0)


def ss_entropy(traj, i0, i1):
    try:
        # dssp needs full traj protein; slice residues via topology
        # Use CA indices as residue indices in single-chain traj
        ca = ca_slice(traj)
        # mdtraj compute_dssp on full traj
        dssp = md.compute_dssp(traj, simplified=True)  # frames x residues
        sub = dssp[:, i0:i1]
        # per-res Shannon over {H,E,C}
        ents = []
        for j in range(sub.shape[1]):
            col = sub[:, j]
            _, counts = np.unique(col, return_counts=True)
            p = counts / counts.sum()
            ents.append(float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()))
        return float(np.mean(ents))
    except Exception:
        return np.nan


def native_contacts_from_pdb(pdb_path: Path, chain_id: str, i0: int, i1: int, cutoff_nm=0.8, min_sep=4):
    """Reference CA contacts within residue range [i0,i1) on given PDB chain (first model)."""
    t = md.load(str(pdb_path))
    # ESMFold Fab: chain A heavy, B light — but residue indices may be offset on B
    # Select by chain id then CA in order
    tops = list(t.topology.chains)
    ch_map = {c.index: c for c in tops}
    # Prefer chain id letter
    target = None
    for c in tops:
        if c.chain_id == chain_id:
            target = c
            break
    if target is None:
        target = tops[0 if chain_id == "A" else min(1, len(tops) - 1)]
    ca_atoms = [a.index for a in target.atoms if a.name == "CA"]
    if len(ca_atoms) < i1:
        # fallback: all CA in order for that chain length mismatch
        pass
    ca_atoms = ca_atoms[i0:i1] if len(ca_atoms) >= i1 else ca_atoms
    # pairs within selection
    n = len(ca_atoms)
    pairs_local = [(i, j) for i in range(n) for j in range(i + min_sep, n)]
    if not pairs_local:
        return np.zeros((0, 2), dtype=int), []
    atom_pairs = np.array([[ca_atoms[i], ca_atoms[j]] for i, j in pairs_local], dtype=int)
    d = md.compute_distances(t, atom_pairs)[0]
    keep = d < cutoff_nm
    native_atom_pairs = atom_pairs[keep]
    # also return local index pairs for mapping onto BioEmu single-chain CA 0..n-1
    native_local = np.array([[i, j] for (i, j), k in zip(pairs_local, keep) if k], dtype=int)
    return native_local, native_atom_pairs


def q_fraction(ca, native_local_pairs, cutoff_nm=0.8):
    if native_local_pairs is None or len(native_local_pairs) == 0:
        return np.full(ca.n_frames, np.nan)
    # ca atoms are 0..n-1 corresponding to residues
    # compute_distances needs atom indices in ca traj — CA traj atom i = residue i
    pairs = native_local_pairs.astype(int)
    d = md.compute_distances(ca, pairs)
    return (d < cutoff_nm).mean(axis=1)


def light_descriptors(ab_id: str, n_max: int | None = None):
    row = MAN.loc[ab_id]
    vl, cl_end = int(row.VL_len), int(row.light_length)
    cl0 = vl
    out_dir = SAMPLES / f"{ab_id}_VLCL"
    traj = load_traj(out_dir, n_max=n_max)
    ca = ca_slice(traj)
    assert ca.n_atoms == cl_end, (ab_id, ca.n_atoms, cl_end)

    # L1 whole-chain alignment
    ca_w = ca_slice(traj)
    ca_w.superpose(ca_w, 0)
    rmsf_w = md.rmsf(ca_w, ca_w, 0)
    med_r, q90_r = pairwise_ca_rmsd(ca_w)
    rg = md.compute_rg(ca_w)
    cst = contact_stats(ca_w)
    ss_ent = ss_entropy(traj, 0, cl_end)

    feat = {
        "id": ab_id,
        "n_valid": int(traj.n_frames),
        "L1_ca_rmsd_median": med_r,
        "L1_ca_rmsd_q90": q90_r,
        "L1_ca_rmsf_mean": float(rmsf_w.mean()),
        "L1_ca_rmsf_q90": float(np.quantile(rmsf_w, 0.9)),
        "L1_rg_mean": float(rg.mean()),
        "L1_rg_sd": float(rg.std()),
        "L1_contact_mean_occ": cst["mean_occ"],
        "L1_contact_frac_persistent": cst["frac_persist"],
        "L1_contact_frac_labile": cst["frac_labile"],
        "L1_contact_occ_entropy": cst["entropy"],
        "L1_ss_entropy": ss_ent,
    }

    # L2 VL under LOCAL_VARIABLE_ALIGNMENT
    ca_l = ca_slice(traj)
    align_idx = list(range(0, vl))
    rmsf_l = align_and_rmsf(ca_l, align_idx)
    med_vl, q90_vl = pairwise_ca_rmsd(ca_l.atom_slice(align_idx))
    cst_vl = contact_stats(ca_l, 0, vl)
    reg = region_rmsf(rmsf_l[:vl], ab_id, "L", vl)
    feat.update(
        {
            "L2_VL_rmsf_mean_local": float(rmsf_l[:vl].mean()),
            "L2_VL_rmsf_q90_local": float(np.quantile(rmsf_l[:vl], 0.9)),
            "L2_VL_rmsf_cdr_local": reg["rmsf_cdr"],
            "L2_VL_rmsf_cdr3_local": reg["rmsf_cdr3"],
            "L2_VL_contact_persist_local": cst_vl["frac_persist"],
            "L2_VL_contact_entropy_local": cst_vl["entropy"],
            "L2_VL_pairwise_rmsd_median_local": med_vl,
            # whole-chain frame VL RMSF (includes domain motion)
            "L2_VL_rmsf_mean_whole": float(rmsf_w[:vl].mean()),
            "L2_VL_rmsf_q90_whole": float(np.quantile(rmsf_w[:vl], 0.9)),
        }
    )

    # L4 coupling
    xyz = ca_w.xyz  # nm
    com_d = []
    angs = []
    for f in range(ca_w.n_frames):
        c1 = com_xyz(xyz[f], 0, vl)
        c2 = com_xyz(xyz[f], cl0, cl_end)
        com_d.append(float(np.linalg.norm(c1 - c2)))
        angs.append(orientation_angle(xyz[f], 0, vl, cl0, cl_end))
    ic = interdomain_contacts(ca_w, 0, vl, cl0, cl_end)
    feat.update(
        {
            "L4_VLCL_com_dist_mean": float(np.mean(com_d)),
            "L4_VLCL_com_dist_sd": float(np.std(com_d)),
            "L4_VLCL_orient_angle_mean": float(np.mean(angs)),
            "L4_VLCL_orient_angle_sd": float(np.std(angs)),
            "L4_VLCL_inter_contact_mean": ic["mean_count"],
            "L4_VLCL_inter_contact_sd": ic["sd_count"],
            "L4_VLCL_inter_contact_persist": ic["frac_persist"],
            "L4_VLCL_inter_contact_entropy": ic["entropy"],
        }
    )

    # L5 ESMFold-reference native likeness (light chain = chain B)
    pdb = FAB_STR / f"{ab_id}.pdb"
    if pdb.exists():
        # Build reference contacts from Fab light chain CA in residue order
        tref = md.load(str(pdb))
        # chain B
        chains = list(tref.topology.chains)
        light = chains[1] if len(chains) > 1 else chains[0]
        ca_ref = [a.index for a in light.atoms if a.name == "CA"]
        if len(ca_ref) == cl_end:
            # native pairs on reference structure
            def native_pairs(i0, i1):
                atoms = ca_ref[i0:i1]
                n = len(atoms)
                pl = [(i, j) for i in range(n) for j in range(i + 4, n)]
                if not pl:
                    return np.zeros((0, 2), int)
                ap = np.array([[atoms[i], atoms[j]] for i, j in pl], int)
                d0 = md.compute_distances(tref, ap)[0]
                keep = d0 < 0.8
                return np.array([[i0 + i, i0 + j] for (i, j), k in zip(pl, keep) if k], int)

            # For BioEmu single-chain, residue i maps to CA atom i
            q_all = q_fraction(ca_w, native_pairs(0, cl_end))
            q_vl = q_fraction(ca_w, native_pairs(0, vl))
            q_cl = q_fraction(ca_w, native_pairs(vl, cl_end))
            # interdomain native: contacts between VL and CL in reference
            atoms_vl = ca_ref[:vl]
            atoms_cl = ca_ref[vl:cl_end]
            pl = [(i, j) for i in range(vl) for j in range(cl_end - vl)]
            if pl:
                ap = np.array([[atoms_vl[i], atoms_cl[j]] for i, j in pl], int)
                d0 = md.compute_distances(tref, ap)[0]
                keep = d0 < 0.8
                native_inter = np.array([[i, vl + j] for (i, j), k in zip(pl, keep) if k], int)
            else:
                native_inter = np.zeros((0, 2), int)
            q_inter = q_fraction(ca_w, native_inter)

            def agg(q, prefix):
                q = np.asarray(q, float)
                m = np.isfinite(q)
                if not m.any():
                    return {}
                qq = q[m]
                return {
                    f"{prefix}_mean": float(qq.mean()),
                    f"{prefix}_q10": float(np.quantile(qq, 0.1)),
                    f"{prefix}_sd": float(qq.std()),
                    f"{prefix}_frac_high": float(np.mean(qq >= 0.7)),
                }

            feat.update(agg(q_all, "L5_Q_full"))
            feat.update(agg(q_vl, "L5_Q_VL"))
            feat.update(agg(q_cl, "L5_Q_CL"))
            feat.update(agg(q_inter, "L5_Q_inter"))

    return feat


def isolated_vl_stats(ab_id: str, n_max: int = 16):
    d = ISO / f"{ab_id}_VL"
    traj = load_traj(d, n_max=n_max)
    ca = ca_slice(traj)
    ca.superpose(ca, 0)
    rmsf = md.rmsf(ca, ca, 0)
    med, _ = pairwise_ca_rmsd(ca)
    cst = contact_stats(ca)
    vl = ca.n_atoms
    reg = region_rmsf(rmsf, ab_id, "L", vl)
    return {
        "iso_VL_rmsf_mean": float(rmsf.mean()),
        "iso_VL_rmsf_cdr": reg["rmsf_cdr"],
        "iso_VL_rmsf_cdr3": reg["rmsf_cdr3"],
        "iso_VL_pairwise_rmsd_median": med,
        "iso_VL_contact_persist": cst["frac_persist"],
        "iso_VL_contact_entropy": cst["entropy"],
        "iso_VL_ss_entropy": ss_entropy(traj, 0, vl),
    }


def light_context_delta(ab_id: str, n_ctx: int | None, n_iso: int = 16):
    f = light_descriptors(ab_id, n_max=n_ctx)
    iso = isolated_vl_stats(ab_id, n_max=n_iso)
    delta = {
        "id": ab_id,
        "L3_delta_VL_rmsf_mean": f["L2_VL_rmsf_mean_local"] - iso["iso_VL_rmsf_mean"],
        "L3_delta_VL_rmsf_cdr": f["L2_VL_rmsf_cdr_local"] - iso["iso_VL_rmsf_cdr"],
        "L3_delta_VL_rmsf_cdr3": f["L2_VL_rmsf_cdr3_local"] - iso["iso_VL_rmsf_cdr3"],
        "L3_delta_VL_pairwise_rmsd": f["L2_VL_pairwise_rmsd_median_local"] - iso["iso_VL_pairwise_rmsd_median"],
        "L3_delta_VL_contact_persist": f["L2_VL_contact_persist_local"] - iso["iso_VL_contact_persist"],
        "L3_delta_VL_contact_entropy": f["L2_VL_contact_entropy_local"] - iso["iso_VL_contact_entropy"],
        "L3_delta_VL_ss_entropy": f.get("L1_ss_entropy", np.nan) - iso["iso_VL_ss_entropy"],
    }
    # note: L1_ss is whole chain; better VL-only ss
    try:
        traj = load_traj(SAMPLES / f"{ab_id}_VLCL", n_max=n_ctx)
        vl = int(MAN.loc[ab_id, "VL_len"])
        delta["L3_delta_VL_ss_entropy"] = ss_entropy(traj, 0, vl) - iso["iso_VL_ss_entropy"]
    except Exception:
        pass
    return delta, f, iso


def heavy_descriptors(ab_id: str, n_max: int | None = None):
    row = MAN.loc[ab_id]
    vh, h_end = int(row.VH_len), int(row.heavy_length)
    ch0 = vh
    out_dir = SAMPLES / f"{ab_id}_VHCH1"
    traj = load_traj(out_dir, n_max=n_max)
    ca = ca_slice(traj)
    assert ca.n_atoms == h_end, (ab_id, ca.n_atoms, h_end)

    ca_w = ca_slice(traj)
    ca_w.superpose(ca_w, 0)
    rmsf_w = md.rmsf(ca_w, ca_w, 0)
    med_r, q90_r = pairwise_ca_rmsd(ca_w)
    rg = md.compute_rg(ca_w)
    # CH1 local RMSD to ESMFold Fab CH1
    pdb = FAB_STR / f"{ab_id}.pdb"
    ch1_rmsd = []
    ch1_q = []
    ch1_rg = []
    if pdb.exists():
        tref = md.load(str(pdb))
        heavy = list(tref.topology.chains)[0]
        ca_ref = [a.index for a in heavy.atoms if a.name == "CA"]
        if len(ca_ref) == h_end:
            # reference CH1 coords
            ref_ch1 = tref.xyz[0, ca_ref[ch0:h_end], :]
            # native CH1 contacts
            atoms = ca_ref[ch0:h_end]
            n = len(atoms)
            pl = [(i, j) for i in range(n) for j in range(i + 4, n)]
            if pl:
                ap = np.array([[atoms[i], atoms[j]] for i, j in pl], int)
                d0 = md.compute_distances(tref, ap)[0]
                keep = d0 < 0.8
                native_local = np.array([[ch0 + i, ch0 + j] for (i, j), k in zip(pl, keep) if k], int)
            else:
                native_local = np.zeros((0, 2), int)
            for f in range(ca_w.n_frames):
                # local align CH1
                mob = ca_w.xyz[f, ch0:h_end, :]
                # Kabsch
                def kabsch_rmsd(a, b):
                    a = a - a.mean(0)
                    b = b - b.mean(0)
                    H = a.T @ b
                    U, S, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T
                    if np.linalg.det(R) < 0:
                        Vt[-1] *= -1
                        R = Vt.T @ U.T
                    a2 = a @ R
                    return float(np.sqrt(((a2 - b) ** 2).sum(axis=1).mean()))

                ch1_rmsd.append(kabsch_rmsd(mob, ref_ch1))
                ch1_rg.append(float(np.linalg.norm(mob - mob.mean(0), axis=1).mean()))
            ch1_q = q_fraction(ca_w, native_local)

    ca_l = ca_slice(traj)
    rmsf_l = align_and_rmsf(ca_l, list(range(0, vh)))
    reg = region_rmsf(rmsf_l[:vh], ab_id, "H", vh)
    cst_vh = contact_stats(ca_l, 0, vh)

    # isolated VH
    iso_dir = ISO / f"{ab_id}_VH"
    iso = load_traj(iso_dir, n_max=16)
    ca_i = ca_slice(iso)
    ca_i.superpose(ca_i, 0)
    rmsf_i = md.rmsf(ca_i, ca_i, 0)
    reg_i = region_rmsf(rmsf_i, ab_id, "H", ca_i.n_atoms)
    cst_i = contact_stats(ca_i)

    xyz = ca_w.xyz
    com_d, angs = [], []
    for f in range(ca_w.n_frames):
        com_d.append(float(np.linalg.norm(com_xyz(xyz[f], 0, vh) - com_xyz(xyz[f], ch0, h_end))))
        angs.append(orientation_angle(xyz[f], 0, vh, ch0, h_end))
    ic = interdomain_contacts(ca_w, 0, vh, ch0, h_end)

    feat = {
        "id": ab_id,
        "arm_label": "UNPAIRED_HEAVY_CONTEXT",
        "n_valid": int(traj.n_frames),
        "H_full_ca_rmsd_median": med_r,
        "H_full_ca_rmsf_mean": float(rmsf_w.mean()),
        "H_full_rg_mean": float(rg.mean()),
        "H_full_rg_sd": float(rg.std()),
        "H1_delta_VH_rmsf_mean": float(rmsf_l[:vh].mean()) - float(rmsf_i.mean()),
        "H1_delta_VH_rmsf_cdr": reg["rmsf_cdr"] - reg_i["rmsf_cdr"],
        "H1_delta_VH_rmsf_cdr3": reg["rmsf_cdr3"] - reg_i["rmsf_cdr3"],
        "H1_delta_VH_contact_persist": cst_vh["frac_persist"] - cst_i["frac_persist"],
        "H1_delta_VH_contact_entropy": cst_vh["entropy"] - cst_i["entropy"],
        "H2_CH1_rmsd_to_esmfold_mean": float(np.mean(ch1_rmsd)) if ch1_rmsd else np.nan,
        "H2_CH1_rmsd_to_esmfold_q90": float(np.quantile(ch1_rmsd, 0.9)) if ch1_rmsd else np.nan,
        "H2_CH1_Q_mean": float(np.nanmean(ch1_q)) if len(ch1_q) else np.nan,
        "H2_CH1_Q_q10": float(np.nanquantile(ch1_q, 0.1)) if len(ch1_q) else np.nan,
        "H2_CH1_Q_frac_high": float(np.mean(np.asarray(ch1_q) >= 0.7)) if len(ch1_q) else np.nan,
        "H2_CH1_compactness_mean": float(np.mean(ch1_rg)) if ch1_rg else np.nan,
        "H2_CH1_ss_entropy": ss_entropy(traj, ch0, h_end),
        "H3_VHCH1_com_dist_mean": float(np.mean(com_d)),
        "H3_VHCH1_com_dist_sd": float(np.std(com_d)),
        "H3_VHCH1_orient_angle_sd": float(np.std(angs)),
        "H3_VHCH1_inter_contact_mean": ic["mean_count"],
        "H3_VHCH1_inter_contact_persist": ic["frac_persist"],
        "H3_VHCH1_inter_contact_sd": ic["sd_count"],
    }
    return feat


def prefixes_for_convergence(ab_id: str, arm: str, Ns=(16, 32, 64)):
    rows = []
    for n in Ns:
        try:
            if arm == "LIGHT":
                f = light_descriptors(ab_id, n_max=n)
                keys = [k for k in f if k.startswith("L1_") or k.startswith("L2_VL_rmsf_mean_local") or k.startswith("L4_VLCL_com")]
            else:
                f = heavy_descriptors(ab_id, n_max=n)
                keys = [k for k in f if k.startswith("H_") or k.startswith("H2_") or k.startswith("H1_delta_VH_rmsf_mean")]
            row = {"id": ab_id, "arm": arm, "N_prefix": n, "n_valid": f.get("n_valid", np.nan)}
            for k in keys:
                row[k] = f[k]
            rows.append(row)
        except Exception as e:
            rows.append({"id": ab_id, "arm": arm, "N_prefix": n, "error": f"{type(e).__name__}:{e}"})
    return rows


def decide_N(conv_df: pd.DataFrame, arm: str, metrics: list[str]):
    sub = conv_df[conv_df.arm == arm]
    ids = sub.id.unique()
    records = []
    for N in (16, 32):
        spear_list, nad_list = [], []
        for m in metrics:
            if m not in sub.columns:
                continue
            for ab in ids:
                a = sub[(sub.id == ab) & (sub.N_prefix == N)]
                b = sub[(sub.id == ab) & (sub.N_prefix == 64)]
                if len(a) != 1 or len(b) != 1:
                    continue
                # collect across abs for this metric later
            xs, ys = [], []
            for ab in ids:
                a = sub[(sub.id == ab) & (sub.N_prefix == N)]
                b = sub[(sub.id == ab) & (sub.N_prefix == 64)]
                if len(a) != 1 or len(b) != 1:
                    continue
                if m not in a or pd.isna(a.iloc[0][m]) or pd.isna(b.iloc[0][m]):
                    continue
                xs.append(float(a.iloc[0][m]))
                ys.append(float(b.iloc[0][m]))
            if len(xs) < 5:
                continue
            xs, ys = np.asarray(xs), np.asarray(ys)
            sp = spearmanr(xs, ys).statistic
            nad = np.median(np.abs(xs - ys) / (np.abs(ys) + 1e-8))
            records.append({"arm": arm, "N": N, "metric": m, "spearman_vs_64": float(sp), "median_NAD": float(nad), "n_pairs": len(xs)})
            spear_list.append(sp)
            nad_list.append(nad)
    tab = pd.DataFrame(records)
    chosen = 64
    for N in (16, 32):
        t = tab[tab.N == N]
        if len(t) == 0:
            continue
        if float(t.spearman_vs_64.median()) >= 0.95 and float(t.median_NAD.median()) <= 0.10:
            chosen = N
            break
    return tab, chosen


def main():
    mode = "convergence" if "--convergence" in sys.argv else "full"
    if mode == "convergence":
        ids = pd.read_csv(CTX / "pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv").id.tolist()
        rows = []
        for ab in ids:
            for arm in ("LIGHT", "HEAVY"):
                d = SAMPLES / f"{ab}_{'VLCL' if arm=='LIGHT' else 'VHCH1'}"
                if not d.exists():
                    continue
                rows.extend(prefixes_for_convergence(ab, arm))
        cdf = pd.DataFrame(rows)
        cdf.to_csv(CTX / "results/BIOEMU_CONTEXT_SAMPLE_CONVERGENCE_RAW.csv", index=False)
        light_metrics = [
            "L1_ca_rmsd_median",
            "L1_ca_rmsf_mean",
            "L1_rg_mean",
            "L1_contact_mean_occ",
            "L2_VL_rmsf_mean_local",
            "L4_VLCL_com_dist_mean",
            "L4_VLCL_com_dist_sd",
        ]
        heavy_metrics = [
            "H_full_ca_rmsd_median",
            "H_full_ca_rmsf_mean",
            "H_full_rg_mean",
            "H2_CH1_rmsd_to_esmfold_mean",
            "H2_CH1_Q_mean",
            "H1_delta_VH_rmsf_mean",
        ]
        tL, nL = decide_N(cdf, "LIGHT", light_metrics)
        tH, nH = decide_N(cdf, "HEAVY", heavy_metrics)
        tab = pd.concat([tL, tH], ignore_index=True)
        tab.to_csv(CTX / "BIOEMU_CONTEXT_SAMPLE_CONVERGENCE.csv", index=False)
        (CTX / "BIOEMU_CONTEXT_FROZEN_N_LIGHT.txt").write_text(str(nL) + "\n")
        (CTX / "BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt").write_text(str(nH) + "\n")
        md = CTX / "BIOEMU_CONTEXT_SAMPLE_COUNT_DECISION.md"
        md.write_text(
            f"""# BioEmu context sample-count decision

**SPEC_FROZEN_BEFORE_TARGET_SCORING**

| Arm | Frozen N |
|-----|----------|
| LIGHT_CHAIN (VL+CL) | **{nL}** |
| UNPAIRED_HEAVY (VH+CH1) | **{nH}** |

Rule: smallest of {{16,32}} with median Spearman vs N=64 ≥ 0.95 and median NAD ≤ 0.10; else 64.

See `BIOEMU_CONTEXT_SAMPLE_CONVERGENCE.csv`.

Model: bioemu-v1.2; MSA: singleseq a3m of reconstructed Fab chain; filter_samples=ON.
"""
        )
        print("LIGHT N", nL, "HEAVY N", nH)
        return

    # full feature tables
    nL = int((CTX / "BIOEMU_CONTEXT_FROZEN_N_LIGHT.txt").read_text().strip())
    nH = int((CTX / "BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt").read_text().strip()) if (CTX / "BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt").exists() else nL
    ids = MAN.index.tolist()
    if "--ids" in sys.argv:
        # comma list or file
        pass
    L1, L3, L4, L5, H = [], [], [], [], []
    for i, ab in enumerate(ids):
        ld = SAMPLES / f"{ab}_VLCL"
        if ld.exists():
            try:
                delta, f, _ = light_context_delta(ab, n_ctx=nL, n_iso=16)
                L1.append({k: v for k, v in f.items() if k == "id" or k.startswith("L1_") or k.startswith("L2_")})
                L3.append(delta)
                L4.append({k: v for k, v in f.items() if k == "id" or k.startswith("L4_")})
                L5.append({k: v for k, v in f.items() if k == "id" or k.startswith("L5_")})
            except Exception as e:
                print("LIGHT fail", ab, e, flush=True)
        hd = SAMPLES / f"{ab}_VHCH1"
        if hd.exists():
            try:
                H.append(heavy_descriptors(ab, n_max=nH))
            except Exception as e:
                print("HEAVY fail", ab, e, flush=True)
        if (i + 1) % 20 == 0:
            print(f"features {i+1}/{len(ids)}", flush=True)

    pd.DataFrame(L1).to_csv(CTX / "BIOEMU_FULL_LIGHT_FEATURES.csv", index=False)
    pd.DataFrame(L3).to_csv(CTX / "BIOEMU_LIGHT_CONTEXT_DELTA_FEATURES.csv", index=False)
    pd.DataFrame(L4).to_csv(CTX / "BIOEMU_LIGHT_VC_COUPLING_FEATURES.csv", index=False)
    pd.DataFrame(L5).to_csv(CTX / "BIOEMU_LIGHT_NATIVE_LIKENESS_FEATURES.csv", index=False)
    if H:
        pd.DataFrame(H).to_csv(CTX / "BIOEMU_UNPAIRED_HEAVY_FEATURES.csv", index=False)
    print("wrote feature tables", len(L1), len(H))


if __name__ == "__main__":
    main()
