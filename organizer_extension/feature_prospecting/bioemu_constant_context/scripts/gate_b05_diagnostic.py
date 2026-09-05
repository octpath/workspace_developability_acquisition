#!/usr/bin/env python3
"""Gate B0.5 — Low physical-pass diagnostic for BioEmu VL+CL context.

Produces:
  BIOEMU_FILTER_FAILURES.csv
  BIOEMU_DEFAULT_VS_STEERED.csv (when steered samples exist)
  results/B05_*.csv intermediates
  Updates BIOEMU_LOW_PASS_DIAGNOSTIC.md via separate writer or prints summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "bioemu_constant_context"
FAB_STR = FP / "fab_reconstruction/structures/esmfold_fab"
MAN = pd.read_csv(CTX / "BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv").set_index("id")
PILOT = pd.read_csv(CTX / "pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv").id.tolist()
SAMPLES = CTX / "cache/bioemu_samples"
B05 = CTX / "cache/b05_diagnostic"
PHYS_STEER = Path(
    "/workspace_developability_acquisition/organizer_extension/feature_prospecting/"
    "foundation_stability/envs/bioemu/lib/python3.11/site-packages/bioemu/config/steering/physical_steering.yaml"
)


def load_npz_batch(out_dir: Path):
    files = sorted(out_dir.glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(out_dir)
    seq = str(np.load(files[0])["sequence"].item())
    pos = torch.tensor(np.concatenate([np.load(f)["pos"] for f in files]))
    ori = torch.tensor(np.concatenate([np.load(f)["node_orientations"] for f in files]))
    return pos, ori, seq


def rebuild_raw_traj(out_dir: Path, dest_dir: Path | None = None) -> md.Trajectory:
    """Convert all npz to unfiltered trajectory (filter_samples=False)."""
    from bioemu.convert_chemgraph import save_pdb_and_xtc

    pos, ori, seq = load_npz_batch(out_dir)
    dest = dest_dir or (B05 / "filter_analysis" / out_dir.name)
    dest.mkdir(parents=True, exist_ok=True)
    top = dest / "topology_raw.pdb"
    xtc = dest / "samples_raw.xtc"
    if not (top.exists() and xtc.exists() and md.load(str(xtc), top=str(top)).n_frames == pos.shape[0]):
        save_pdb_and_xtc(
            pos_nm=pos,
            node_orientations=ori,
            sequence=seq,
            topology_path=top,
            xtc_path=xtc,
            filter_samples=False,
        )
    return md.load(str(xtc), top=str(top))


def filter_masks(traj: md.Trajectory):
    from bioemu.convert_chemgraph import _filter_unphysical_traj_masks

    ca_ok, cn_ok, noclash = _filter_unphysical_traj_masks(traj)
    return ca_ok, cn_ok, noclash


def continuity_break_locus(traj: md.Trajectory, vl_len: int, max_ca=4.5, max_cn=2.0):
    """Per-frame: where first chain-break occurs (VL / junction / CL / none)."""
    n = traj.n_residues
    pairs = np.array([(i, i + 1) for i in range(n - 1)])
    ca, _ = md.compute_contacts(traj, scheme="ca", contacts=pairs, periodic=False)
    ca = md.utils.in_units_of(ca, "nanometers", "angstrom")
    cn_pairs = []
    for i, j in pairs:
        ri, rj = traj.topology.residue(i), traj.topology.residue(j)
        c = list(ri.atoms_by_name("C"))[0].index
        n_ = list(rj.atoms_by_name("N"))[0].index
        cn_pairs.append((c, n_))
    cn = md.utils.in_units_of(md.compute_distances(traj, cn_pairs, periodic=False), "nanometers", "angstrom")
    loci = []
    for f in range(traj.n_frames):
        bad = np.where((ca[f] >= max_ca) | (cn[f] >= max_cn))[0]
        if len(bad) == 0:
            loci.append("none")
            continue
        i = int(bad[0])
        if i < vl_len - 1:
            loci.append("VL")
        elif i == vl_len - 1:
            loci.append("VL_CL_junction")
        else:
            loci.append("CL")
    return loci


def clash_locus(traj: md.Trajectory, vl_len: int, clash_distance=1.0):
    """Approximate clash region: VL-internal / CL-internal / interdomain / none."""
    from scipy.spatial import cKDTree

    atom2res = np.asarray([a.residue.index for a in traj.topology.atoms])
    out = []
    r_nm = md.utils.in_units_of(clash_distance, "angstrom", "nanometers")
    for f in range(traj.n_frames):
        tree = cKDTree(traj.xyz[f])
        pairs = tree.query_pairs(r=r_nm)
        kind = "none"
        for a, b in pairs:
            ra, rb = atom2res[a], atom2res[b]
            if abs(rb - ra) <= 2:
                continue
            # found a clash
            in_vl = ra < vl_len and rb < vl_len
            in_cl = ra >= vl_len and rb >= vl_len
            if in_vl:
                kind = "VL_internal_clash"
            elif in_cl:
                kind = "CL_internal_clash"
            else:
                kind = "interdomain_clash"
                break  # interdomain is most informative for multi-domain
            # keep scanning for interdomain
        if kind != "interdomain_clash":
            # reclassify if any interdomain exists
            for a, b in pairs:
                ra, rb = atom2res[a], atom2res[b]
                if abs(rb - ra) <= 2:
                    continue
                if (ra < vl_len) != (rb < vl_len):
                    kind = "interdomain_clash"
                    break
        out.append(kind)
    return out


def classify_failures(traj: md.Trajectory, vl_len: int) -> pd.DataFrame:
    ca_ok, cn_ok, noclash = filter_masks(traj)
    contig_ok = ca_ok & cn_ok
    pass_all = contig_ok & noclash
    break_only = (~contig_ok) & noclash
    clash_only = contig_ok & (~noclash)
    both = (~contig_ok) & (~noclash)
    loci_break = continuity_break_locus(traj, vl_len)
    loci_clash = clash_locus(traj, vl_len)
    rows = []
    for i in range(traj.n_frames):
        if pass_all[i]:
            reason = "PASS"
        elif both[i]:
            reason = "both_break_and_clash"
        elif break_only[i]:
            reason = "chain_discontinuity"
        elif clash_only[i]:
            reason = "steric_clash"
        else:
            reason = "other"
        rows.append(
            {
                "frame": i,
                "reason": reason,
                "ca_ok": bool(ca_ok[i]),
                "cn_ok": bool(cn_ok[i]),
                "noclash": bool(noclash[i]),
                "break_locus": loci_break[i],
                "clash_locus": loci_clash[i],
            }
        )
    return pd.DataFrame(rows)


def com_dist(ca_xyz, i0, i1, j0, j1):
    return float(np.linalg.norm(ca_xyz[i0:i1].mean(0) - ca_xyz[j0:j1].mean(0)))


def orient_angle(ca_xyz, a0, a1, b0, b1):
    def axis(x):
        x = x - x.mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return vt[0]

    c = float(np.clip(np.abs(np.dot(axis(ca_xyz[a0:a1]), axis(ca_xyz[b0:b1]))), -1, 1))
    return float(np.degrees(np.arccos(c)))


def frame_metrics(traj: md.Trajectory, vl_len: int) -> pd.DataFrame:
    ca = traj.atom_slice(traj.topology.select("name CA"))
    n = ca.n_atoms
    rg = md.compute_rg(ca)
    rows = []
    for f in range(ca.n_frames):
        rows.append(
            {
                "frame": f,
                "rg": float(rg[f]),
                "vl_cl_com": com_dist(ca.xyz[f], 0, vl_len, vl_len, n),
                "orient_angle": orient_angle(ca.xyz[f], 0, vl_len, vl_len, n),
            }
        )
    # continuity max CA gap
    pairs = np.array([(i, i + 1) for i in range(n - 1)])
    ca_d, _ = md.compute_contacts(ca, scheme="ca", contacts=pairs, periodic=False)
    ca_d = md.utils.in_units_of(ca_d, "nanometers", "angstrom")
    for f in range(ca.n_frames):
        rows[f]["max_ca_gap"] = float(ca_d[f].max())
        rows[f]["n_ca_breaks"] = int((ca_d[f] >= 4.5).sum())
    return pd.DataFrame(rows)


def ensemble_summary(traj: md.Trajectory, vl_len: int, ab_id: str) -> dict:
    ca = traj.atom_slice(traj.topology.select("name CA"))
    if ca.n_frames < 1:
        return {"id": ab_id, "n": 0}
    ca.superpose(ca, 0)
    rmsf = md.rmsf(ca, ca, 0)
    # pairwise rmsd subsample
    n = min(ca.n_frames, 24)
    idx = np.linspace(0, ca.n_frames - 1, n).astype(int)
    sub = ca[idx]
    vals = []
    for i in range(len(idx) - 1):
        vals.extend(md.rmsd(sub[i + 1 :], sub[i]).tolist())
    # contacts
    pairs = np.array([[i, j] for i in range(ca.n_atoms) for j in range(i + 4, ca.n_atoms)], dtype=int)
    if len(pairs):
        occ = (md.compute_distances(ca, pairs) < 0.8).mean(axis=0)
        persist = float((occ >= 0.8).mean())
    else:
        persist = np.nan
    mets = frame_metrics(traj, vl_len)
    # Q to ESMFold light chain if available
    q_mean = np.nan
    pdb = FAB_STR / f"{ab_id}.pdb"
    if pdb.exists() and ca.n_atoms == int(MAN.loc[ab_id, "light_length"]):
        tref = md.load(str(pdb))
        light = list(tref.topology.chains)[1]
        ca_ref = [a.index for a in light.atoms if a.name == "CA"]
        if len(ca_ref) == ca.n_atoms:
            pl = [(i, j) for i in range(ca.n_atoms) for j in range(i + 4, ca.n_atoms)]
            ap = np.array([[ca_ref[i], ca_ref[j]] for i, j in pl], int)
            d0 = md.compute_distances(tref, ap)[0]
            native = np.array([[i, j] for (i, j), k in zip(pl, d0 < 0.8) if k], int)
            if len(native):
                q = (md.compute_distances(ca, native) < 0.8).mean(axis=1)
                q_mean = float(q.mean())
    return {
        "id": ab_id,
        "n": int(ca.n_frames),
        "rmsd_median": float(np.median(vals)) if vals else np.nan,
        "rmsd_q90": float(np.quantile(vals, 0.9)) if vals else np.nan,
        "rmsf_mean": float(rmsf.mean()),
        "contact_persist": persist,
        "rg_mean": float(mets.rg.mean()),
        "rg_sd": float(mets.rg.std()),
        "com_mean": float(mets.vl_cl_com.mean()),
        "com_sd": float(mets.vl_cl_com.std()),
        "orient_mean": float(mets.orient_angle.mean()),
        "orient_sd": float(mets.orient_angle.std()),
        "Q_esmfold_mean": q_mean,
    }


def analyze_filter_failures():
    rows = []
    bias_rows = []
    for ab in PILOT:
        src = SAMPLES / f"{ab}_VLCL"
        if not src.exists():
            continue
        vl = int(MAN.loc[ab, "VL_len"])
        print("filter analysis", ab, flush=True)
        raw = rebuild_raw_traj(src)
        clf = classify_failures(raw, vl)
        clf.insert(0, "id", ab)
        rows.append(clf)
        # selection bias: RAW vs PASS metrics
        mets = frame_metrics(raw, vl)
        mets["pass"] = clf.reason.eq("PASS").values
        for flag, name in [(True, "PASS"), (False, "FAIL")]:
            sub = mets[mets["pass"] == flag]
            if len(sub) == 0:
                continue
            bias_rows.append(
                {
                    "id": ab,
                    "subset": name,
                    "n": len(sub),
                    "rg_mean": float(sub.rg.mean()),
                    "rg_sd": float(sub.rg.std()),
                    "com_mean": float(sub.vl_cl_com.mean()),
                    "com_sd": float(sub.vl_cl_com.std()),
                    "orient_mean": float(sub.orient_angle.mean()),
                    "orient_sd": float(sub.orient_angle.std()),
                    "max_ca_gap_mean": float(sub.max_ca_gap.mean()),
                    "n_ca_breaks_mean": float(sub.n_ca_breaks.mean()),
                }
            )
        # reason summary
        print(ab, clf.reason.value_counts().to_dict(), flush=True)

    allf = pd.concat(rows, ignore_index=True)
    allf.to_csv(CTX / "BIOEMU_FILTER_FAILURES.csv", index=False)
    # aggregate
    agg = (
        allf.groupby(["id", "reason"]).size().unstack(fill_value=0)
    )
    agg["n_total"] = agg.sum(axis=1)
    if "PASS" in agg.columns:
        agg["pass_rate"] = agg["PASS"] / agg["n_total"]
    agg.to_csv(CTX / "results/B05_FILTER_FAILURE_SUMMARY.csv")
    pd.DataFrame(bias_rows).to_csv(CTX / "results/B05_RAW_VS_PASS_BIAS.csv", index=False)

    # locus among failures
    fails = allf[allf.reason != "PASS"]
    loc = pd.crosstab(fails.get("break_locus", []), fails.get("reason", []))
    loc.to_csv(CTX / "results/B05_BREAK_LOCUS_BY_REASON.csv")
    clash_ct = fails.clash_locus.value_counts()
    clash_ct.to_csv(CTX / "results/B05_CLASH_LOCUS_COUNTS.csv")
    return allf, pd.DataFrame(bias_rows)


def sample_construct(sequence: str, out_dir: Path, n: int, seed: int, steered: bool = False):
    from bioemu.sample import main as bioemu_sample, count_samples_in_output_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    have = count_samples_in_output_dir(out_dir) if any(out_dir.glob("batch_*.npz")) else 0
    if have >= n and (out_dir / "samples.xtc").exists():
        return {"status": "CACHED", "npz": have, "phys": md.load(str(out_dir / "samples.xtc"), top=str(out_dir / "topology.pdb")).n_frames}

    a3m = out_dir / "seq.a3m"
    if not a3m.exists():
        a3m.write_text(f">seq\n{sequence}\n")
    t0 = time.time()
    kwargs = dict(
        sequence=str(a3m),
        num_samples=n,
        output_dir=str(out_dir),
        model_name="bioemu-v1.2",
        cache_embeds_dir=str(CTX / "cache/bioemu_embeds"),
        filter_samples=True,
        base_seed=seed,
        batch_size_100=20,
    )
    if steered:
        kwargs["denoiser_config"] = str(PHYS_STEER)
    bioemu_sample(**kwargs)
    phys = md.load(str(out_dir / "samples.xtc"), top=str(out_dir / "topology.pdb")).n_frames
    npz = count_samples_in_output_dir(out_dir)
    return {"status": "OK", "npz": npz, "phys": phys, "pass_rate": phys / max(npz, 1), "runtime_s": time.time() - t0}


def run_domain_compare(n_abs: int = 4, n_samples: int = 64):
    """Target-blind subset: first 4 pilot Abs spanning kappa/lambda if possible."""
    pil = pd.read_csv(CTX / "pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv")
    # take 2 kappa + 2 lambda
    sel = []
    for locus, k in [("kappa", 2), ("lambda", 2)]:
        sub = pil[pil.light_locus == locus] if "light_locus" in pil.columns else pil
        if "light_locus" not in pil.columns:
            # join from manifest
            pil2 = pil.merge(MAN.reset_index()[["id", "light_locus"]], on="id")
            sub = pil2[pil2.light_locus == locus]
        sel.extend(sub.id.head(k).tolist())
    sel = sel[:n_abs]
    rows = []
    for ab in sel:
        row = MAN.loc[ab]
        vl = int(row.VL_len)
        constructs = {
            "VL": str(row.light_fab_seq)[:vl],
            "CL": str(row.light_fab_seq)[vl:],
            "VLCL": str(row.light_fab_seq),
        }
        for name, seq in constructs.items():
            out = B05 / "domain_compare" / f"{ab}_{name}"
            print(f"domain {ab} {name} len={len(seq)}", flush=True)
            res = sample_construct(seq, out, n_samples, seed=abs(hash(f"{ab}_{name}_b05")) % 2**31)
            rows.append({"id": ab, "construct": name, "seq_len": len(seq), **res})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(CTX / "results/B05_DOMAIN_COMPARE_PASS.csv", index=False)
    return rows


def run_default_vs_steered(n_samples: int = 64):
    rows = []
    for ab in PILOT:
        seq = str(MAN.loc[ab, "light_fab_seq"])
        # A: default — prefer rebuild stats from existing pilot samples
        src = SAMPLES / f"{ab}_VLCL"
        # For fair compare, also sample fresh default into b05 if needed; use existing npz for A
        t0 = time.time()
        if src.exists() and any(src.glob("batch_*.npz")):
            # use first n_samples npz frames via raw rebuild + filter
            raw = rebuild_raw_traj(src, B05 / "filter_analysis" / f"{ab}_VLCL")
            raw = raw[:n_samples] if raw.n_frames >= n_samples else raw
            ca_ok, cn_ok, noclash = filter_masks(raw)
            pass_idx = np.where(ca_ok & cn_ok & noclash)[0]
            pass_traj = raw.slice(pass_idx) if len(pass_idx) else raw.slice([])
            # write pass traj for ensemble summary
            summ = ensemble_summary(pass_traj, int(MAN.loc[ab, "VL_len"]), ab) if len(pass_idx) else {"id": ab, "n": 0}
            rows.append(
                {
                    "id": ab,
                    "method": "A_default_postfilter",
                    "npz": int(raw.n_frames),
                    "phys": int(len(pass_idx)),
                    "pass_rate": float(len(pass_idx) / max(raw.n_frames, 1)),
                    "runtime_s": np.nan,  # prior pilot runtime
                    **{f"ens_{k}": v for k, v in summ.items() if k != "id"},
                }
            )
        # B: steered
        out = B05 / "steered" / f"{ab}_VLCL"
        print(f"steered {ab}", flush=True)
        res = sample_construct(seq, out, n_samples, seed=abs(hash(f"{ab}_steer")) % 2**31, steered=True)
        # ensemble on filtered xtc
        if (out / "samples.xtc").exists():
            traj = md.load(str(out / "samples.xtc"), top=str(out / "topology.pdb"))
            summ = ensemble_summary(traj, int(MAN.loc[ab, "VL_len"]), ab)
        else:
            summ = {"n": 0}
        rows.append(
            {
                "id": ab,
                "method": "B_physical_steered_postfilter",
                "npz": res.get("npz"),
                "phys": res.get("phys"),
                "pass_rate": res.get("pass_rate"),
                "runtime_s": res.get("runtime_s"),
                **{f"ens_{k}": v for k, v in summ.items() if k != "id"},
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CTX / "BIOEMU_DEFAULT_VS_STEERED.csv", index=False)
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    B05.mkdir(parents=True, exist_ok=True)
    if mode in ("filter", "all"):
        analyze_filter_failures()
    if mode in ("domain", "all"):
        run_domain_compare()
    if mode in ("steer", "all"):
        run_default_vs_steered()
    print("B05 mode done:", mode)


if __name__ == "__main__":
    main()
