#!/usr/bin/env python3
"""Extract BioEmu ensemble descriptors + sample-count convergence metrics."""
from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
SAMPLES = V2 / "cache/bioemu_samples"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"
SPEC = json.loads((V2 / "FOUNDATION_STABILITY_V2_SPEC.json").read_text())


def load_traj(out_dir: Path, n_max: int | None = None):
    top = out_dir / "topology.pdb"
    # samples may be xtc or multiple
    xtc = list(out_dir.glob("*.xtc"))
    if not top.exists() or not xtc:
        # try pdb ensemble
        pdbs = sorted(out_dir.glob("*.pdb"))
        if len(pdbs) <= 1:
            raise FileNotFoundError(out_dir)
        # load via mdtraj
        traj = md.load(str(pdbs[0]))
        for p in pdbs[1:]:
            traj = traj.join(md.load(str(p)))
    else:
        traj = md.load(str(xtc[0]), top=str(top))
        for x in xtc[1:]:
            traj = traj.join(md.load(str(x), top=str(top)))
    if n_max is not None:
        traj = traj[:n_max]
    return traj


def pairwise_ca_rmsd(traj):
    ca = traj.atom_slice(traj.topology.select("name CA"))
    n = ca.n_frames
    if n < 2:
        return np.nan, np.nan
    # Subsample frames for pairwise spread (target-blind; max 24 frames)
    if n > 24:
        idx = np.linspace(0, n - 1, 24).astype(int)
        ca = ca[idx]
        n = ca.n_frames
    vals = []
    for i in range(n):
        # vectorized rmsd of all later frames vs i
        if i + 1 >= n:
            break
        vals.extend(md.rmsd(ca[i + 1 :], ca[i]).tolist())
    vals = np.asarray(vals, float)
    return float(np.median(vals)), float(np.quantile(vals, 0.9))


def descriptors(traj, ab_id: str, chain_label: str):
    ca_idx = traj.topology.select("name CA")
    ca = traj.atom_slice(ca_idx)
    n_res = ca.n_atoms
    # RMSF relative to mean structure
    avg = ca.slice(0)  # placeholder
    # mdtraj rmsf
    rmsf = md.rmsf(ca, ca, 0)
    # better: align then rmsf
    ca.superpose(ca, 0)
    rmsf = md.rmsf(ca, ca, 0)
    rg = md.compute_rg(ca)

    # contacts |i-j|>=4, CA<8A occupancy
    pairs = np.array([[i, j] for i in range(n_res) for j in range(i + 4, n_res)], dtype=int)
    if len(pairs) == 0:
        mean_occ = frac_persist = frac_labile = ent = np.nan
    else:
        dists = md.compute_distances(ca, pairs)  # nm
        contact = dists < 0.8
        occ = contact.mean(axis=0)
        mean_occ = float(occ.mean())
        frac_persist = float(np.mean(occ >= 0.8))
        frac_labile = float(np.mean((occ > 0.2) & (occ < 0.8)))
        # entropy of occupancy binary-ish
        p = np.clip(occ, 1e-6, 1 - 1e-6)
        ent = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))

    med_rmsd, q90_rmsd = pairwise_ca_rmsd(ca)

    # region RMSF via CDR index
    cdr = pd.read_csv(CDR_IDX)
    ch = "H" if chain_label == "VH" else "L"
    sub = cdr[(cdr.id == ab_id) & (cdr.chain.astype(str) == ch)].sort_values("sequence_index")
    rmsf_cdr = rmsf_fw = rmsf_cdr3 = np.nan
    if len(sub) and len(sub) == n_res:
        is_cdr = sub.region.astype(str).str.upper().str.contains("CDR")
        is_fw = ~is_cdr
        is_cdr3 = sub.region.astype(str).str.upper().str.contains("CDR3") | sub.region.astype(str).str.upper().str.contains("H3") | sub.region.astype(str).str.upper().str.contains("L3")
        if is_cdr.any():
            rmsf_cdr = float(rmsf[is_cdr.values].mean())
        if is_fw.any():
            rmsf_fw = float(rmsf[is_fw.values].mean())
        if is_cdr3.any():
            rmsf_cdr3 = float(rmsf[is_cdr3.values].mean())

    return {
        "n_valid": int(traj.n_frames),
        "n_res": n_res,
        "ca_rmsd_median": med_rmsd,
        "ca_rmsd_q90": q90_rmsd,
        "ca_rmsf_mean": float(np.mean(rmsf)),
        "ca_rmsf_q90": float(np.quantile(rmsf, 0.9)),
        "ca_rmsf_max": float(np.max(rmsf)),
        "rg_mean": float(np.mean(rg)),
        "rg_sd": float(np.std(rg)),
        "contact_mean_occ": mean_occ,
        "contact_frac_persistent": frac_persist,
        "contact_frac_labile": frac_labile,
        "contact_occ_entropy": ent,
        "rmsf_cdr_mean": rmsf_cdr,
        "rmsf_fw_mean": rmsf_fw,
        "rmsf_cdr3_mean": rmsf_cdr3,
    }


def prefixes_features(ab_id, chain, Ns=(16, 32, 64)):
    out_dir = SAMPLES / f"{ab_id}_{chain}"
    rows = []
    for n in Ns:
        try:
            traj = load_traj(out_dir, n_max=n)
            if traj.n_frames < max(8, n // 2):
                continue
            d = descriptors(traj, ab_id, chain)
            d.update({"id": ab_id, "chain": chain, "N_prefix": n})
            rows.append(d)
        except Exception as e:
            rows.append({"id": ab_id, "chain": chain, "N_prefix": n, "error": f"{type(e).__name__}:{e}"})
    return rows


def convergence_table(ids):
    rows = []
    for ab in ids:
        for ch in ("VH", "VL"):
            rows.extend(prefixes_features(ab, ch))
    df = pd.DataFrame(rows)
    # compare N=16,32 vs 64
    metrics = [
        "ca_rmsd_median",
        "ca_rmsf_mean",
        "ca_rmsf_q90",
        "rg_mean",
        "contact_mean_occ",
        "rmsf_cdr_mean",
        "rmsf_fw_mean",
    ]
    conv_rows = []
    for n in (16, 32):
        for m in metrics:
            xs, ys = [], []
            for (ab, ch), g in df.groupby(["id", "chain"]):
                a = g[g.N_prefix == n]
                b = g[g.N_prefix == 64]
                if len(a) == 0 or len(b) == 0:
                    continue
                if m not in a.columns or not np.isfinite(a.iloc[0][m]) or not np.isfinite(b.iloc[0][m]):
                    continue
                xs.append(float(a.iloc[0][m]))
                ys.append(float(b.iloc[0][m]))
            if len(xs) < 5:
                continue
            xs, ys = np.asarray(xs), np.asarray(ys)
            sp = float(spearmanr(xs, ys).statistic)
            nad = float(np.median(np.abs(xs - ys) / (np.abs(ys) + 1e-8)))
            conv_rows.append({"N": n, "metric": m, "spearman_vs_64": sp, "median_NAD": nad, "n_pairs": len(xs)})
    return df, pd.DataFrame(conv_rows)


def decide_N(conv: pd.DataFrame) -> int:
    # primary rule: smallest N in {16,32} with median spearman>=0.95 and median NAD<=0.10 across metrics
    for n in (16, 32):
        sub = conv[conv.N == n]
        if len(sub) == 0:
            continue
        if float(sub.spearman_vs_64.median()) >= 0.95 and float(sub.median_NAD.median()) <= 0.10:
            return n
    return 64


def combine_vh_vl(vh: dict, vl: dict):
    row = {"id": vh["id"]}
    keys = [
        "ca_rmsd_median",
        "ca_rmsd_q90",
        "ca_rmsf_mean",
        "ca_rmsf_q90",
        "ca_rmsf_max",
        "rg_mean",
        "rg_sd",
        "contact_mean_occ",
        "contact_frac_persistent",
        "contact_frac_labile",
        "contact_occ_entropy",
        "rmsf_cdr_mean",
        "rmsf_fw_mean",
        "rmsf_cdr3_mean",
    ]
    for k in keys:
        row[f"VH_{k}"] = vh.get(k, np.nan)
        row[f"VL_{k}"] = vl.get(k, np.nan)
        a, b = vh.get(k, np.nan), vl.get(k, np.nan)
        row[f"mean_{k}"] = np.nanmean([a, b])
        row[f"max_{k}"] = np.nanmax([a, b]) if np.isfinite(a) or np.isfinite(b) else np.nan
        row[f"absdiff_{k}"] = np.abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
    # keep <=30: select primary set
    primary = [
        "VH_ca_rmsd_median",
        "VL_ca_rmsd_median",
        "mean_ca_rmsd_median",
        "VH_ca_rmsf_mean",
        "VL_ca_rmsf_mean",
        "mean_ca_rmsf_mean",
        "max_ca_rmsf_q90",
        "VH_rg_mean",
        "VL_rg_mean",
        "mean_rg_sd",
        "VH_contact_mean_occ",
        "VL_contact_mean_occ",
        "mean_contact_frac_persistent",
        "mean_contact_frac_labile",
        "mean_contact_occ_entropy",
        "VH_rmsf_cdr_mean",
        "VL_rmsf_cdr_mean",
        "VH_rmsf_fw_mean",
        "VL_rmsf_fw_mean",
        "VH_rmsf_cdr3_mean",
        "VL_rmsf_cdr3_mean",
        "absdiff_ca_rmsf_mean",
        "absdiff_rg_mean",
        "max_ca_rmsd_q90",
        "mean_ca_rmsf_q90",
    ]
    out = {"id": row["id"]}
    for k in primary:
        out[k] = row.get(k, np.nan)
    out["n_features"] = len(primary)
    return out


def main():
    import sys

    if "--convergence" in sys.argv:
        ids = pd.read_csv(V2 / "pilots/BIOEMU_CONVERGENCE_ANTIBODIES.csv").id.tolist()
        raw, conv = convergence_table(ids)
        raw.to_csv(V2 / "cache/bioemu_features/convergence_raw_prefixes.csv", index=False)
        conv.to_csv(V2 / "BIOEMU_SAMPLE_CONVERGENCE.csv", index=False)
        n = decide_N(conv)
        (V2 / "BIOEMU_FROZEN_N.txt").write_text(str(n) + "\n")
        (V2 / "BIOEMU_SAMPLE_COUNT_DECISION.md").write_text(
            f"# BioEmu sample count decision\n\n"
            f"**SPEC_FROZEN_BEFORE_TARGET_SCORING**\n\n"
            f"Selected N = **{n}**\n\n"
            f"Rule: smallest of {{16,32}} with median Spearman vs N=64 ≥ 0.95 and median NAD ≤ 0.10; else 64.\n\n"
            f"See `BIOEMU_SAMPLE_CONVERGENCE.csv`.\n"
        )
        print("FROZEN_N", n)
        print(conv.to_string(index=False))
        return

    # full feature extract at frozen N
    n = int((V2 / "BIOEMU_FROZEN_N.txt").read_text().strip())
    ids = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").id.tolist()
    rows = []
    for ab in ids:
        try:
            tvh = load_traj(SAMPLES / f"{ab}_VH", n_max=n)
            tvl = load_traj(SAMPLES / f"{ab}_VL", n_max=n)
            vh = descriptors(tvh, ab, "VH")
            vl = descriptors(tvl, ab, "VL")
            vh["id"] = ab
            vl["id"] = ab
            rows.append(combine_vh_vl(vh, vl))
        except Exception as e:
            rows.append({"id": ab, "extraction_status": f"FAIL:{type(e).__name__}:{e}"})
    df = pd.DataFrame(rows)
    df.to_csv(V2 / "BIOEMU_V12_FEATURES.csv", index=False)
    print("wrote features", len(df))


if __name__ == "__main__":
    main()
