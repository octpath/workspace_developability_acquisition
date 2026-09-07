#!/usr/bin/env python3
"""Physical-frame convergence, MC noise, filter bias, Nphys freeze, features."""
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
R = FP / "bioemu_isolated_reassessment"
V2 = FP / "foundation_stability_v2"
SAMPLES = R / "cache/bioemu_samples"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"
SPEC = json.loads((R / "BIOEMU_ISOLATED_REASSESS_SPEC.json").read_text())
SEED = int(SPEC["convergence"]["seed"])
N_MC = int(SPEC["convergence"]["mc_subsets"])
NPHYS_LIST = [4, 8, 16, 32, 64]


def load_traj(d: Path, n_max: int | None = None):
    traj = md.load(str(d / "samples.xtc"), top=str(d / "topology.pdb"))
    if n_max is not None:
        traj = traj[:n_max]
    return traj


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


def contact_stats(ca, min_sep=4, cutoff_nm=0.8):
    n = ca.n_atoms
    pairs = np.array([[i, j] for i in range(n) for j in range(i + min_sep, n)], dtype=int)
    if len(pairs) == 0:
        return dict(mean_occ=np.nan, frac_persist=np.nan, frac_labile=np.nan, entropy=np.nan)
    occ = (md.compute_distances(ca, pairs) < cutoff_nm).mean(axis=0)
    p = np.clip(occ, 1e-6, 1 - 1e-6)
    return dict(
        mean_occ=float(occ.mean()),
        frac_persist=float(np.mean(occ >= 0.8)),
        frac_labile=float(np.mean((occ > 0.2) & (occ < 0.8))),
        entropy=float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)))),
    )


def region_rmsf(rmsf, ab_id, chain_label):
    cdr = pd.read_csv(CDR_IDX)
    ch = "H" if chain_label == "VH" else "L"
    sub = cdr[(cdr.id == ab_id) & (cdr.chain.astype(str) == ch)].sort_values("sequence_index")
    out = dict(rmsf_cdr=np.nan, rmsf_fw=np.nan, rmsf_cdr3=np.nan)
    if len(sub) == 0 or len(sub) != len(rmsf):
        return out
    is_cdr = sub.region.astype(str).str.upper().str.contains("CDR")
    is_cdr3 = sub.region.astype(str).str.upper().str.contains("CDR3")
    if is_cdr.any():
        out["rmsf_cdr"] = float(rmsf[is_cdr.values].mean())
    if (~is_cdr).any():
        out["rmsf_fw"] = float(rmsf[~is_cdr.values].mean())
    if is_cdr3.any():
        out["rmsf_cdr3"] = float(rmsf[is_cdr3.values].mean())
    return out


def descriptors(traj, ab_id: str, chain: str) -> dict:
    ca = traj.atom_slice(traj.topology.select("name CA"))
    ca.superpose(ca, 0)
    rmsf = md.rmsf(ca, ca, 0)
    rg = md.compute_rg(ca)
    med, q90 = pairwise_ca_rmsd(ca)
    cst = contact_stats(ca)
    reg = region_rmsf(rmsf, ab_id, chain)
    return {
        "n_physical": int(traj.n_frames),
        "ca_rmsd_median": med,
        "ca_rmsd_q90": q90,
        "ca_rmsf_mean": float(rmsf.mean()),
        "ca_rmsf_q90": float(np.quantile(rmsf, 0.9)),
        "rg_mean": float(rg.mean()),
        "rg_sd": float(rg.std()),
        "contact_mean_occ": cst["mean_occ"],
        "contact_frac_persistent": cst["frac_persist"],
        "contact_frac_labile": cst["frac_labile"],
        "contact_occ_entropy": cst["entropy"],
        "rmsf_cdr_mean": reg["rmsf_cdr"],
        "rmsf_fw_mean": reg["rmsf_fw"],
        "rmsf_cdr3_mean": reg["rmsf_cdr3"],
    }


DESC_KEYS = [
    "ca_rmsd_median",
    "ca_rmsd_q90",
    "ca_rmsf_mean",
    "ca_rmsf_q90",
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


def subset_indices(n_total: int, n_take: int, seed: int, draw: int) -> np.ndarray:
    rng = np.random.default_rng(seed + draw * 100003 + n_take)
    if n_take >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=n_take, replace=False))


def run_convergence(ids: list[str]):
    """MC subset convergence for pilot chains with >=64 physical frames."""
    conv_rows = []
    mc_rows = []
    for ab in ids:
        for ch in ("VH", "VL"):
            d = SAMPLES / f"{ab}_{ch}"
            if not (d / "samples.xtc").exists():
                continue
            full = load_traj(d)
            if full.n_frames < 64:
                print("skip <64", ab, ch, full.n_frames, flush=True)
                continue
            # deterministic first 64 physical as reference pool
            pool = full[:64]
            ref = descriptors(pool, ab, ch)
            for nphys in NPHYS_LIST:
                vals = {k: [] for k in DESC_KEYS}
                for draw in range(1 if nphys == 64 else N_MC):
                    idx = subset_indices(64, nphys, SEED + hash(f"{ab}_{ch}") % 10000, draw)
                    desc = descriptors(pool[idx], ab, ch)
                    for k in DESC_KEYS:
                        vals[k].append(desc[k])
                    if nphys < 64:
                        mc_rows.append(
                            {
                                "id": ab,
                                "chain": ch,
                                "Nphys": nphys,
                                "draw": draw,
                                **{k: desc[k] for k in DESC_KEYS},
                            }
                        )
                for k in DESC_KEYS:
                    arr = np.asarray(vals[k], float)
                    ref_v = ref[k]
                    # across draws: median estimate, then vs ref
                    est = float(np.nanmedian(arr))
                    nad = abs(est - ref_v) / (abs(ref_v) + 1e-8) if np.isfinite(ref_v) else np.nan
                    mc_sd = float(np.nanstd(arr)) if nphys < 64 else 0.0
                    conv_rows.append(
                        {
                            "id": ab,
                            "chain": ch,
                            "Nphys": nphys,
                            "metric": k,
                            "estimate": est,
                            "ref64": ref_v,
                            "NAD": nad,
                            "mc_sd": mc_sd,
                        }
                    )
            print("conv done", ab, ch, flush=True)

    cdf = pd.DataFrame(conv_rows)
    mdf = pd.DataFrame(mc_rows)
    # Spearman across Abs for each chain, metric, Nphys vs ref64
    summ = []
    noise = []
    for ch in ("VH", "VL"):
        for metric in DESC_KEYS:
            for nphys in NPHYS_LIST:
                sub = cdf[(cdf.chain == ch) & (cdf.metric == metric) & (cdf.Nphys == nphys)]
                if len(sub) < 5:
                    continue
                sp = spearmanr(sub.estimate, sub.ref64).statistic if nphys < 64 else 1.0
                summ.append(
                    {
                        "chain": ch,
                        "Nphys": nphys,
                        "metric": metric,
                        "spearman_vs_64": float(sp) if np.isfinite(sp) else np.nan,
                        "median_NAD": float(sub.NAD.median()),
                        "n_abs": len(sub),
                    }
                )
                # MC noise ratio
                if nphys < 64 and len(mdf):
                    # within-ab MC sd median / between-ab sd of estimates
                    msub = mdf[(mdf.chain == ch) & (mdf.Nphys == nphys)]
                    if metric not in msub.columns or len(msub) == 0:
                        continue
                    within = msub.groupby("id")[metric].std()
                    between = msub.groupby("id")[metric].mean().std()
                    ratio = float(within.median() / (between + 1e-12)) if between == between else np.nan
                    noise.append(
                        {
                            "chain": ch,
                            "Nphys": nphys,
                            "metric": metric,
                            "median_within_mc_sd": float(within.median()),
                            "between_ab_sd": float(between) if np.isfinite(between) else np.nan,
                            "MC_noise_ratio": ratio,
                        }
                    )

    pd.DataFrame(summ).to_csv(R / "BIOEMU_ISOLATED_PHYSICAL_CONVERGENCE.csv", index=False)
    pd.DataFrame(noise).to_csv(R / "BIOEMU_ISOLATED_MC_NOISE.csv", index=False)
    cdf.to_csv(R / "results/CONVERGENCE_RAW.csv", index=False)

    # decide Nphys per chain on major families
    major = [
        "ca_rmsd_median",
        "ca_rmsf_mean",
        "rg_mean",
        "contact_mean_occ",
        "contact_frac_persistent",
        "contact_occ_entropy",
        "rmsf_cdr_mean",
        "rmsf_cdr3_mean",
    ]
    decisions = {}
    lines = ["# BioEmu isolated PHYSICAL-frame N decision", "", "**SPEC_FROZEN_BEFORE_TARGET_SCORING**", ""]
    sdf = pd.DataFrame(summ)
    for ch in ("VH", "VL"):
        chosen = 64
        for nphys in (8, 16, 32):
            t = sdf[(sdf.chain == ch) & (sdf.Nphys == nphys) & (sdf.metric.isin(major))]
            if len(t) == 0:
                continue
            if float(t.spearman_vs_64.median()) >= 0.95 and float(t.median_NAD.median()) <= 0.10:
                chosen = nphys
                break
        decisions[ch] = chosen
        lines.append(f"- **{ch}**: Nphys = **{chosen}**")
        # noise at ~5 and chosen
        ndf = pd.DataFrame(noise)
        for nphys in (4, 8, 16, 32):
            nt = ndf[(ndf.chain == ch) & (ndf.Nphys == nphys) & (ndf.metric.isin(major))]
            if len(nt):
                lines.append(
                    f"  - Nphys={nphys}: median MC_noise_ratio (major) = {nt.MC_noise_ratio.median():.3f}"
                )
    # freeze single N for both = max for simplicity / cohort
    n_freeze = max(decisions.values())
    (R / "BIOEMU_ISOLATED_FROZEN_NPHYS.txt").write_text(str(n_freeze) + "\n")
    (R / "BIOEMU_ISOLATED_FROZEN_NPHYS_VH.txt").write_text(str(decisions["VH"]) + "\n")
    (R / "BIOEMU_ISOLATED_FROZEN_NPHYS_VL.txt").write_text(str(decisions["VL"]) + "\n")
    lines += [
        "",
        f"**Cohort frozen Nphys (max of VH/VL) = {n_freeze}**",
        "",
        "Rule: smallest of {8,16,32} with median Spearman vs N64 ≥ 0.95 and median NAD ≤ 0.10 on major families; else 64.",
        "",
        "Seeds: deterministic MC subsets (20 draws) from first 64 physical frames; seed in SPEC.",
        "",
        "See `BIOEMU_ISOLATED_PHYSICAL_CONVERGENCE.csv`, `BIOEMU_ISOLATED_MC_NOISE.csv`.",
    ]
    if n_freeze >= 32:
        # check if even 32 inadequate
        weak = False
        for ch, n in decisions.items():
            if n >= 64:
                weak = True
        if weak:
            lines.append("")
            lines.append("Classification note: may be `BIOEMU_ISOLATED_ENSEMBLE_HIGH_VARIANCE` if Nphys=64 required.")
    (R / "BIOEMU_ISOLATED_NPHYS_DECISION.md").write_text("\n".join(lines) + "\n")
    print("decisions", decisions, "cohort", n_freeze)
    return decisions, n_freeze


def filter_bias(ids: list[str]):
    """RAW vs PASS on pilot using v2/reassess NPZ rebuild if needed — use existing phys vs metrics on filtered only safe pre-filter from raw rebuild."""
    from bioemu.convert_chemgraph import save_pdb_and_xtc, _filter_unphysical_traj_masks
    import torch

    rows = []
    for ab in ids:
        for ch in ("VH", "VL"):
            d = SAMPLES / f"{ab}_{ch}"
            files = sorted(d.glob("batch_*.npz"))
            if not files:
                continue
            seq = str(np.load(files[0])["sequence"].item())
            pos = torch.tensor(np.concatenate([np.load(f)["pos"] for f in files]))
            ori = torch.tensor(np.concatenate([np.load(f)["node_orientations"] for f in files]))
            # cap frames for speed
            if pos.shape[0] > 128:
                pos, ori = pos[:128], ori[:128]
            tmp = R / "cache/filter_bias_tmp" / f"{ab}_{ch}"
            tmp.mkdir(parents=True, exist_ok=True)
            top, xtc = tmp / "raw.pdb", tmp / "raw.xtc"
            save_pdb_and_xtc(pos, ori, seq, top, xtc, filter_samples=False)
            raw = md.load(str(xtc), top=str(top))
            ca_ok, cn_ok, noclash = _filter_unphysical_traj_masks(raw)
            passed = ca_ok & cn_ok & noclash
            ca = raw.atom_slice(raw.topology.select("name CA"))
            rg = md.compute_rg(ca)
            # max CA gap
            pairs = np.array([(i, i + 1) for i in range(ca.n_atoms - 1)])
            cad, _ = md.compute_contacts(ca, scheme="ca", contacts=pairs, periodic=False)
            cad = md.utils.in_units_of(cad, "nanometers", "angstrom")
            max_gap = cad.max(axis=1)
            for flag, name in [(True, "PASS"), (False, "FAIL")]:
                m = passed if flag else ~passed
                if m.sum() == 0:
                    continue
                rows.append(
                    {
                        "id": ab,
                        "chain": ch,
                        "subset": name,
                        "n": int(m.sum()),
                        "rg_mean": float(rg[m].mean()),
                        "rg_sd": float(rg[m].std()),
                        "max_ca_gap_mean": float(max_gap[m].mean()),
                        "frac_ca_break": float((max_gap[m] >= 4.5).mean()),
                    }
                )
            print("bias", ab, ch, "pass", int(passed.sum()), "/", len(passed), flush=True)
    pd.DataFrame(rows).to_csv(R / "BIOEMU_ISOLATED_FILTER_BIAS.csv", index=False)


def build_features(ids: list[str] | None, n_phys: int):
    """Build cohort features from deterministic first n_phys physical frames."""
    if ids is None:
        ids = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").id.tolist()
    rows = []
    for ab in ids:
        try:
            tvh = load_traj(SAMPLES / f"{ab}_VH", n_max=n_phys)
            tvl = load_traj(SAMPLES / f"{ab}_VL", n_max=n_phys)
        except Exception as e:
            print("feat skip", ab, e, flush=True)
            continue
        if tvh.n_frames < max(4, n_phys // 2) or tvl.n_frames < max(4, n_phys // 2):
            print("feat insufficient", ab, tvh.n_frames, tvl.n_frames, flush=True)
            continue
        dh, dl = descriptors(tvh, ab, "VH"), descriptors(tvl, ab, "VL")
        row = {"id": ab, "nphys_used": n_phys, "VH_n": dh["n_physical"], "VL_n": dl["n_physical"]}
        for k in DESC_KEYS:
            row[f"VH_{k}"] = dh[k]
            row[f"VL_{k}"] = dl[k]
            row[f"mean_{k}"] = np.nanmean([dh[k], dl[k]])
            row[f"max_{k}"] = np.nanmax([dh[k], dl[k]])
            row[f"absdiff_{k}"] = abs(dh[k] - dl[k]) if np.isfinite(dh[k]) and np.isfinite(dl[k]) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(R / "BIOEMU_ISOLATED_REASSESS_FEATURES.csv", index=False)
    print("features", len(df))
    return df


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "convergence"
    ids = pd.read_csv(R / "pilots/BIOEMU_ISOLATED_REASSESS_PILOT.csv").id.tolist()
    if cmd == "convergence":
        run_convergence(ids)
    elif cmd == "bias":
        filter_bias(ids)
    elif cmd == "features":
        n = int((R / "BIOEMU_ISOLATED_FROZEN_NPHYS.txt").read_text().strip())
        ids_all = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").id.tolist()
        build_features(ids_all, n)
    elif cmd == "pilot_features":
        n = int((R / "BIOEMU_ISOLATED_FROZEN_NPHYS.txt").read_text().strip())
        build_features(ids, n)
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
