#!/usr/bin/env python3
"""12-Fab ESMFold pilot + optional full cohort (resumable). Uses GPU1 by default to coexist with BioEmu on GPU0."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
SEQ = OUT / "sequences"
STR = OUT / "structures"
QC = OUT / "qc"
LOGS = OUT / "logs"
PILOT = STR / "pilot"
FULL = STR / "esmfold_fab"

# Coexist with BioEmu on physical GPU0
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Default GPU0 (RTX 3090). GPU1 (1080 Ti) is incompatible with current PyTorch CUDA arches.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FAB_ESMFOLD_GPU", "0")

AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def parse_pdb_ca(pdb_text: str):
    seqs, coords, bfactors, resseqs = {}, {}, {}, {}
    seen = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        resseq = int(line[22:26])
        key = (ch, resseq)
        if key in seen:
            continue
        seen.add(key)
        aa = AA3.get(line[17:20].strip(), "X")
        seqs.setdefault(ch, []).append(aa)
        coords.setdefault(ch, []).append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        bfactors.setdefault(ch, []).append(float(line[60:66]))
        resseqs.setdefault(ch, []).append(resseq)
    return (
        {c: "".join(s) for c, s in seqs.items()},
        {c: np.array(v) for c, v in coords.items()},
        {c: np.array(v) for c, v in bfactors.items()},
        resseqs,
    )


def parse_pdb_atoms(pdb_text: str):
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        el = line[76:78].strip() or line[12:16].strip()[0]
        if el == "H":
            continue
        atoms.append(
            {
                "ch": line[21],
                "resseq": int(line[22:26]),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                "el": el,
            }
        )
    return atoms


def domain_slices(vh_len, ch1_len, vl_len, cl_len):
    return {
        "VH": (0, vh_len),
        "CH1": (vh_len, vh_len + ch1_len),
        "VL": (0, vl_len),
        "CL": (vl_len, vl_len + cl_len),
    }


def mean_plddt(bf, i0, i1):
    if len(bf) < i1:
        return float("nan")
    return float(np.mean(bf[i0:i1]))


def interface_contacts(ca_a, ca_b, cutoff=8.0):
    if len(ca_a) == 0 or len(ca_b) == 0:
        return 0
    d = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
    return int((d < cutoff).sum())


def severe_clashes(atoms, cutoff=1.5):
    # sample pairwise heavy atoms excluding same/adjacent residue
    coords = np.array([a["xyz"] for a in atoms])
    meta = atoms
    n = len(coords)
    severe = 0
    # O(n^2) may be heavy for Fab (~3500 atoms); use grid or subsample CA+CB only for speed
    # Use all heavy but skip if n large via stride? Prefer CA-only clash proxy + sidechain S
    idx = [i for i, a in enumerate(meta) if a["name"] in ("CA", "CB", "N", "C", "O", "SG")]
    for ii in range(len(idx)):
        i = idx[ii]
        for jj in range(ii + 1, len(idx)):
            j = idx[jj]
            if meta[i]["ch"] == meta[j]["ch"] and abs(meta[i]["resseq"] - meta[j]["resseq"]) <= 1:
                continue
            if np.linalg.norm(coords[i] - coords[j]) < cutoff:
                severe += 1
    return severe


def cys_sg_coords(atoms, chain, resseq):
    for a in atoms:
        if a["ch"] == chain and a["resseq"] == resseq and a["name"] == "SG":
            return a["xyz"]
    return None


def hl_disulfide_distance(atoms, heavy_len, light_len, light_locus, ch1_len):
    """Approximate: last Cys on light; on heavy use last Cys (EPKSC) and CH1 Cys candidates."""
    # find Cys residue numbers per chain from SG
    h_cys, l_cys = [], []
    for a in atoms:
        if a["name"] != "SG":
            continue
        if a["ch"] == "A":
            h_cys.append(a["resseq"])
        elif a["ch"] == "B":
            l_cys.append(a["resseq"])
    if not h_cys or not l_cys:
        return float("nan"), ""
    l_term = max(l_cys)
    # heavy candidates: max (hinge) and any in CH1 region
    h_hinge = max(h_cys)
    dists = []
    notes = []
    for hc in sorted(set(h_cys)):
        sa = cys_sg_coords(atoms, "A", hc)
        sb = cys_sg_coords(atoms, "B", l_term)
        if sa is None or sb is None:
            continue
        d = float(np.linalg.norm(sa - sb))
        dists.append((d, hc, l_term))
    if not dists:
        return float("nan"), "no_sg"
    dists.sort()
    best = dists[0]
    note = f"min_SG_dist={best[0]:.2f}_H{best[1]}_L{best[2]};locus={light_locus}"
    return best[0], note


def select_pilot(fab: pd.DataFrame, n=12) -> list[str]:
    """Target-blind stratified by locus and length quartiles + germline diversity."""
    rng = np.random.default_rng(42)
    picks = []
    for locus in ("kappa", "lambda"):
        sub = fab[fab["light_locus"] == locus].copy()
        sub["vhq"] = pd.qcut(sub["VH_len_used"], 2, labels=False, duplicates="drop")
        sub["vlq"] = pd.qcut(sub["VL_len_used"], 2, labels=False, duplicates="drop")
        # take diverse germline if available
        for _, g in sub.groupby(["vhq", "vlq"], dropna=False):
            if len(picks) >= n:
                break
            # prefer unique V genes
            g = g.drop_duplicates(subset=["heavy_V_gene", "light_V_gene"])
            if len(g) == 0:
                continue
            row = g.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            picks.append(row["id"])
        # fill
        while len([p for p in picks if fab.set_index("id").loc[p, "light_locus"] == locus]) < n // 2:
            cand = sub[~sub["id"].isin(picks)]
            if cand.empty:
                break
            picks.append(cand.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["id"])
    # ensure extremes
    for col, which in [("VH_len_used", "idxmin"), ("VH_len_used", "idxmax"), ("VL_len_used", "idxmin"), ("VL_len_used", "idxmax")]:
        idx = getattr(fab[col], which)()
        aid = fab.loc[idx, "id"]
        if aid not in picks:
            picks.append(aid)
    # trim/pad to exactly 12
    picks = list(dict.fromkeys(picks))
    if len(picks) > n:
        picks = picks[:n]
    while len(picks) < n:
        extra = fab[~fab["id"].isin(picks)].sample(1, random_state=0).iloc[0]["id"]
        picks.append(extra)
    return picks[:n]


def qc_one(pdb_text: str, row: pd.Series) -> dict:
    seqs, coords, bf, _ = parse_pdb_ca(pdb_text)
    atoms = parse_pdb_atoms(pdb_text)
    chains = sorted(seqs.keys())
    out = {
        "id": row["id"],
        "n_chains": len(chains),
        "chain_ids": "".join(chains),
        "both_chains": len(chains) >= 2,
    }
    if len(chains) < 2:
        out["success"] = False
        out["notes"] = "missing_chain"
        return out
    # map by length
    c0, c1 = chains[0], chains[1]
    if len(seqs[c0]) == row["heavy_length"] and len(seqs[c1]) == row["light_length"]:
        H, L = c0, c1
    elif len(seqs[c1]) == row["heavy_length"] and len(seqs[c0]) == row["light_length"]:
        H, L = c1, c0
    else:
        out["success"] = False
        out["notes"] = f"length_mismatch H={len(seqs[c0])}/{len(seqs[c1])} vs {row['heavy_length']}/{row['light_length']}"
        return out
    # remapped assume A=H B=L for distance helpers: rewrite if needed
    if H != "A" or L != "B":
        # still OK for metrics using H/L keys
        pass
    vh, vl = row["VH_len_used"], row["VL_len_used"]
    ch1 = row["heavy_length"] - vh
    cl = row["light_length"] - vl
    out.update(
        {
            "success": True,
            "length_match": True,
            "global_pLDDT": float(np.mean(np.concatenate([bf[H], bf[L]]))),
            "VH_pLDDT": mean_plddt(bf[H], 0, vh),
            "CH1_pLDDT": mean_plddt(bf[H], vh, vh + ch1),
            "VL_pLDDT": mean_plddt(bf[L], 0, vl),
            "CL_pLDDT": mean_plddt(bf[L], vl, vl + cl),
            "VH_VL_contacts_8A": interface_contacts(coords[H][:vh], coords[L][:vl]),
            "CH1_CL_contacts_8A": interface_contacts(coords[H][vh:], coords[L][vl:]),
            "com_distance_A": float(np.linalg.norm(coords[H].mean(0) - coords[L].mean(0))),
            "Fv_const_com_distance_A": float(
                np.linalg.norm(
                    np.concatenate([coords[H][:vh], coords[L][:vl]]).mean(0)
                    - np.concatenate([coords[H][vh:], coords[L][vl:]]).mean(0)
                )
            ),
        }
    )
    # remap atoms chain labels if H/L are not A/B — hl_disulfide expects A/B as heavy/light
    atoms2 = []
    for a in atoms:
        aa = dict(a)
        if a["ch"] == H:
            aa["ch"] = "A"
        elif a["ch"] == L:
            aa["ch"] = "B"
        atoms2.append(aa)
    dss, note = hl_disulfide_distance(atoms2, row["heavy_length"], row["light_length"], row["light_locus"], ch1)
    out["heavy_light_disulfide_distance"] = dss
    out["disulfide_note"] = note
    out["severe_clash_count"] = severe_clashes(atoms2)
    # architecture flags
    flags = []
    if out["CH1_CL_contacts_8A"] < 5:
        flags.append("weak_CH1_CL_interface")
    if out["VH_VL_contacts_8A"] < 20:
        flags.append("weak_VH_VL_interface")
    if out["com_distance_A"] > 60:
        flags.append("chains_separated")
    if out["Fv_const_com_distance_A"] > 50:
        flags.append("Fv_const_detached")
    out["architecture_flags"] = ";".join(flags)
    out["notes"] = ""
    return out


def load_model(chunk_size: int):
    import esm

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    if hasattr(model, "set_chunk_size"):
        model.set_chunk_size(chunk_size)
    return model, esm


def predict(model, heavy: str, light: str, num_recycles=None):
    seq = f"{heavy}:{light}"
    with torch.no_grad():
        pdb = model.infer_pdb(seq) if num_recycles is None else model.infer_pdb(seq)  # infer_pdb uses default recycles
        # fair-esm infer_pdb doesn't pass num_recycles; use infer + output_to_pdb if needed
    return pdb


def predict_robust(model, heavy: str, light: str):
    seq = f"{heavy}:{light}"
    with torch.no_grad():
        if hasattr(model, "infer_pdb"):
            try:
                return model.infer_pdb(seq)
            except Exception:
                out = model.infer(seq)
                return model.output_to_pdb(out)[0]
        out = model.infer(seq)
        return model.output_to_pdb(out)[0]


def run_batch(ids: list[str], fab: pd.DataFrame, out_dir: Path, tag: str, chunk_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    model, esm = load_model(chunk_size)
    free, total = torch.cuda.mem_get_info(0)
    cfg = {
        "package": "facebookresearch/esm (fair-esm)",
        "esm_version": getattr(esm, "__version__", None),
        "model": "esmfold_v1",
        "input_format": "HEAVY_FAB:LIGHT_FAB",
        "api": "model.infer_pdb(sequence)",
        "chunk_size": chunk_size,
        "num_recycles": "default_training_max_4",
        "residue_index_offset": 512,
        "chain_linker": "G*25_internal",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(total / 1024**2, 1),
        "vram_free_after_load_mb": round(free / 1024**2, 1),
        "note": "Reuses gate_b2 ESMFold native multimer config; GPU1 default to coexist with BioEmu",
    }
    (STR / "ESMFOLD_FAB_CONFIG.json").write_text(json.dumps(cfg, indent=2))

    rows = []
    for aid in ids:
        row = fab[fab["id"] == aid].iloc[0]
        pdb_path = out_dir / f"{aid}.pdb"
        t0 = time.perf_counter()
        if pdb_path.exists() and pdb_path.stat().st_size > 1000:
            pdb = pdb_path.read_text()
            runtime = 0.0
            status = "cached"
        else:
            try:
                pdb = predict_robust(model, row["heavy_fab_seq"], row["light_fab_seq"])
                pdb_path.write_text(pdb)
                runtime = time.perf_counter() - t0
                status = "ok"
            except Exception as e:
                # retry with smaller chunk
                try:
                    if hasattr(model, "set_chunk_size"):
                        model.set_chunk_size(max(32, chunk_size // 2))
                    pdb = predict_robust(model, row["heavy_fab_seq"], row["light_fab_seq"])
                    pdb_path.write_text(pdb)
                    runtime = time.perf_counter() - t0
                    status = "ok_retry_chunk"
                except Exception as e2:
                    rows.append(
                        {
                            "id": aid,
                            "status": "fail",
                            "runtime": time.perf_counter() - t0,
                            "notes": f"{e} | retry: {e2}",
                            "file_path": "",
                        }
                    )
                    print("FAIL", aid, e2, flush=True)
                    continue
        q = qc_one(pdb, row)
        q.update(
            {
                "status": status if q.get("success") else "qc_fail",
                "runtime": runtime,
                "heavy_length": row["heavy_length"],
                "light_length": row["light_length"],
                "file_path": str(pdb_path),
                "SHA256": sha256_file(pdb_path),
                "light_locus": row["light_locus"],
            }
        )
        rows.append(q)
        print(tag, aid, q.get("status"), "pLDDT", round(q.get("global_pLDDT", -1), 1), "CH1CL", q.get("CH1_CL_contacts_8A"), flush=True)
    return pd.DataFrame(rows)


def pilot_verdict(df: pd.DataFrame) -> str:
    ok = df[df["status"].astype(str).str.startswith("ok") | (df["status"] == "cached")]
    ok = ok[ok.get("success", True) == True] if "success" in ok.columns else ok
    n = len(df)
    n_ok = int(ok["success"].sum()) if "success" in df.columns else len(ok)
    if n_ok < n * 0.5:
        return "FAIL_MULTIMER_GEOMETRY"
    weak = 0
    for _, r in df.iterrows():
        flags = str(r.get("architecture_flags") or "")
        if "chains_separated" in flags or "Fv_const_detached" in flags or "weak_CH1_CL_interface" in flags:
            weak += 1
    if weak >= max(3, n // 2):
        return "FAIL_MULTIMER_GEOMETRY"
    if weak > 0 or (ok["CH1_pLDDT"].mean() < 60) or (ok["CL_pLDDT"].mean() < 60):
        return "PASS_WITH_CAUTION"
    return "PASS_FULL_COHORT"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    fab = pd.read_csv(SEQ / "SHEHATA_RECONSTRUCTED_FAB.csv")
    chunk = int(os.environ.get("FAB_ESMFOLD_CHUNK", "48"))  # conserve VRAM vs BioEmu on GPU0

    if mode == "pilot":
        picks = select_pilot(fab, 12)
        (STR / "pilot_ids.json").write_text(json.dumps(picks, indent=2))
        print("PILOT ids", picks, flush=True)
        df = run_batch(picks, fab, PILOT, "PILOT", chunk)
        df.to_csv(QC / "ESMFOLD_FAB_PILOT_QC.csv", index=False)
        verd = pilot_verdict(df) if "global_pLDDT" in df.columns and df["global_pLDDT"].notna().any() else "FAIL_MULTIMER_GEOMETRY"
        ok = df[df.get("success", False) == True] if "success" in df.columns else df.iloc[0:0]
        if len(ok) == 0:
            report = f"""# ESMFOLD FAB PILOT REPORT

**Verdict:** `{verd}`  
**N:** {len(df)}  
**Success:** 0

All predictions failed. See `ESMFOLD_FAB_PILOT_QC.csv` / logs.

```
{df[['id','status','notes']].to_string(index=False) if 'status' in df.columns else df.head().to_string()}
```
"""
        else:
            report = f"""# ESMFOLD FAB PILOT REPORT

**Verdict:** `{verd}`  
**N:** {len(df)}  
**Success (length-matched):** {int(ok['success'].sum())}

## Summary stats (successful)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| global_pLDDT | {ok['global_pLDDT'].mean():.1f} | {ok['global_pLDDT'].min():.1f} | {ok['global_pLDDT'].max():.1f} |
| VH_pLDDT | {ok['VH_pLDDT'].mean():.1f} | {ok['VH_pLDDT'].min():.1f} | {ok['VH_pLDDT'].max():.1f} |
| VL_pLDDT | {ok['VL_pLDDT'].mean():.1f} | {ok['VL_pLDDT'].min():.1f} | {ok['VL_pLDDT'].max():.1f} |
| CH1_pLDDT | {ok['CH1_pLDDT'].mean():.1f} | {ok['CH1_pLDDT'].min():.1f} | {ok['CH1_pLDDT'].max():.1f} |
| CL_pLDDT | {ok['CL_pLDDT'].mean():.1f} | {ok['CL_pLDDT'].min():.1f} | {ok['CL_pLDDT'].max():.1f} |
| CH1_CL_contacts_8A | {ok['CH1_CL_contacts_8A'].mean():.1f} | {ok['CH1_CL_contacts_8A'].min()} | {ok['CH1_CL_contacts_8A'].max()} |
| VH_VL_contacts_8A | {ok['VH_VL_contacts_8A'].mean():.1f} | {ok['VH_VL_contacts_8A'].min()} | {ok['VH_VL_contacts_8A'].max()} |
| H-L SS distance (Å) | {ok['heavy_light_disulfide_distance'].mean():.2f} | {ok['heavy_light_disulfide_distance'].min():.2f} | {ok['heavy_light_disulfide_distance'].max():.2f} |
| severe_clash_count | {ok['severe_clash_count'].mean():.1f} | {ok['severe_clash_count'].min()} | {ok['severe_clash_count'].max()} |

## Architecture flags

{ok[['id','architecture_flags','com_distance_A','Fv_const_com_distance_A']].to_string(index=False)}

## Interpretation

- Structures are **reconstructed experimental-like Fabs**, not exact Shehata experimental sequences.
- ESMFold does not enforce disulfides; H–L SG distances are QC only.
- Verdict rule is target-blind (geometry/confidence only).
"""
        (QC / "ESMFOLD_FAB_PILOT_REPORT.md").write_text(report)
        (STR / "PILOT_VERDICT.txt").write_text(verd + "\n")
        print("VERDICT", verd, flush=True)
        return

    if mode == "full":
        verd = (STR / "PILOT_VERDICT.txt").read_text().strip() if (STR / "PILOT_VERDICT.txt").exists() else "UNKNOWN"
        if verd == "FAIL_MULTIMER_GEOMETRY":
            print("STOP full: pilot failed")
            return
        ids = fab["id"].tolist()
        df = run_batch(ids, fab, FULL, "FULL", chunk)
        # merge pilot columns for manifest
        cols = [
            "id", "status", "runtime", "heavy_length", "light_length", "global_pLDDT",
            "VH_pLDDT", "VL_pLDDT", "CH1_pLDDT", "CL_pLDDT", "severe_clash_count",
            "heavy_light_disulfide_distance", "file_path", "SHA256", "notes",
            "architecture_flags", "CH1_CL_contacts_8A", "VH_VL_contacts_8A",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df[cols].to_csv(STR / "ESMFOLD_FAB_STRUCTURE_MANIFEST.csv", index=False)
        df.to_csv(QC / "ESMFOLD_FAB_FULL_QC.csv", index=False)
        ok = df[df["success"] == True] if "success" in df.columns else df
        summary = f"""# ESMFOLD FAB FULL QC SUMMARY

Pilot verdict: `{verd}`  
Completion: {len(ok)}/{len(df)} successful length-matched structures.

| Metric | Mean | Median | P10 | P90 |
|--------|------|--------|-----|-----|
| global_pLDDT | {ok['global_pLDDT'].mean():.1f} | {ok['global_pLDDT'].median():.1f} | {ok['global_pLDDT'].quantile(0.1):.1f} | {ok['global_pLDDT'].quantile(0.9):.1f} |
| CH1_pLDDT | {ok['CH1_pLDDT'].mean():.1f} | {ok['CH1_pLDDT'].median():.1f} | {ok['CH1_pLDDT'].quantile(0.1):.1f} | {ok['CH1_pLDDT'].quantile(0.9):.1f} |
| CL_pLDDT | {ok['CL_pLDDT'].mean():.1f} | {ok['CL_pLDDT'].median():.1f} | {ok['CL_pLDDT'].quantile(0.1):.1f} | {ok['CL_pLDDT'].quantile(0.9):.1f} |
| CH1_CL_contacts | {ok['CH1_CL_contacts_8A'].mean():.1f} | {ok['CH1_CL_contacts_8A'].median():.1f} | {ok['CH1_CL_contacts_8A'].quantile(0.1):.1f} | {ok['CH1_CL_contacts_8A'].quantile(0.9):.1f} |
| H-L SS Å | {ok['heavy_light_disulfide_distance'].mean():.2f} | {ok['heavy_light_disulfide_distance'].median():.2f} | {ok['heavy_light_disulfide_distance'].quantile(0.1):.2f} | {ok['heavy_light_disulfide_distance'].quantile(0.9):.2f} |
| severe clashes | {ok['severe_clash_count'].mean():.1f} | {ok['severe_clash_count'].median():.1f} | {ok['severe_clash_count'].quantile(0.1):.1f} | {ok['severe_clash_count'].quantile(0.9):.1f} |
"""
        (QC / "ESMFOLD_FAB_FULL_QC_SUMMARY.md").write_text(summary)
        print("FULL done", len(ok), "/", len(df))
        return

    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
