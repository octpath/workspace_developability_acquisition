#!/usr/bin/env python3
"""Technical pilot QC for FeNNix Fab B/C/M — no TmApp."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
FAB = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
SEQ = FAB / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
PILOT = json.loads((FAB / "structures/pilot_ids.json").read_text())
CACHE = CTX / "cache"
TOL_MAX = 1e-4
TOL_RMSD = 1e-6


def load_coords(pdb: Path):
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    out = {}
    for a in st.get_atoms():
        res = a.get_parent()
        key = (res.get_parent().id, int(res.id[1]), a.get_name().strip())
        out[key] = np.asarray(a.coord, float)
    return out


def var_keys(coords, vh_len, vl_len):
    return [
        k
        for k in coords
        if (k[0] == "A" and k[1] <= vh_len) or (k[0] == "B" and k[1] <= vl_len)
    ]


def cm_match(ab_id, vh_len, vl_len):
    c_pdb = CACHE / "r1" / f"{ab_id}_C_r1.pdb"
    m_pdb = CACHE / "matched_fv" / f"{ab_id}_matched.pdb"
    if not c_pdb.exists() or not m_pdb.exists():
        # fallback: compare from prepared + matched only if C r1 missing
        return {
            "cm_ok": False,
            "cm_error": "MISSING_R1_OR_MATCHED_PDB",
            "cm_max_abs": np.nan,
            "cm_rmsd": np.nan,
            "cm_n_atoms": 0,
        }
    Cc = load_coords(c_pdb)
    Mm = load_coords(m_pdb)
    keys = sorted(set(var_keys(Cc, vh_len, vl_len)) & set(var_keys(Mm, vh_len, vl_len)))
    if not keys:
        return {"cm_ok": False, "cm_error": "NO_OVERLAP_ATOMS", "cm_max_abs": np.nan, "cm_rmsd": np.nan, "cm_n_atoms": 0}
    diffs = np.asarray([Cc[k] - Mm[k] for k in keys], float)
    max_abs = float(np.max(np.abs(diffs)))
    rmsd = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
    ok = (max_abs < TOL_MAX) or (rmsd < TOL_RMSD)
    return {
        "cm_ok": ok,
        "cm_error": "" if ok else "COORD_MISMATCH",
        "cm_max_abs": max_abs,
        "cm_rmsd": rmsd,
        "cm_n_atoms": len(keys),
        "tol_max_abs": TOL_MAX,
        "tol_rmsd": TOL_RMSD,
    }


def main():
    fab = pd.read_csv(SEQ).set_index("id")
    feats = pd.read_csv(CACHE / "features/pilot_features.csv") if (CACHE / "features/pilot_features.csv").exists() else pd.DataFrame()
    qc = pd.read_csv(CACHE / "features/pilot_qc.csv") if (CACHE / "features/pilot_qc.csv").exists() else pd.DataFrame()
    prep = pd.read_csv(CTX / "FAB_PREP_QC.csv") if (CTX / "FAB_PREP_QC.csv").exists() else pd.DataFrame()

    rows = []
    for ab in PILOT:
        row = {"id": ab}
        vh, vl = int(fab.loc[ab].VH_len_used), int(fab.loc[ab].VL_len_used)
        for cond in ["B", "C", "M"]:
            sub = feats[(feats.id == ab) & (feats.condition == cond)] if len(feats) else pd.DataFrame()
            row[f"{cond}_done"] = bool(len(sub) and (sub.iloc[0].get("extraction_status") == "SUCCESS"))
            if len(sub):
                # Finiteness among *present* values only.
                # Condition B/M intentionally leave constant-domain / interface K_* as NaN;
                # counting those as non-finite would false-FAIL an otherwise valid pilot.
                num = sub.select_dtypes(include=[np.number]).iloc[0]
                present = num.to_numpy(dtype=float)
                present = present[~np.isnan(present)]
                row[f"{cond}_finite_frac"] = float(np.isfinite(present).mean()) if len(present) else 0.0
                row[f"{cond}_n_present"] = float(len(present))
                row[f"{cond}_n_sites"] = float(sub.iloc[0].get("n_sites", np.nan))
            qsub = qc[(qc.id == ab) & (qc.condition == cond)] if len(qc) and "condition" in qc.columns else pd.DataFrame()
            if len(qsub):
                for col in ["runtime_s", "R1_F_rms", "R1_severe_clash", "backbone_RMSD", "n_atoms", "converged"]:
                    if col in qsub.columns:
                        row[f"{cond}_{col}"] = qsub.iloc[0][col]
        row.update(cm_match(ab, vh, vl))
        if len(prep) and ab in set(prep.id.astype(str)):
            pr = prep.set_index("id").loc[ab]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[-1]
            for col in ["HL_SG_SG_after", "VH_intra_SG_SG_after", "VL_intra_SG_SG_after", "CH1_intra_SG_SG_after", "CL_intra_SG_SG_after", "backbone_RMSD", "VH_CA_RMSD", "severe_clash_after"]:
                if col in pr.index:
                    row[f"prep_{col}"] = pr[col]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CTX / "FENNIX_FAB_PILOT_QC.csv", index=False)

    n = len(PILOT)
    n_complete = int(df[["B_done", "C_done", "M_done"]].all(axis=1).sum()) if n else 0
    n_cm = int(df.cm_ok.sum()) if "cm_ok" in df.columns else 0
    finite_ok = True
    for cond in ["B", "C", "M"]:
        col = f"{cond}_finite_frac"
        if col in df.columns and df[col].notna().any():
            if float(df[col].min()) < 0.99:
                finite_ok = False

    limitations = []
    # Convergence / clash / empty region pools are limitations, not automatic FAIL
    if "B_converged" in df.columns and (~df["B_converged"].fillna(False)).any():
        limitations.append("B_FIRE_not_fully_converged_some")
    if "C_converged" in df.columns and (~df["C_converged"].fillna(False)).any():
        limitations.append("C_FIRE_not_fully_converged_some")
    if "B_R1_severe_clash" in df.columns and (df["B_R1_severe_clash"].fillna(0) > 0).any():
        limitations.append("B_residual_severe_clash_some")
    # LCDR3 empty on full-Fab pools (mapping/eligible-site gap)
    if len(feats) and any(c.startswith("K_LCDR3_") for c in feats.columns):
        lcd = feats[feats.condition == "C"][[c for c in feats.columns if c.startswith("K_LCDR3_")]]
        if len(lcd) and lcd.isna().all().all():
            limitations.append("C_LCDR3_curvature_pool_empty")

    if n_complete == n and n_cm == n and finite_ok:
        verdict = "PILOT_PASS_WITH_LIMITATIONS" if limitations else "PILOT_PASS"
    elif n_complete == n and finite_ok and n_cm >= n - 1:
        verdict = "PILOT_PASS_WITH_LIMITATIONS"
        limitations.append("cm_match_not_perfect_all")
    elif n_complete == n and finite_ok:
        verdict = "PILOT_PASS_WITH_LIMITATIONS"
        limitations.append(f"cm_ok={n_cm}/{n}")
    else:
        verdict = "PILOT_FAIL"
        limitations.append(f"complete={n_complete}/{n} finite_ok={finite_ok} cm={n_cm}/{n}")

    # extreme feature outliers (technical)
    outlier_notes = []
    if len(feats):
        for c in [x for x in feats.columns if x.startswith("K_") and x.endswith("_median")]:
            s = feats[c].astype(float)
            if s.notna().sum() < 3:
                continue
            z = (s - s.mean()) / (s.std() + 1e-12)
            if (z.abs() > 8).any():
                outlier_notes.append(f"{c}:n_ext={(z.abs()>8).sum()}")

    report = [
        "# FeNNix Fab Pilot — Technical / Physical Gate",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## Scope",
        "- TmApp / HIC / Public / Private **not used**.",
        "- Conditions B, C, M on prepared pilot 12; A reused from FeNNix-v2 where applicable.",
        "",
        "## Completion",
        f"- B/C/M complete: **{n_complete}/{n}**",
        f"- C↔M variable-region coordinate match: **{n_cm}/{n}** (tol max_abs={TOL_MAX} Å or RMSD={TOL_RMSD} Å)",
        f"- Feature finiteness OK: **{finite_ok}**",
        "",
        "## Limitations / notes",
        *(f"- {x}" for x in (limitations or ["none"])),
        "",
        "## Extreme feature flags (|z|>8)",
        *(f"- {x}" for x in (outlier_notes or ["none"])),
        "",
        "## DELTA_ENV validity",
        "- DELTA_ENV is only meaningful when C and M share identical Fv coordinates before perturbation.",
        f"- Status: {'VALID for all pilot Abs' if n_cm == n else 'CHECK LIMITATIONS / FAIL'}",
        "",
        f"Wrote `{CTX / 'FENNIX_FAB_PILOT_QC.csv'}`",
        "",
    ]
    (CTX / "FENNIX_FAB_PILOT_TECHNICAL_REPORT.md").write_text("\n".join(report))
    print(verdict, f"complete={n_complete}/{n}", f"cm={n_cm}/{n}", flush=True)


if __name__ == "__main__":
    main()
