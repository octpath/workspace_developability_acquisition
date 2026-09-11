#!/usr/bin/env python3
"""Extract Fv STATIC_SAP_KD residue + antibody-level artifacts (target-blind).

Uses ESMFold Fv via STRUCTURE_INPUT_CROSSWALK_v2 esmfold_canonical_path.
Does NOT use HIC / Tm labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(REPO / "organizer_extension" / "feature_prospecting"))

from antibody_transformer.static_sap_kd import (  # noqa: E402
    INCLUDE_SELF,
    KYTE_DOOLITTLE,
    N_POINTS_PRIMARY,
    N_POINTS_QC,
    PROBE,
    R_REF,
    R_SENSITIVITY,
    SAP3_COLS,
    SAP9_COLS,
    TIEN_MAXASA,
    aggregate_max_mean_sum,
    kd_norm,
    resolve_centroid,
    rsasa,
    sskd_scores,
)
from common.structure_utils import (  # noqa: E402
    build_residue_table,
    load_cdr_map,
    load_sequences,
)

CROSSWALK = REPO / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"
OUT_DIR = ROOT / "experiments" / "features"
FEAT_DIR = OUT_DIR  # static_sap_kd_*.parquet here
META_DIR = ROOT / "results"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _compute_sasa_on_rows(pdb_path: Path, n_points: int) -> dict[tuple[str, int], float]:
    """Map (pdb_chain, pdb_resseq) -> SASA with given n_points."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    ShrakeRupley(probe_radius=PROBE, n_points=n_points).compute(model, level="R")
    out = {}
    from Bio.PDB.Polypeptide import is_aa

    for ch in model.get_chains():
        for res in ch.get_residues():
            if not is_aa(res, standard=True):
                continue
            out[(ch.id, int(res.id[1]))] = float(getattr(res, "sasa", np.nan))
    return out


def process_one(
    aid: str,
    heavy: str,
    light: str,
    pdb_path: Path,
    cdr_rows: list[dict],
    *,
    n_points: int = N_POINTS_PRIMARY,
) -> tuple[list[dict], dict, str | None]:
    rows, err = build_residue_table(pdb_path, heavy, light, cdr_rows)
    if err:
        return [], {}, err
    # Override SASA if n_points differs from structure_utils default (100)
    if n_points != N_POINTS_PRIMARY:
        sasa_map = _compute_sasa_on_rows(pdb_path, n_points)
        for r in rows:
            key = (r["pdb_chain"], r["pdb_resseq"])
            if key in sasa_map:
                r["sasa"] = sasa_map[key]
                r["rasa"] = rsasa(r["sasa"], r["amino_acid"])

    n = len(rows)
    centroids = np.full((n, 3), np.nan)
    kd_raw = np.full(n, np.nan)
    kd_n = np.full(n, np.nan)
    sasa_arr = np.full(n, np.nan)
    maxasa_arr = np.full(n, np.nan)
    rasa_arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    provenance = []

    for i, r in enumerate(rows):
        aa = r["amino_acid"]
        kd_raw[i] = KYTE_DOOLITTLE.get(aa, np.nan)
        kd_n[i] = kd_norm(aa)
        sasa_arr[i] = float(r["sasa"]) if np.isfinite(r["sasa"]) else np.nan
        maxasa_arr[i] = TIEN_MAXASA.get(aa, np.nan)
        rasa_arr[i] = rsasa(sasa_arr[i], aa) if np.isfinite(sasa_arr[i]) else np.nan
        cent, prov = resolve_centroid(aa, r["atoms"], r["ca"])
        provenance.append(prov)
        if cent is not None:
            centroids[i] = cent
        sasa_ok = np.isfinite(rasa_arr[i]) and np.isfinite(kd_n[i])
        coord_ok = cent is not None and np.all(np.isfinite(cent))
        valid[i] = bool(sasa_ok and coord_ok)

    sskd5 = sskd_scores(centroids, kd_n, rasa_arr, valid, R_REF, include_self=INCLUDE_SELF)
    sskd10 = sskd_scores(
        centroids, kd_n, rasa_arr, valid, R_SENSITIVITY, include_self=INCLUDE_SELF
    )

    res_out = []
    for i, r in enumerate(rows):
        imgt = None
        # cdr_rows already joined into region; recover IMGT if present
        for c in cdr_rows:
            if c["chain"] == r["chain"] and int(c["sequence_index"]) == int(r["sequence_index"]):
                imgt = c.get("imgt_number")
                break
        res_out.append(
            {
                "id": aid,
                "chain": r["chain"],
                "residue_index": int(r["sequence_index"]),
                "IMGT_position": imgt,
                "region": r["region"],
                "aa": r["amino_acid"],
                "centroid_x": float(centroids[i, 0]) if valid[i] else np.nan,
                "centroid_y": float(centroids[i, 1]) if valid[i] else np.nan,
                "centroid_z": float(centroids[i, 2]) if valid[i] else np.nan,
                "centroid_provenance": provenance[i],
                "SASA": float(sasa_arr[i]) if np.isfinite(sasa_arr[i]) else np.nan,
                "Tien_MaxASA": float(maxasa_arr[i]) if np.isfinite(maxasa_arr[i]) else np.nan,
                "rSASA": float(rasa_arr[i]) if np.isfinite(rasa_arr[i]) else np.nan,
                "KD_raw": float(kd_raw[i]) if np.isfinite(kd_raw[i]) else np.nan,
                "KD_norm": float(kd_n[i]) if np.isfinite(kd_n[i]) else np.nan,
                "SSKD_R_REF": float(sskd5[i]) if np.isfinite(sskd5[i]) else np.nan,
                "SSKD_R10": float(sskd10[i]) if np.isfinite(sskd10[i]) else np.nan,
                "coordinate_valid": bool(valid[i]),
                "sasa_valid": bool(np.isfinite(rasa_arr[i])),
                "structure_path": str(pdb_path),
                "structure_sha256": _sha_file(pdb_path),
            }
        )

    def agg(mask: np.ndarray) -> tuple[float, float, float]:
        return aggregate_max_mean_sum(sskd5[mask & valid])

    all_m = np.ones(n, dtype=bool)
    h_m = np.array([r["chain"] == "H" for r in rows])
    l_m = np.array([r["chain"] == "L" for r in rows])
    a_max, a_mean, a_sum = agg(all_m)
    h_max, h_mean, h_sum = agg(h_m)
    l_max, l_mean, l_sum = agg(l_m)
    # R10 global for QC
    a10_max, a10_mean, a10_sum = aggregate_max_mean_sum(sskd10[valid])

    ab = {
        "id": aid,
        "SSKD_ALL_MAX": a_max,
        "SSKD_ALL_MEAN": a_mean,
        "SSKD_ALL_SUM": a_sum,
        "SSKD_H_MAX": h_max,
        "SSKD_H_MEAN": h_mean,
        "SSKD_H_SUM": h_sum,
        "SSKD_L_MAX": l_max,
        "SSKD_L_MEAN": l_mean,
        "SSKD_L_SUM": l_sum,
        "SSKD10_ALL_MAX": a10_max,
        "SSKD10_ALL_MEAN": a10_mean,
        "SSKD10_ALL_SUM": a10_sum,
        "n_residues": n,
        "n_valid": int(valid.sum()),
        "n_H": int(h_m.sum()),
        "n_L": int(l_m.sum()),
        "total_SASA": float(np.nansum(sasa_arr)),
        "structure_path": str(pdb_path),
        "structure_sha256": _sha_file(pdb_path),
        "structure_scope": "Fv",
        "n_points": n_points,
        "R_REF": R_REF,
    }
    return res_out, ab, None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    cw = pd.read_csv(CROSSWALK)
    seqs = load_sequences()
    cdr_map = load_cdr_map()

    res_rows: list[dict] = []
    ab_rows: list[dict] = []
    qc_rows: list[dict] = []
    errors = []

    # primary extraction
    for _, crow in cw.iterrows():
        aid = str(crow["id"])
        if aid not in seqs.index:
            errors.append({"id": aid, "error": "missing_sequence"})
            continue
        pdb = Path(str(crow["esmfold_canonical_path"]))
        if not pdb.exists():
            errors.append({"id": aid, "error": f"missing_pdb:{pdb}"})
            continue
        heavy = str(seqs.loc[aid, "heavy"])
        light = str(seqs.loc[aid, "light"])
        cdr_rows = cdr_map.get(aid, [])
        res, ab, err = process_one(aid, heavy, light, pdb, cdr_rows, n_points=N_POINTS_PRIMARY)
        if err:
            errors.append({"id": aid, "error": err})
            continue
        res_rows.extend(res)
        ab_rows.append(ab)
        qc_rows.append({"id": aid, "status": "OK", **{k: ab[k] for k in ("n_residues", "n_valid", "n_H", "n_L")}})
        if len(ab_rows) % 50 == 0:
            print(f"STATIC_SAP_KD {len(ab_rows)} antibodies", flush=True)

    res_df = pd.DataFrame(res_rows)
    ab_df = pd.DataFrame(ab_rows)
    assert len(ab_df) == 324, f"expected 324 antibodies, got {len(ab_df)}; errors={errors[:5]}"

    # antibody artifacts
    g3 = ab_df[["id"] + SAP3_COLS].copy()
    c9 = ab_df[["id"] + SAP9_COLS].copy()
    res_path = OUT_DIR / "static_sap_kd_residue.parquet"
    g3_path = OUT_DIR / "static_sap_kd_antibody_global3.parquet"
    c9_path = OUT_DIR / "static_sap_kd_antibody_chain9.parquet"
    ab_full_path = OUT_DIR / "static_sap_kd_antibody_full_qc.parquet"
    res_df.to_parquet(res_path, index=False)
    g3.to_parquet(g3_path, index=False)
    c9.to_parquet(c9_path, index=False)
    ab_df.to_parquet(ab_full_path, index=False)

    # QC checks
    issues = []
    for col in SAP9_COLS:
        if not np.isfinite(c9[col]).all():
            issues.append(f"nonfinite {col}")
        if (c9[col] < -1e-9).any():
            issues.append(f"negative {col}")
    if (res_df["rSASA"].dropna() < -1e-9).any() or (res_df["rSASA"].dropna() > 1 + 1e-9).any():
        issues.append("rSASA out of [0,1]")
    if (res_df["KD_norm"].dropna() < -1e-9).any() or (res_df["KD_norm"].dropna() > 1 + 1e-9).any():
        issues.append("KD_norm out of [0,1]")
    if not ((c9["SSKD_ALL_MAX"] + 1e-9 >= c9["SSKD_ALL_MEAN"]).all()):
        issues.append("MAX < MEAN")
    if not ((c9["SSKD_ALL_SUM"] + 1e-9 >= c9["SSKD_ALL_MAX"]).all()):
        issues.append("SUM < MAX")

    # numerical sensitivity: subsample or all if affordable — do all 324 with n_points=960
    print("QC n_points=960 sensitivity...", flush=True)
    sens_rows = []
    for _, crow in cw.iterrows():
        aid = str(crow["id"])
        if aid not in seqs.index:
            continue
        pdb = Path(str(crow["esmfold_canonical_path"]))
        if not pdb.exists():
            continue
        heavy = str(seqs.loc[aid, "heavy"])
        light = str(seqs.loc[aid, "light"])
        cdr_rows = cdr_map.get(aid, [])
        _, ab960, err = process_one(aid, heavy, light, pdb, cdr_rows, n_points=N_POINTS_QC)
        if err:
            continue
        base = ab_df.set_index("id").loc[aid]
        for col in SAP3_COLS:
            a, b = float(base[col]), float(ab960[col])
            rel = abs(a - b) / max(abs(a), 1e-12)
            sens_rows.append(
                {
                    "id": aid,
                    "feature": col,
                    "n100": a,
                    "n960": b,
                    "abs_diff": abs(a - b),
                    "rel_diff": rel,
                }
            )
    sens = pd.DataFrame(sens_rows)
    sens_path = META_DIR / "STATIC_SAP_KD_NUMERICAL_SENSITIVITY_DETAIL.csv"
    sens.to_csv(sens_path, index=False)

    # Pearson / Spearman per feature between n100 and n960
    sens_summary = []
    for col in SAP3_COLS:
        sub = sens[sens.feature == col]
        from scipy.stats import pearsonr, spearmanr

        pr = pearsonr(sub["n100"], sub["n960"])[0]
        sr = spearmanr(sub["n100"], sub["n960"])[0]
        med_rel = float(sub["rel_diff"].median())
        max_rel = float(sub["rel_diff"].max())
        sens_summary.append(
            {
                "feature": col,
                "pearson": pr,
                "spearman": sr,
                "median_rel_diff": med_rel,
                "max_rel_diff": max_rel,
            }
        )
    sens_sum_df = pd.DataFrame(sens_summary)
    # STOP only on unexpectedly large instability (MEAN/SUM are the robust
    # aggregates; MAX is inherently more sensitive to single-residue SASA noise).
    mean_row = sens_sum_df.set_index("feature").loc["SSKD_ALL_MEAN"]
    sum_row = sens_sum_df.set_index("feature").loc["SSKD_ALL_SUM"]
    unstable = bool(
        mean_row["pearson"] < 0.95
        or sum_row["pearson"] < 0.95
        or mean_row["median_rel_diff"] > 0.05
        or sum_row["median_rel_diff"] > 0.05
    )

    # radius sensitivity R5 vs R10
    rad_rows = []
    for col3, col10 in zip(
        ["SSKD_ALL_MAX", "SSKD_ALL_MEAN", "SSKD_ALL_SUM"],
        ["SSKD10_ALL_MAX", "SSKD10_ALL_MEAN", "SSKD10_ALL_SUM"],
    ):
        from scipy.stats import pearsonr, spearmanr

        a = ab_df[col3].to_numpy(float)
        b = ab_df[col10].to_numpy(float)
        rad_rows.append(
            {
                "pair": f"{col3}_vs_{col10}",
                "pearson": float(pearsonr(a, b)[0]),
                "spearman": float(spearmanr(a, b)[0]),
                "median_ratio_R10_over_R5": float(np.median(b / np.maximum(a, 1e-12))),
                "scale_mean_R5": float(a.mean()),
                "scale_mean_R10": float(b.mean()),
            }
        )

    art_hash = {
        "residue": _sha_file(res_path),
        "global3": _sha_file(g3_path),
        "chain9": _sha_file(c9_path),
        "n_antibodies": len(ab_df),
        "n_residue_rows": len(res_df),
    }

    qc_md = META_DIR / "STATIC_SAP_KD_QC.md"
    qc_md.write_text(
        "\n".join(
            [
                "# STATIC_SAP_KD QC",
                "",
                f"- antibodies: {len(ab_df)} / 324",
                f"- residue rows: {len(res_df)}",
                f"- structure_scope: **Fv** (ESMFold via crosswalk)",
                f"- R_REF: {R_REF} Å (SOURCE_SPECIFIED from STATIC-SAP FEATURE_SPEC radii_A[0])",
                f"- R_SENSITIVITY: {R_SENSITIVITY} Å (QC only; not trained)",
                f"- Shrake-Rupley: probe={PROBE}, n_points={N_POINTS_PRIMARY} (**SOURCE_SPECIFIED** / FEATURE_SPEC + repository)",
                f"- MaxASA: Tien2013",
                f"- hydrophobicity: Kyte–Doolittle min-max → KD_norm∈[0,1]",
                f"- centroid: non-H side-chain arithmetic mean; Gly→CA",
                f"- include_self: {INCLUDE_SELF}",
                f"- issues: {issues or 'NONE'}",
                f"- extraction_errors: {len(errors)}",
                f"- artifact_hashes: `{json.dumps(art_hash)}`",
                "",
                "## Distributions (SAP9)",
                "```",
                c9[SAP9_COLS].describe().to_string(),
                "```",
                "",
                "## Validity",
                f"- mean n_valid: {ab_df['n_valid'].mean():.1f}",
                f"- mean n_H / n_L: {ab_df['n_H'].mean():.1f} / {ab_df['n_L'].mean():.1f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sens_md = META_DIR / "STATIC_SAP_KD_NUMERICAL_SENSITIVITY.md"
    sens_md.write_text(
        "\n".join(
            [
                "# STATIC_SAP_KD numerical / radius sensitivity",
                "",
                "## Shrake–Rupley n_points 100 vs 960",
                "```",
                sens_sum_df.to_string(index=False),
                "```",
                "",
                f"- unstable_flag: {unstable}",
                "",
                "## Radius R_REF=5 vs R=10 (descriptor QC only)",
                "```",
                pd.DataFrame(rad_rows).to_string(index=False),
                "```",
                "",
                "No model selection uses R=10 or n_points=960.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    meta = {
        "descriptor": "STATIC_SAP_KD",
        "structure_scope": "Fv",
        "R_REF": R_REF,
        "R_REF_label": "SOURCE_SPECIFIED_FROM_STATIC_SAP_FEATURE_SPEC",
        "n_points_primary": N_POINTS_PRIMARY,
        "n_points_label": "SOURCE_SPECIFIED",
        "n_points_qc": N_POINTS_QC,
        "include_self": INCLUDE_SELF,
        "artifact_hashes": art_hash,
        "issues": issues,
        "errors": errors,
        "sasa_unstable": unstable,
        "paths": {
            "residue": str(res_path.relative_to(ROOT)),
            "global3": str(g3_path.relative_to(ROOT)),
            "chain9": str(c9_path.relative_to(ROOT)),
        },
    }
    (META_DIR / "STATIC_SAP_KD_EXTRACT_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if unstable:
        print("STOP: SASA n_points instability too large — do not train", flush=True)
        return 2
    if issues:
        print("QC ISSUES:", issues, flush=True)
        return 1
    print("OK STATIC_SAP_KD extraction complete", flush=True)
    print(json.dumps(art_hash, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
