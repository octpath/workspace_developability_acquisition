#!/usr/bin/env python3
"""Stage 1: dataset audit + assay definitions + TRIPLE_CORE tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    DATA,
    ORGANIZER_ONLY_COLS,
    REPORTS,
    SRC_JOINED,
    SRC_XLSX,
    TARGET_COLS,
    clean_aa,
    ensure_dirs,
    pair_hash,
    seq_hash,
    sha256_file,
    write_json,
)


def quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {f"q{int(q*100):02d}": float(s.quantile(q)) for q in qs}


def outlier_iqr(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((s < lo) | (s > hi)).sum())
    return {"iqr_lo": float(lo), "iqr_hi": float(hi), "n_outliers_iqr": n, "frac": float(n / len(s))}


def target_summary(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    return {
        "n_complete": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "missing_frac": float(s.isna().mean()),
        "mean": float(s.mean()) if s.notna().any() else None,
        "sd": float(s.std()) if s.notna().any() else None,
        "median": float(s.median()) if s.notna().any() else None,
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
        "quantiles": quantiles(s),
        "outliers_iqr": outlier_iqr(s),
    }


def load_mmc2_regions() -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(SRC_XLSX, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    out = []
    for row in rows[1:]:
        if not row[0] or not isinstance(row[0], str) or not row[0].startswith("ADI-"):
            continue
        d = dict(zip(header, row))
        out.append(
            {
                "antibody_id": d["Clone name"],
                "author_vh_fr1": d["VH FR1"],
                "author_vh_cdr1": d["VH CDR1"],
                "author_vh_fr2": d["VH FR2"],
                "author_vh_cdr2": d["VH CDR2"],
                "author_vh_fr3": d["VH FR3"],
                "author_vh_cdr3": d["VH CDR3"],
                "author_vh_fr4": d["VH FR4"],
                "author_vl_fr1": d["VL FR1"],
                "author_vl_cdr1": d["VL CDR1"],
                "author_vl_fr2": d["VL FR2"],
                "author_vl_cdr2": d["VL CDR2"],
                "author_vl_fr3": d["VL FR3"],
                "author_vl_cdr3": d["VL CDR3"],
                "author_vl_fr4": d["VL FR4"],
                "author_vh_protein": d["VH Protein"],
                "author_vl_protein": d["VL Protein"],
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(SRC_JOINED)
    assert df["antibody_id"].is_unique
    assert len(df) == 400

    df["heavy_raw"] = df["heavy"].astype(str)
    df["light_raw"] = df["light"].astype(str)
    df["heavy"] = df["heavy_raw"].map(clean_aa)
    df["light"] = df["light_raw"].map(clean_aa)
    df["heavy_had_gap"] = df["heavy_raw"].str.contains("-", regex=False)
    df["light_had_gap"] = df["light_raw"].str.contains("-", regex=False)
    df["vh_len"] = df["heavy"].str.len()
    df["vl_len"] = df["light"].str.len()
    df["seq_pair_hash"] = [pair_hash(h, l) for h, l in zip(df["heavy"], df["light"])]
    df["vh_hash"] = df["heavy"].map(seq_hash)
    df["vl_hash"] = df["light"].map(seq_hash)

    # Filtering rules (documented; no silent drops for modeling tables)
    filter_rules = [
        "Start from interim/shehata_full_joined.csv (400 ADI clones with VH+VL protein).",
        "mmc2.xlsx has 402 sheet rows of which 2 are legend footnotes (not clones) → 400 clones.",
        "Gap characters '-' in author protein strings are stripped for modeling sequences (clean_aa).",
        "Non-standard letters outside ACDEFGHIKLMNPQRSTVWY are stripped if present.",
        "Rows are NEVER deleted solely for missing labels; subset tables are created instead.",
        "TRIPLE_CORE = rows with non-null PSR, HIC, and TmApp.",
        "PARTICIPANT_LEGAL inputs for future competition: antibody_id, heavy, light only.",
        "ORGANIZER_ONLY_AUDIT: b_cell_subset, author vh/vl_germline, join metadata.",
    ]

    regions = load_mmc2_regions()
    df = df.merge(regions, on="antibody_id", how="left")
    assert df["author_vh_protein"].notna().all()

    # Duplicate / conflict checks
    pair_dups = df.groupby(["heavy", "light"]).size()
    n_dup_pairs = int((pair_dups > 1).sum())
    conflict_rows = []
    for cols, name in [
        (["psr_score"], "PSR"),
        (["hic_rt_min"], "HIC"),
        (["tm_app_C"], "TmApp"),
    ]:
        g = df.groupby(["heavy", "light"])[cols[0]].nunique(dropna=True)
        # same pair different labels
        bad = g[g > 1]
        if len(bad):
            conflict_rows.append((name, int(len(bad))))

    audit = {
        "source_csv": str(SRC_JOINED),
        "source_csv_sha256": sha256_file(SRC_JOINED),
        "source_xlsx": str(SRC_XLSX),
        "source_xlsx_sha256": sha256_file(SRC_XLSX),
        "n_rows": int(len(df)),
        "n_unique_antibody_ids": int(df["antibody_id"].nunique()),
        "n_unique_VH": int(df["heavy"].nunique()),
        "n_unique_VL": int(df["light"].nunique()),
        "n_unique_VH_VL_pairs": int(df.groupby(["heavy", "light"]).ngroups),
        "n_duplicate_VH_VL_pairs": n_dup_pairs,
        "conflicting_labels_same_pair": conflict_rows,
        "missing_VH": int((df["heavy"].str.len() == 0).sum()),
        "missing_VL": int((df["light"].str.len() == 0).sum()),
        "n_heavy_with_author_gaps": int(df["heavy_had_gap"].sum()),
        "n_light_with_author_gaps": int(df["light_had_gap"].sum()),
        "vh_len": {
            "min": int(df["vh_len"].min()),
            "max": int(df["vh_len"].max()),
            "mean": float(df["vh_len"].mean()),
            "median": float(df["vh_len"].median()),
        },
        "vl_len": {
            "min": int(df["vl_len"].min()),
            "max": int(df["vl_len"].max()),
            "mean": float(df["vl_len"].mean()),
            "median": float(df["vl_len"].median()),
        },
        "b_cell_subset_counts": df["b_cell_subset"].value_counts().to_dict(),
        "targets": {k: target_summary(df, c) for k, c in TARGET_COLS.items()},
        "n_triple_complete": int(
            df.dropna(subset=list(TARGET_COLS.values())).shape[0]
        ),
        "filter_rules": filter_rules,
        "organizer_only_columns": ORGANIZER_ONLY_COLS,
    }

    # Expected counts confirmation
    assert audit["targets"]["PSR"]["n_complete"] == 398
    assert audit["targets"]["HIC"]["n_complete"] == 348
    assert audit["targets"]["TmApp"]["n_complete"] == 346
    assert audit["n_triple_complete"] == 324

    # Save tables
    full_path = DATA / "shehata_b1_full.csv"
    df.to_csv(full_path, index=False)

    triple = df.dropna(subset=list(TARGET_COLS.values())).copy()
    assert len(triple) == 324
    triple.to_csv(DATA / "triple_core.csv", index=False)

    for name, col in TARGET_COLS.items():
        sub = df[df[col].notna()].copy()
        sub.to_csv(DATA / f"{name.lower()}_full.csv", index=False)

    write_json(DATA / "dataset_audit.json", audit)

    # Markdown audit
    lines = [
        "# Dataset audit — Shehata 2019 (Gate B1)",
        "",
        f"- Source CSV: `{SRC_JOINED}` (sha256 `{audit['source_csv_sha256'][:16]}…`)",
        f"- Source XLSX: `{SRC_XLSX}` (sha256 `{audit['source_xlsx_sha256'][:16]}…`)",
        "",
        "## Counts",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| N rows | {audit['n_rows']} |",
        f"| Unique antibody IDs | {audit['n_unique_antibody_ids']} |",
        f"| Unique VH | {audit['n_unique_VH']} |",
        f"| Unique VL | {audit['n_unique_VL']} |",
        f"| Unique VH/VL pairs | {audit['n_unique_VH_VL_pairs']} |",
        f"| Duplicate VH/VL pairs | {audit['n_duplicate_VH_VL_pairs']} |",
        f"| Missing VH | {audit['missing_VH']} |",
        f"| Missing VL | {audit['missing_VL']} |",
        f"| VH with author gap chars | {audit['n_heavy_with_author_gaps']} |",
        f"| VL with author gap chars | {audit['n_light_with_author_gaps']} |",
        f"| TRIPLE_CORE (PSR∩HIC∩TmApp) | {audit['n_triple_complete']} |",
        "",
        "## Sequence length distributions",
        "",
        f"- VH: min={audit['vh_len']['min']}, median={audit['vh_len']['median']:.1f}, max={audit['vh_len']['max']}",
        f"- VL: min={audit['vl_len']['min']}, median={audit['vl_len']['median']:.1f}, max={audit['vl_len']['max']}",
        "",
        "## B-cell subset (ORGANIZER_ONLY)",
        "",
    ]
    for k, v in audit["b_cell_subset_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Targets", ""]
    for tname, tsum in audit["targets"].items():
        lines += [
            f"### {tname} (`{TARGET_COLS[tname]}`)",
            "",
            f"- N complete: **{tsum['n_complete']}** (missing {tsum['n_missing']})",
            f"- mean±SD: {tsum['mean']:.4g} ± {tsum['sd']:.4g}",
            f"- median: {tsum['median']:.4g}",
            f"- min/max: {tsum['min']:.4g} / {tsum['max']:.4g}",
            f"- IQR outliers: {tsum['outliers_iqr']['n_outliers_iqr']} ({tsum['outliers_iqr']['frac']:.1%})",
            f"- quantiles: " + ", ".join(f"{k}={v:.4g}" for k, v in tsum["quantiles"].items()),
            "",
        ]
    lines += [
        "## Expected count confirmation",
        "",
        "| Target | Expected | Observed |",
        "|--------|----------:|----------:|",
        "| PSR | ~398 | 398 |",
        "| HIC | ~348 | 348 |",
        "| TmApp | ~346 | 346 |",
        "| triple | ~324 | 324 |",
        "",
        "## Filtering rules",
        "",
    ]
    for r in filter_rules:
        lines.append(f"1. {r}")
    lines += [
        "",
        "## Conflicting labels",
        "",
        f"- Same VH/VL pair with conflicting labels: {conflict_rows or 'none'}",
        "",
        "## Participant vs organizer columns",
        "",
        "- PARTICIPANT_LEGAL: `antibody_id`, `heavy`, `light`",
        f"- ORGANIZER_ONLY_AUDIT: {', '.join(ORGANIZER_ONLY_COLS)}",
        "",
    ]
    (REPORTS / "dataset_audit.md").write_text("\n".join(lines) + "\n")

    assay = """# Assay definitions — Shehata et al. 2019

Source: Cell Reports 2019 supplement `mmc2.xlsx` column headers and paper terminology.

## PSR — PSR Score

- Column (supplement): **PSR Score**
- Internal column: `psr_score`
- Meaning: polyspecificity reagent / nonspecific binding assay score from the study panel.
- Higher values indicate greater polyspecific / nonspecific binding (generally less desirable for developability).
- Continuous score; N complete = 398 of 400 paired clones.

## HIC — HIC retention time (min)

- Column (supplement): **HIC retention time (min)**
- Internal column: `hic_rt_min`
- Meaning: hydrophobic interaction chromatography retention time in minutes.
- Higher retention generally corresponds to stronger hydrophobic interaction with the HIC resin.
- This is a **hydrophobicity / developability proxy**, **not** an aggregation assay.
- Continuous; units: minutes; N complete = 348.

## TmApp — TmApp (°C)

- Column (supplement): **TmApp (°C)**
- Internal column: `tm_app_C`
- Meaning: apparent thermal transition / conformational stability measure reported by the authors as TmApp.
- Units: **degrees Celsius (°C)**.
- Higher TmApp generally corresponds to greater apparent thermal stability.
- Continuous; N complete = 346.

## Biological structure of the panel

Antibodies originate from human B-cell repertoire subsets reported in the supplement:

- Naïve
- IgM memory
- IgG memory
- LLPCs (long-lived plasma cells)

Author-provided germline annotations (`VH Germline`, `VL Germline`) and B-cell subset are **ORGANIZER_ONLY_AUDIT** variables for confounding diagnostics, not participant features.

## Feature legality

| Class | Contents |
|-------|----------|
| PARTICIPANT_LEGAL | `id` / `antibody_id`, `heavy`, `light` (+ derived sequence/structure features from those alone) |
| ORGANIZER_ONLY_AUDIT | `b_cell_subset`, author `vh_germline` / `vl_germline`, study metadata |
| ILLEGAL_FOR_PARTICIPANTS | Models that use organizer-only columns as inputs (oracle / confounding audits only) |
"""
    (REPORTS / "assay_definitions.md").write_text(assay)

    from b1_common import METRICS

    corr = triple[list(TARGET_COLS.values())].corr(method="spearman")
    corr.to_csv(METRICS / "target_corr_spearman.csv")
    print("AUDIT_OK", audit["n_rows"], "triple", audit["n_triple_complete"])
    print("wrote", REPORTS / "dataset_audit.md")


if __name__ == "__main__":
    main()
