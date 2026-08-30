#!/usr/bin/env python3
"""Build Meltome accession→Tm index from Nature Supplementary Table S2 (MOESM4).

Does NOT download raw PRIDE MS files. Sequences can be joined later via UniProt accessions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "raw" / "meltome" / "supplements" / "MOESM4.xlsx"
META = ROOT / "raw" / "meltome" / "supplements" / "MOESM3.xlsx"
OUT = ROOT / "interim" / "meltome_tm_index.csv"
OUT_META = ROOT / "interim" / "meltome_dataset_metadata.csv"


def main() -> None:
    meta = pd.read_excel(META, sheet_name="Meltome data set")
    meta.to_csv(OUT_META, index=False)

    xl = pd.ExcelFile(XLSX)
    frames = []
    for sheet in xl.sheet_names:
        if not sheet.startswith("ma_"):
            continue
        df = pd.read_excel(XLSX, sheet_name=sheet)
        df.insert(0, "dataset_sheet", sheet)
        frames.append(df)
    all_tm = pd.concat(frames, ignore_index=True)
    # attach organism/context from meta via Dataset ID if possible
    # sheets ma_0001 correspond to rows; keep raw and merge loosely on order if Dataset ID present
    if "Dataset ID" in meta.columns:
        # ma_0001 -> often dataset id in meta row order; keep both
        pass
    all_tm.to_csv(OUT, index=False)

    n_prot = all_tm["Protein ID"].nunique() if "Protein ID" in all_tm.columns else None
    n_tm = all_tm["Melting point [°C]"].notna().sum() if "Melting point [°C]" in all_tm.columns else None
    print(
        f"Meltome TM index: rows={len(all_tm)} sheets={len(frames)} "
        f"unique_protein_ids={n_prot} tm_nonnull={n_tm}"
    )
    print(f"Metadata datasets: {len(meta)}")
    print(meta[["Dataset ID", "Organism", "Strain/tissue", "Cells/lysate"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
