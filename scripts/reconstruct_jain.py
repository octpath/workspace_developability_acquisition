#!/usr/bin/env python3
"""Join Jain 2017 sequences (SD02) with biophysical assays (SD03)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "jain" / "obstacle_jain"
OUT = ROOT / "interim" / "jain2017_joined.csv"
AUDIT = ROOT / "interim" / "jain2017_join_audit.csv"
COVERAGE = ROOT / "interim" / "jain2017_assay_coverage.csv"


def main() -> None:
    sd01 = pd.read_csv(RAW / "jain_sd01.csv")
    sd02 = pd.read_csv(RAW / "jain_sd02.csv")
    sd03 = pd.read_csv(RAW / "jain_sd03.csv")
    # Drop Excel footer / note rows (NaN name; annotation strings)
    sd03 = sd03[
        sd03["Name"].notna()
        & ~sd03["Name"].astype(str).str.startswith("aArbitrarily")
    ].copy()

    seq = sd02[["Name", "VH", "VL", "LC Class", "Source"]].copy()
    meta = sd01[["Name", "Clinical Status", "Type", "Original mAb Isotype or Format"]].copy()
    assays = sd03.copy()

    joined = seq.merge(assays, on="Name", how="inner", indicator=True)
    joined = joined.merge(meta, on="Name", how="left")

    audit = pd.DataFrame(
        {
            "antibody_id": joined["Name"],
            "sequence_complete": joined["VH"].notna() & joined["VL"].notna(),
            "in_sd02_and_sd03": joined["_merge"] == "both",
            "join_confidence": "EXACT_ID",
            "label_source": "pnas.1616408114.sd03",
            "sequence_source": "pnas.1616408114.sd02",
        }
    )
    audit.to_csv(AUDIT, index=False)

    assay_cols = [c for c in assays.columns if c != "Name"]
    cov = []
    for c in assay_cols:
        both = joined.loc[joined["VH"].notna() & joined["VL"].notna(), c].notna().sum()
        cov.append({"assay": c, "n_labels": int(both), "n_seq_plus_label": int(both)})
    pd.DataFrame(cov).to_csv(COVERAGE, index=False)

    out = joined.drop(columns=["_merge"])
    out.insert(0, "antibody_id", out["Name"])
    out.to_csv(OUT, index=False)

    print(f"Jain joined rows={len(out)}; sd02={len(sd02)} sd03_clean={len(sd03)}")
    print(pd.DataFrame(cov).to_string(index=False))


if __name__ == "__main__":
    main()
