#!/usr/bin/env python3
"""Reconstruct Shehata paired-sequence developability tables from public derivatives."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw" / "shehata" / "obstacle_shehata" / "shehata.csv"
HF = ROOT / "raw" / "shehata" / "hf" / "data" / "test.csv"
OUT_LABELS = ROOT / "interim" / "shehata_labels.csv"
OUT_FASTA = ROOT / "interim" / "shehata_sequences.fasta"
OUT_AUDIT = ROOT / "interim" / "shehata_join_audit.csv"


def main() -> None:
    df = pd.read_csv(SRC)
    # Prefer obstacle derivative with explicit heavy/light columns
    assert {"id", "heavy_seq", "light_seq", "psr_score"}.issubset(df.columns)

    out = pd.DataFrame(
        {
            "antibody_id": df["id"],
            "heavy": df["heavy_seq"],
            "light": df["light_seq"],
            "psr_score": df["psr_score"],
            "psr_binary_label": df["label"],
            "b_cell_subset": df["b_cell_subset"],
            "source": df["source"],
            "label_source": "shehata2019_derivative_csv_from_mmc2",
            "sequence_source": "shehata2019_mmc2_via_obstacle_pipeline",
            "join_confidence": "AUTHOR_PROVIDED_MAPPING",
            "notes": "Original mmc2.xlsx also reports Tm/HIC/charge per paper; raw xlsx not in public git (Cell Reports not OA). Only PSR recovered in public CSV derivatives.",
        }
    )
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LABELS, index=False)

    lines = []
    for _, r in out.iterrows():
        lines.append(f">{r['antibody_id']}_H\n{r['heavy']}")
        lines.append(f">{r['antibody_id']}_L\n{r['light']}")
    OUT_FASTA.write_text("\n".join(lines) + "\n")

    # Cross-check HF VH-only
    if HF.exists():
        hf = pd.read_csv(HF)
        merged = out.merge(hf, left_on="antibody_id", right_on="id", how="outer", suffixes=("", "_hf"))
        audit = pd.DataFrame(
            {
                "antibody_id": merged["antibody_id"].fillna(merged.get("id")),
                "has_heavy_light": merged["heavy"].notna() & merged["light"].notna(),
                "has_psr": merged["psr_score"].notna(),
                "hf_vh_match": merged["heavy"].fillna("") == merged.get("sequence", pd.Series([""] * len(merged))).fillna(""),
                "join_confidence": "AUTHOR_PROVIDED_MAPPING",
            }
        )
    else:
        audit = out[["antibody_id"]].copy()
        audit["has_heavy_light"] = True
        audit["has_psr"] = True
        audit["join_confidence"] = "AUTHOR_PROVIDED_MAPPING"
    audit.to_csv(OUT_AUDIT, index=False)

    print(
        f"Shehata: n={len(out)} paired VH/VL+PSR; "
        f"binary_high={(out['psr_binary_label']==1).sum()}; "
        f"unique_H={out['heavy'].nunique()} unique_L={out['light'].nunique()} "
        f"unique_pairs={out.drop_duplicates(['heavy','light']).shape[0]}"
    )
    print("OTHER ASSAYS (Tm/HIC): NOT in public CSV; require original mmc2.xlsx (ACCESS/OA limited)")


if __name__ == "__main__":
    main()
