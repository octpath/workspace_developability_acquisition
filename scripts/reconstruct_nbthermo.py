#!/usr/bin/env python3
"""Flatten NbThermo database.json to sequence+Tm table; quantify assay heterogeneity."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "raw" / "nbthermo" / "database.json"
OUT = ROOT / "interim" / "nbthermo_flattened.csv"
SUMMARY = ROOT / "interim" / "nbthermo_summary.json"


TM_METHODS = ["nanoDSF", "DSF (SYPRO)", "DSC", "Circular dichroism", "Other"]


def first_tm(tm_block: dict) -> tuple[float | None, str | None]:
    for method in TM_METHODS:
        block = (tm_block or {}).get(method) or {}
        val = block.get("value")
        if val is not None and val != "":
            try:
                return float(val), method
            except (TypeError, ValueError):
                continue
    return None, None


def main() -> None:
    data = json.loads(DB.read_text())
    rows = []
    for entry in data:
        seq_block = entry.get("Sequence") or {}
        # sequence may be nested
        seq = None
        if isinstance(seq_block, dict):
            for k in ("value", "AA", "sequence", "Sequence"):
                if seq_block.get(k):
                    seq = seq_block[k]
                    break
            if seq is None:
                # try nested
                for v in seq_block.values():
                    if isinstance(v, str) and len(v) > 20 and set(v.upper()) <= set("ACDEFGHIKLMNPQRSTVWY"):
                        seq = v
                        break
                    if isinstance(v, dict) and v.get("value"):
                        seq = v["value"]
                        break
        elif isinstance(seq_block, str):
            seq = seq_block

        tm_val, tm_method = first_tm(entry.get("Tm") or {})
        info = entry.get("Info") or {}
        name = ((info.get("Name") or {}).get("value")) if isinstance(info.get("Name"), dict) else info.get("Name")
        doi = ((info.get("Reference") or {}).get("DOI")) if isinstance(info.get("Reference"), dict) else None
        origin = entry.get("Origin")
        if isinstance(origin, dict):
            origin = origin.get("value") or json.dumps(origin)

        # count how many Tm methods populated
        tm_block = entry.get("Tm") or {}
        n_tm_methods = 0
        method_vals = {}
        for m in TM_METHODS:
            v = (tm_block.get(m) or {}).get("value")
            method_vals[m] = v
            if v is not None and v != "":
                n_tm_methods += 1

        rows.append(
            {
                "id": entry.get("id"),
                "name": name,
                "sequence": seq,
                "sequence_length": len(seq) if isinstance(seq, str) else None,
                "tm": tm_val,
                "tm_method_primary": tm_method,
                "n_tm_methods_reported": n_tm_methods,
                "doi": doi,
                "origin": origin,
                **{f"tm_{m}": method_vals[m] for m in TM_METHODS},
            }
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    with_seq = df["sequence"].notna() & (df["sequence"].astype(str).str.len() > 0)
    with_tm = df["tm"].notna()
    summary = {
        "n_json_entries": int(len(df)),
        "n_unique_ids": int(df["id"].nunique()),
        "n_with_sequence": int(with_seq.sum()),
        "n_with_any_tm": int(with_tm.sum()),
        "n_sequence_plus_tm": int((with_seq & with_tm).sum()),
        "n_unique_sequences": int(df.loc[with_seq, "sequence"].nunique()),
        "n_exact_duplicate_sequences": int(
            with_seq.sum() - df.loc[with_seq, "sequence"].nunique()
        ),
        "tm_method_counts": {
            m: int(df[f"tm_{m}"].notna().sum()) for m in TM_METHODS
        },
        "n_unique_dois": int(df["doi"].dropna().nunique()),
        "note": "Paper reports 564 Nbs; JSON dump contains 548 entries — count discrepancy recorded.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
