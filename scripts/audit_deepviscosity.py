#!/usr/bin/env python3
"""Audit DeepViscosity public materials; extract any recoverable partial labels."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "raw" / "deepviscosity" / "DeepViscosity"
OUT_LABELS = ROOT / "interim" / "deepviscosity_labels_partial.csv"
OUT_SEQ = ROOT / "interim" / "deepviscosity_sequences_partial.fasta"
OUT_AUDIT = ROOT / "interim" / "deepviscosity_join_audit.csv"
OUT_STATUS = ROOT / "interim" / "deepviscosity_status.json"


def extract_viscosities_from_notebook(nb_path: Path) -> pd.DataFrame:
    data = json.loads(nb_path.read_text())
    texts = []
    for cell in data.get("cells", []):
        for out in cell.get("outputs") or []:
            if "text" in out:
                t = out["text"]
                texts.append("".join(t) if isinstance(t, list) else str(t))
            if "data" in out and "text/plain" in out["data"]:
                t = out["data"]["text/plain"]
                texts.append("".join(t) if isinstance(t, list) else str(t))
    joined = "\n".join(texts)
    pairs = re.findall(r"\('?(mAb\d+)'?,\s*([0-9.]+)\)", joined)
    pairs += re.findall(r"'mAb':\s*'(mAb\d+)',\s*'Viscosity':\s*([0-9.]+)", joined)
    pairs += re.findall(
        r"'Reference_mAb':\s*'(mAb\d+)',\s*'Reference_Viscosity':\s*([0-9.]+)", joined
    )
    rows: dict[str, set[float]] = {}
    for m, v in pairs:
        rows.setdefault(m, set()).add(float(v))
    clean = []
    for m, vs in sorted(rows.items(), key=lambda x: int(x[0][3:])):
        clean.append(
            {
                "antibody_id": m,
                "viscosity_cP": list(vs)[0],
                "n_values_seen": len(vs),
                "conflict": len(vs) > 1,
                "label_source": "sequence_grouping.ipynb_outputs",
                "condition_note": "paper: 20 mM His-HCl pH6.0, 150 mg/mL, 25C (not re-verified per-row)",
            }
        )
    return pd.DataFrame(clean)


def fasta_to_dict(path: Path) -> dict[str, str]:
    seqs = {}
    name = None
    chunks = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(chunks)
            name = line[1:].strip().split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def main() -> None:
    ROOT.joinpath("interim").mkdir(parents=True, exist_ok=True)
    labels = extract_viscosities_from_notebook(REPO / "sequence_grouping.ipynb")
    labels.to_csv(OUT_LABELS, index=False)

    # Demo/example sequences shipped with predictor (NOT the 229 training set)
    h = fasta_to_dict(REPO / "seq_H.fasta")
    l = fasta_to_dict(REPO / "seq_L.fasta")
    demo = pd.read_csv(REPO / "DeepViscosity_input.csv")

    lines = []
    for _, r in demo.iterrows():
        name = r["Name"]
        lines.append(f">{name}_H\n{r['Heavy_Chain']}")
        lines.append(f">{name}_L\n{r['Light_Chain']}")
    OUT_SEQ.write_text("\n".join(lines) + "\n")

    # Join audit: notebook viscosities have anonymized mAb IDs; demo seqs are different set
    audit_rows = []
    for _, r in labels.iterrows():
        audit_rows.append(
            {
                "antibody_id": r["antibody_id"],
                "has_continuous_viscosity": True,
                "has_paired_sequence_in_repo": r["antibody_id"] in h and r["antibody_id"] in l,
                "join_confidence": "UNRESOLVED",
                "notes": "Viscosity recovered from notebook cluster outputs; training sequences proprietary (AZ_seq_vis.csv not in repo)",
            }
        )
    for name in demo["Name"]:
        audit_rows.append(
            {
                "antibody_id": name,
                "has_continuous_viscosity": name in set(labels["antibody_id"]),
                "has_paired_sequence_in_repo": True,
                "join_confidence": "EXACT_ID" if name in set(labels["antibody_id"]) else "SEQUENCE_ONLY_NO_LABEL",
                "notes": "Example/independent-demo sequences in DeepViscosity_input.csv (n=16)",
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_AUDIT, index=False)

    status = {
        "final_status": "PARTIAL_DATASET_RECONSTRUCTABLE",
        "detail_status": [
            "CONTINUOUS_TARGET_NOT_RECOVERED_FOR_FULL_229",
            "PARTIAL_CONTINUOUS_FROM_NOTEBOOK_OUTPUTS",
            "BINARY_TARGET_NOT_PUBLICLY_TABLED",
            "PUBLIC_SEQUENCE_LABEL_JOIN_NOT_RECOVERABLE_FOR_FULL_229",
        ],
        "n_continuous_partial_labels": int(len(labels)),
        "n_demo_paired_sequences": int(len(demo)),
        "n_high_confidence_joined_rows": int(
            ((audit["has_continuous_viscosity"]) & (audit["has_paired_sequence_in_repo"])).sum()
        ),
        "training_files_referenced_but_absent": [
            "data/AZ_seq_vis.csv",
            "data/AZ_DeepSP_features.csv",
            "data/AZ_seq_clustered.csv",
            "data/DeepVis_independent_testset.csv",
            "data/AZ_Apgar_others.csv",
        ],
        "paper_statement": "DV_mAb_229 datasets used for training and validation are proprietary and were not shared.",
        "formulation_condition_reported": "20 mM histidine-HCl, pH 6.0, 150 mg/mL, 25 C",
        "binary_threshold_cP": 20,
    }
    OUT_STATUS.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
