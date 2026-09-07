#!/usr/bin/env python3
"""Verify crash-associated antibodies have complete B/C/M SUCCESS + QC artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CTX = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context")
IDS = ["ADI-47060", "ADI-47313", "ADI-45469", "ADI-45472"]
OUT = CTX / "cache/CRASH_ASSOCIATED_ABS_QC.json"


def main():
    feat = pd.read_csv(CTX / "cache/features/features_partial_all.csv")
    qc = pd.read_csv(CTX / "cache/features/qc_partial_all.csv")
    rows = []
    all_ok = True
    for ab in IDS:
        f = feat[feat.id == ab]
        q = qc[qc.id == ab]
        conds = {}
        for c in ("B", "C", "M"):
            sub = f[f.condition == c]
            ok = len(sub) and (sub.extraction_status == "SUCCESS").all()
            conds[c] = {
                "n": int(len(sub)),
                "success": bool(ok),
                "status": list(sub.extraction_status) if len(sub) else [],
            }
            if not ok:
                all_ok = False
        qinfo = []
        for _, r in q.iterrows():
            qinfo.append(
                {
                    "condition": r.get("condition"),
                    "converged": bool(r["converged"]) if "converged" in r and pd.notna(r["converged"]) else None,
                    "R1_F_rms": float(r["R1_F_rms"]) if "R1_F_rms" in r and pd.notna(r["R1_F_rms"]) else None,
                    "runtime_s": float(r["runtime_s"]) if "runtime_s" in r and pd.notna(r["runtime_s"]) else None,
                }
            )
        matched = (CTX / "cache/matched_fv" / f"{ab}_matched.pdb").exists()
        r1 = (CTX / "cache/r1" / f"{ab}_C_r1.pdb").exists()
        if not (matched and r1):
            all_ok = False
        rows.append(
            {
                "id": ab,
                "conditions": conds,
                "qc": qinfo,
                "matched_pdb": matched,
                "r1_pdb": r1,
                "verdict": "COMPLETE_PASS" if all(conds[c]["success"] for c in "BCM") and matched and r1 else "INCOMPLETE",
            }
        )
    doc = {
        "ids": IDS,
        "all_complete": all_ok,
        "note": "FIRE non-convergence alone does not fail extraction_status=SUCCESS; same QC schema as cohort.",
        "results": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
