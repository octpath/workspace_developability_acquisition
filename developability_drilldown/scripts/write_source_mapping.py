#!/usr/bin/env python3
"""Generate assets/SOURCE_MAPPING.md from FULL experiment configs + registry."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import ROOT  # noqa: E402


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "experiments.csv")
    full = df[df["artifact_status"].isin(["FULL", "INCONSISTENT"])]
    lines = [
        "# Source mapping (FULL experiments)",
        "",
        "Old bundle / organizer paths are read-only sources. New artifacts live under `developability_drilldown/`.",
        "",
    ]
    for _, r in full.sort_values(["family", "target", "experiment_id"]).iterrows():
        eid = r["experiment_id"]
        cfg_path = ROOT / str(r["config_path"])
        src = {}
        if cfg_path.exists():
            src = yaml.safe_load(cfg_path.read_text()) or {}
        lines += [
            f"## `{eid}`",
            "",
            f"- **source_model_id:** `{r['source_model_id']}`",
            f"- **target / family:** {r['target']} / {r['family']}",
            f"- **artifact_status:** {r['artifact_status']}",
            f"- **score_source:** `{r['score_source']}`",
            f"- **prediction_source:** `{r['prediction_source']}`",
            f"- **feature_source:** `{r['feature_source']}`",
            f"- **config:** `{r['config_path']}`",
            f"- **feature parquet:** `{r['feature_path']}`",
            f"- **oof_primary:** `{r['oof_primary_path']}`",
            f"- **oof_shadow:** `{r['oof_shadow_path']}`",
            f"- **test prediction:** `{r['test_prediction_path']}`",
        ]
        sp = src.get("source_paths") or {}
        if sp:
            lines.append("- **config.source_paths:**")
            for k, v in sp.items():
                lines.append(f"  - {k}: `{v}`")
        if r["family"] == "XGBOOST":
            lines.append(
                "- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; "
                "otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search)."
            )
        lines.append("")
    out = ROOT / "assets" / "SOURCE_MAPPING.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
