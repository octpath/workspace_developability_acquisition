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
        "Codes are permanent (`experiment_code`). Descriptive `experiment_id` is human-readable only.",
        "",
        "Old bundle / organizer paths are read-only sources.",
        "",
    ]
    for _, r in full.sort_values(["family", "target", "experiment_code"]).iterrows():
        code = r["experiment_code"]
        eid = r["experiment_id"]
        cfg_path = ROOT / str(r["config_path"])
        src = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        lines += [
            f"## `{code}` — `{eid}`",
            "",
            f"- **legacy_experiment_code:** `{r.get('legacy_experiment_code', '')}`",
            f"- **source_model_id:** `{r['source_model_id']}`",
            f"- **feature_set_id:** `{r['feature_set_id']}`",
            f"- **source_recipe_id:** `{r['source_recipe_id']}`",
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
        sp = (src or {}).get("source_paths") or {}
        if sp:
            lines.append("- **config.source_paths:**")
            for k, v in sp.items():
                lines.append(f"  - {k}: `{v}`")
        lines.append("")
    out = ROOT / "assets" / "SOURCE_MAPPING.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
