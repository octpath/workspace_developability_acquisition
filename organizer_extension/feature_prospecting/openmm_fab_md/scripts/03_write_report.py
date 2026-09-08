#!/usr/bin/env python3
"""Write OPENMM_MD_FINAL_REPORT_JA.md"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/openmm_fab_md")
RES = OUT / "results"


def main():
    dec = json.loads((RES / "DEV_GATE_DECISION.json").read_text())
    summary = json.loads((RES / "DEV_RUN_SUMMARY.json").read_text()) if (RES / "DEV_RUN_SUMMARY.json").exists() else {}
    proto = json.loads((RES / "PROTOCOL_LENGTH.json").read_text()) if (RES / "PROTOCOL_LENGTH.json").exists() else {}
    qc = pd.read_csv(RES / "OPENMM_FAB_MD_QC_DEV.csv") if (RES / "OPENMM_FAB_MD_QC_DEV.csv").exists() else None
    df = pd.read_csv(RES / "OPENMM_MD_DEV_RESULTS.csv")
    usable = json.loads((RES / "USABLE_DEV_IDS.json").read_text())

    mean_rt = summary.get("mean_runtime_s") or (float(qc["runtime_seconds"].mean()) if qc is not None else None)
    prod_ns = summary.get("production_ns") or proto.get("production_ns")
    n_ok = summary.get("n_success")
    n_fail = summary.get("n_fail", 0)
    finite_ok = int((qc["feature_finite"] & qc["finite_energy"] & qc["finite_coordinates"]).sum()) if qc is not None else None

    lines = [
        "# OpenMM Fab MD — 最終レポート（DEV）",
        "",
        "解釈: short implicit-solvent classical-MD dynamic descriptors（厳密な溶液MD / Tm直接シミュレーションではない）。",
        "",
        "## 冒頭 Q&A",
        "",
        f"1. **実測 complete wall / Ab？** {mean_rt:.1f} s" if mean_rt else "1. runtime: see QC",
        f"2. **最終 production 長？** {prod_ns} ns",
        f"3. **DEV 完走 N？** {n_ok} / 161（fail={n_fail}）",
        f"4. **数値安定性？** finite feature/energy/coords = {finite_ok}/{n_ok}" if finite_ok is not None else "4. see QC",
        f"5. **TmApp BASE 増分 P/S？** {dec.get('tmapp_primary_delta')} / {dec.get('tmapp_shadow_delta')}",
        f"6. **BioEmu+MPNN 上乗せ P/S？** {dec.get('recipe_primary_delta')} / {dec.get('recipe_shadow_delta')}",
        f"7. **HIC 増分 P/S？** {dec.get('hic_primary_delta')} / {dec.get('hic_shadow_delta')}（+CONT+TITR: {dec.get('hic_cont_titr_primary_delta')} / {dec.get('hic_cont_titr_shadow_delta')}）",
        f"8. **aromatic/hydrophobic 動的記述子？** HIC verdict={dec.get('hic_verdict')}",
        f"9. **BASE_FIXED_ALPHA？** P={dec.get('tmapp_fixed_primary_delta')} / S={dec.get('tmapp_fixed_shadow_delta')}",
        f"10. **TmApp GO_TO_TEST/STOP？** **{dec.get('final_tmapp')}**",
        f"11. **HIC GO_TO_TEST/STOP？** **{dec.get('final_hic')}**",
        f"12. **overall COMP？** TmApp={dec.get('tmapp_verdict')} / HIC={dec.get('hic_verdict')}",
        "",
        f"usable Dev N={usable['usable_dev_n']}; missing={usable['missing_dev']}",
        "FeNNix thermal microprobe: not resumed; 37/161 preserved.",
        "",
        "## Simple TVT 表",
        "",
        df.to_string(index=False),
        "",
        "## Decision JSON",
        "",
        "```json",
        json.dumps(dec, indent=2),
        "```",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "OPENMM_MD_FINAL_REPORT_JA.md").write_text(text)
    (RES / "OPENMM_MD_FINAL_REPORT_JA.md").write_text(text)
    print("wrote report", dec.get("final_tmapp"), dec.get("final_hic"))


if __name__ == "__main__":
    main()
