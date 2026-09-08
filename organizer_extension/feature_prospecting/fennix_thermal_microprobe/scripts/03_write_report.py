#!/usr/bin/env python3
"""Write THERMAL_MICROPROBE_REPORT_JA.md from Dev results + gate decision."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_thermal_microprobe")
RES = OUT / "results"


def main():
    meta = json.loads((RES / "DEV_RUN_META.json").read_text()) if (RES / "DEV_RUN_META.json").exists() else {}
    dec = json.loads((RES / "DEV_GATE_DECISION.json").read_text())
    df = pd.read_csv(RES / "THERMAL_MICROPROBE_DEV_RESULTS.csv")
    usable = json.loads((RES / "USABLE_DEV_IDS.json").read_text())

    def row(comp, fold, mode="FREE_ALPHA"):
        s = df[(df.comparison == comp) & (df.fold == fold) & (df.mode == mode) & (df.status == "OK")]
        return None if s.empty else s.iloc[0]

    lines = []
    lines.append("# FeNNix Thermal Microprobe — レポート")
    lines.append("")
    lines.append("解釈: short finite-temperature **thermal-response / relaxation** descriptor（MD sampling ではない）。")
    lines.append("")
    lines.append("## 冒頭 Q&A")
    lines.append("")
    wall = meta.get("mean_wall_s")
    lines.append(f"1. **実測 runtime / Ab？** {wall:.1f} s（median {meta.get('median_wall_s')} s）" if wall else "1. **実測 runtime / Ab？** （meta 参照）")
    lines.append(f"2. **usable Dev N？** {usable['usable_dev_n']}（欠: {usable['missing_dev']}）")
    tm_p = dec.get("tmapp_primary_delta")
    tm_s = dec.get("tmapp_shadow_delta")
    lines.append(f"3. **BASE+microprobe は TmApp Primary 改善？** {'YES' if tm_p and tm_p>0 else 'NO'}（Δ={tm_p}）")
    lines.append(f"4. **Shadow？** {'YES' if tm_s and tm_s>0 else 'NO'}（Δ={tm_s}）")
    lines.append(f"5. **BioEmu+ProteinMPNN 上乗せ？** Primary Δ={dec.get('recipe_primary_delta')} / Shadow Δ={dec.get('recipe_shadow_delta')}")
    lines.append(f"6. **HIC（endpoint SASA）？** Primary Δ={dec.get('hic_primary_delta')} / Shadow Δ={dec.get('hic_shadow_delta')}")
    lines.append(f"7. **BASE_FIXED_ALPHA？** P Δ={dec.get('tmapp_fixed_primary_delta')} / S Δ={dec.get('tmapp_fixed_shadow_delta')}")
    lines.append(f"8. **GO_TO_TEST or STOP？** **{dec.get('final_gate')}**（TmApp verdict={dec.get('tmapp_verdict')}）")
    lines.append("")
    lines.append("## Dev Simple TVT 表")
    lines.append("")
    lines.append(df.to_string(index=False))
    lines.append("")
    lines.append("## 決定メモ")
    lines.append("")
    lines.append("- Test 生成は Primary∩Shadow 正のときのみ。")
    lines.append("- 本ブロックは 250-step thermal microprobe；2–5 ps microdynamics は再開しない。")
    lines.append("")
    (OUT / "THERMAL_MICROPROBE_REPORT_JA.md").write_text("\n".join(lines) + "\n")
    # also copy results csv to top-level required name
    df.to_csv(OUT / "THERMAL_MICROPROBE_DEV_RESULTS.csv", index=False)
    print("wrote report", dec.get("final_gate"))


if __name__ == "__main__":
    main()
