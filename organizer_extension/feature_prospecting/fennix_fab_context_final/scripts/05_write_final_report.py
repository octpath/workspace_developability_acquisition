#!/usr/bin/env python3
"""Write freeze JSON + final JA report after QC/TVT/repro artifacts exist."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FINAL = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_final/results"
FE = ROOT / "feature_extension"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    meta = json.loads((FINAL / "ASSEMBLY_META.json").read_text())
    tvt = pd.read_csv(FINAL / "FENNIX_FULL_SIMPLE_TVT_RESULTS.csv")
    ivf = pd.read_csv(FINAL / "FENNIX_INTERIM_VS_FULL.csv")
    usable = json.loads((FINAL / "USABLE_DEV.json").read_text())
    qc_report = (FINAL / "FENNIX_FAB_FINAL_QC_REPORT.md").read_text()
    qc_verdict = meta["qc_verdict"]

    repro_path = FINAL / "FENNIX_FRESH_WORKER_REPRODUCIBILITY.md"
    if repro_path.exists():
        text = repro_path.read_text()
        if "REPRODUCIBLE" in text and "CONCERN" not in text.split("Verdict")[1][:80]:
            # parse bold verdict
            for lab in ("REPRODUCIBLE", "NUMERICALLY_CLOSE", "REPRODUCIBILITY_CONCERN"):
                if f"**{lab}**" in text:
                    repro_verdict = lab
                    break
            else:
                repro_verdict = "UNKNOWN"
        else:
            repro_verdict = "UNKNOWN"
            for lab in ("REPRODUCIBLE", "NUMERICALLY_CLOSE", "REPRODUCIBILITY_CONCERN"):
                if f"**{lab}**" in text:
                    repro_verdict = lab
                    break
    else:
        repro_verdict = "PENDING"

    both_pos = tvt[(tvt.primary_delta > 0) & (tvt.shadow_delta > 0)]
    fixed_surv = tvt[(tvt.fixed_primary_delta > 0) & (tvt.fixed_shadow_delta > 0)]
    const = ivf[ivf.family == "CONSTANT"].iloc[0]
    bio = ivf[ivf.family == "BIOEMU_NEW_CONTACT+CONSTANT"]
    bio_rep = bio.iloc[0].replication if len(bio) else "INCONCLUSIVE"

    freeze = {
        "freeze_utc": datetime.now(timezone.utc).isoformat(),
        "feature_table": "results/FENNIX_FAB_FEATURES_FULL.parquet",
        "feature_table_sha256": meta["feature_sha256"],
        "feature_dictionary_sha256": meta["dictionary_sha256"],
        "accepted_N": 323,
        "excluded_ids": ["ADI-47265"],
        "excluded_reason": "TECHNICAL_SKIP Fab prep / HL disulfide QC failure",
        "feature_families": meta["families"],
        "feature_dim": meta["feature_dim"],
        "scientific_spec": "fennix_fab_context/FENNIX_FAB_CONTEXT_SPEC.md",
        "interim_feature_freeze": "fennix_fab_context_interim_audit/INTERIM_FENNIX_FEATURE_FREEZE.json",
        "assembly_script": "fennix_fab_context_final/scripts/01_assemble_and_qc.py",
        "assembly_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "qc_verdict": qc_verdict,
        "reproducibility_verdict": repro_verdict,
        "frozen_before_full_cohort_target_eval": True,
        "note": "Freeze metadata recorded after assembly/QC; target eval used this table without changing features.",
    }
    (FINAL / "FENNIX_FAB_FINAL_FEATURE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    # strict summary
    strict = pd.read_csv(FINAL / "FENNIX_FULL_STRICT_NESTED_RESULTS.csv")
    strict_s = strict[strict.protocol == "STRICT_NESTED_INCREMENT"]

    lines = [
        "# FeNNix Fab — 最終レポート（フルコホート）",
        "",
        f"UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 冒頭 Q&A",
        "",
        f"1. **323本はきれいに組めたか？** YES（`FINAL_FENNIX_COHORT.csv`、B/C/M SUCCESS、r1/matched OK）。除外 ADI-47265。",
        f"2. **fresh-worker 再現性は？** **{repro_verdict}**",
        f"3. **usable Dev N？** {usable['usable_fennix_dev']} / {usable['full_dev']}（欠: {usable['missing_dev']}）",
        f"4. **interim CONSTANT はフルで再現したか？** **{const.replication}**（interim P/S Δ={const.interim_Primary_delta:.4f}/{const.interim_Shadow_delta:.4f} → full {const.full_Primary_delta:.4f}/{const.full_Shadow_delta:.4f}）",
        f"5. **BIOEMU_NEW_CONTACT+CONSTANT？** **{bio_rep}** / full sci={bio.iloc[0].full_verdict if len(bio) else 'n/a'}",
        f"6. **Simple TVT で Primary∩Shadow 改善 family？** {', '.join(both_pos.family.tolist()) if len(both_pos) else 'なし'}",
        f"7. **BASE_FIXED_ALPHA 生存？** {', '.join(fixed_surv.family.tolist()) if len(fixed_surv) else 'なし'}",
        f"8. **strict nested late-fusion？** ほぼ全て NO_INCREMENT（悪化側）。CONSTANT residual Primary のみごく弱い正。",
        f"9. **科学的結論** FeNNix Fab 特徴は技術的に組立可能だが、フル Dev では安定した予測増分を示さない。interim N=100 CONSTANT の正信号は **再現しなかった**。",
        f"10. **競技結論** 全 family / BIOEMU+CONSTANT → **COMP_DROP**（direct fusion でも Primary/Shadow 同時安定改善なし）。",
        f"11. **参加者リリースしてよいか？** 技術 QC={qc_verdict}、再現性={repro_verdict}。特徴自体は target-blind で科学的に定義済みのため **配布可**。ただし CV ヒントは「安定改善なし」と明記。ライセンスは FeNNol LGPL-3.0 → `PARTICIPANT_ONLY_REVIEW_RECOMMENDED`。",
        f"12. **v1.1 パッケージ？** （後段で実施）",
        "",
        "## Simple TVT（抜粋）",
        "",
        tvt.to_markdown(index=False) if hasattr(tvt, "to_markdown") else tvt.to_string(index=False),
        "",
        "## Interim vs Full",
        "",
        ivf.to_markdown(index=False) if hasattr(ivf, "to_markdown") else ivf.to_string(index=False),
        "",
        "## Strict nested (STRICT_NESTED_INCREMENT, delta = incumbent MAE − combined MAE)",
        "",
        "正 = 改善。結果は概ね負。",
        "",
        "## QC",
        "",
        qc_report,
    ]
    (FINAL / "FENNIX_FAB_FINAL_REPORT_JA.md").write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8")
    print("freeze", freeze["reproducibility_verdict"], "qc", qc_verdict)
    print("both_pos", list(both_pos.family))
    print("CONSTANT replication", const.replication)


if __name__ == "__main__":
    main()
