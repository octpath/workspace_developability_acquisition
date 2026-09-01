#!/usr/bin/env python3
"""Round 1 submission sanity audit + hash freeze (PART D)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FIN = ROOT / "virtual_participant/round1_finalization"
DEV = ROOT / "competition/data/distribution/dev.csv"
TEST = ROOT / "competition/data/distribution/test_features.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def qstats(s: pd.Series) -> dict:
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "sd": float(s.std()),
        "q05": float(s.quantile(0.05)),
        "q95": float(s.quantile(0.95)),
    }


def main():
    ts = datetime.now(timezone.utc).isoformat()
    commit = git_commit()
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    sub = pd.read_csv(FIN / "submissions/ROUND1_PRIMARY_submission.csv")

    checks = []
    checks.append(("162/162 IDs", len(sub) == 162))
    checks.append(("no duplicate", sub.id.duplicated().sum() == 0))
    checks.append(("train/test disjoint", len(set(sub.id) & set(dev.id)) == 0))
    checks.append(("TmApp finite", np.isfinite(sub.TmApp).all()))
    checks.append(("HIC finite", np.isfinite(sub.HIC).all()))
    checks.append(("no NaN", sub.isna().sum().sum() == 0))

    tm_stats = qstats(sub.TmApp)
    hic_stats = qstats(sub.HIC)
    dev_tm = qstats(dev.TmApp)
    dev_hic = qstats(dev.HIC)

    audit_md = f"""# Round 1 Test Prediction Sanity Audit

**状態:** `ROUND1_SUBMISSION_FROZEN_BEFORE_REVEAL`（本 audit 作成時点）  
**タイムスタンプ:** {ts}

## ID checks

"""
    for name, ok in checks:
        audit_md += f"- [{'x' if ok else ' '}] {name}\n"

    audit_md += f"""
## TmApp prediction (Test)

| stat | value |
|---|---:|
"""
    for k, v in tm_stats.items():
        audit_md += f"| {k} | {v:.4f} |\n"

    audit_md += f"""
## HIC prediction (Test)

| stat | value |
|---|---:|
"""
    for k, v in hic_stats.items():
        audit_md += f"| {k} | {v:.4f} |\n"

    audit_md += f"""
## Dev vs Test distribution comparison

| target | Dev mean | Test pred mean | mean shift | Dev SD | Test pred SD | SD ratio |
|---|---:|---:|---:|---:|---:|---:|
| TmApp | {dev_tm['mean']:.4f} | {tm_stats['mean']:.4f} | {tm_stats['mean']-dev_tm['mean']:.4f} | {dev_tm['sd']:.4f} | {tm_stats['sd']:.4f} | {tm_stats['sd']/dev_tm['sd']:.4f} |
| HIC | {dev_hic['mean']:.4f} | {hic_stats['mean']:.4f} | {hic_stats['mean']-dev_hic['mean']:.4f} | {dev_hic['sd']:.4f} | {hic_stats['sd']:.4f} | {hic_stats['sd']/dev_hic['sd']:.4f} |

## Notes

- distribution shift を観測しても model 変更は行わない（Round 1 protocol）。
- 本 audit 時点で organizer secret / Public / Private label は未参照。
- sanity audit: **{'PASS' if all(c[1] for c in checks) else 'FAIL'}**
"""
    (FIN / "ROUND1_TEST_PREDICTION_AUDIT_JA.md").write_text(audit_md)

    files_to_hash = [
        FIN / "submissions/ROUND1_PRIMARY_submission.csv",
        FIN / "submissions/ROUND1_DIAGNOSTIC_TM_CONSERVATIVE.csv",
        FIN / "submissions/ROUND1_DIAGNOSTIC_HIC_CONSERVATIVE.csv",
        FIN / "predictions/TmApp_PRIMARY_predictions.csv",
        FIN / "predictions/HIC_PRIMARY_predictions.csv",
        FIN / "round1_final_model_specs.json",
        FIN / "ROUND1_PRETEST_LOCK.json",
    ]
    hashes = {}
    hash_lines = []
    for p in files_to_hash:
        if p.exists():
            d = sha256(p)
            rel = str(p.relative_to(ROOT))
            hashes[rel] = d
            hash_lines.append(f"{d}  {rel}")

    (FIN / "ROUND1_SUBMISSION_HASHES.txt").write_text("\n".join(hash_lines) + "\n")

    freeze = {
        "timestamp": ts,
        "git_commit": commit,
        "state": "ROUND1_SUBMISSION_FROZEN_BEFORE_REVEAL",
        "primary_submission": "virtual_participant/round1_finalization/submissions/ROUND1_PRIMARY_submission.csv",
        "primary_submission_sha256": hashes.get(
            "virtual_participant/round1_finalization/submissions/ROUND1_PRIMARY_submission.csv"
        ),
        "diagnostic_files": {k: v for k, v in hashes.items() if "DIAGNOSTIC" in k or "predictions" in k},
        "all_hashes": hashes,
        "primary_model_ids": {
            "TmApp": "TmApp__META_performance__ridge_100.0",
            "HIC": "HIC__SIMPLE_blend_seq_surf_adv",
        },
        "test_row_count": 162,
        "sanity_audit_pass": all(c[1] for c in checks),
    }
    (FIN / "ROUND1_SUBMISSION_FREEZE.json").write_text(json.dumps(freeze, indent=2))
    print("FREEZE", freeze["primary_submission_sha256"])
    print("PASS", freeze["sanity_audit_pass"])


if __name__ == "__main__":
    main()
