#!/usr/bin/env python3
"""Fresh-process reproducibility audit for crash-associated Abs.

Backs up production r1/matched artifacts for selected IDs, runs FeNNix with
--audit-tag (features written to audit dir only), restores production artifacts,
then compares audit features vs production features_partial_all.

Does NOT leave production caches permanently modified.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
CACHE = CTX / "cache"
OUT = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_final/results"
FENNOL = ROOT / "organizer_extension/feature_prospecting/foundation_stability/envs/fennol/bin/python"
SCRIPT = CTX / "scripts/02_fennix_fab_curvature.py"
TAG = "fresh_repro_final"
IDS = ["ADI-47313", "ADI-47060", "ADI-45469"]  # include prior stack-smash ID


def backup_paths(ab_id: str, bak: Path) -> list[Path]:
    paths = []
    for c in "BCM":
        p = CACHE / "r1" / f"{ab_id}_{c}.npz"
        if p.exists():
            paths.append(p)
    for p in [
        CACHE / "r1" / f"{ab_id}_C_r1.pdb",
        CACHE / "matched_fv" / f"{ab_id}_matched.pdb",
    ]:
        if p.exists():
            paths.append(p)
    for p in paths:
        dst = bak / p.relative_to(CACHE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    return paths


def restore(bak: Path):
    if not bak.exists():
        return
    for p in bak.rglob("*"):
        if p.is_file():
            dst = CACHE / p.relative_to(bak)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)


def compare(prod: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    prod = prod.set_index(["id", "condition"])
    fresh = fresh.set_index(["id", "condition"])
    common = sorted(set(prod.index) & set(fresh.index))
    num_cols = [
        c
        for c in prod.columns
        if c in fresh.columns and pd.api.types.is_numeric_dtype(prod[c]) and c not in ("n_sites",)
    ]
    rows = []
    for idx in common:
        for c in num_cols:
            a, b = float(prod.loc[idx, c]), float(fresh.loc[idx, c])
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            abs_d = abs(a - b)
            rel = abs_d / (abs(a) + 1e-8)
            rows.append(
                {
                    "id": idx[0],
                    "condition": idx[1],
                    "feature": c,
                    "prod": a,
                    "fresh": b,
                    "abs_diff": abs_d,
                    "rel_diff": rel,
                }
            )
    df = pd.DataFrame(rows)
    # family-level summaries by condition aggregate
    summary = []
    for (ab, cond), g in df.groupby(["id", "condition"]):
        a = g["prod"].to_numpy(dtype=float)
        b = g["fresh"].to_numpy(dtype=float)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = float("nan")
        summary.append(
            {
                "id": ab,
                "condition": cond,
                "n_features": len(g),
                "max_abs_diff": float(g.abs_diff.max()),
                "median_rel_diff": float(g.rel_diff.median()),
                "corr_coef": corr,
            }
        )
    summ = pd.DataFrame(summary)
    # classify
    max_abs = float(df.abs_diff.max()) if len(df) else np.inf
    med_rel = float(df.rel_diff.median()) if len(df) else np.inf
    min_corr = float(summ["corr_coef"].min()) if len(summ) and summ["corr_coef"].notna().any() else 0.0
    if max_abs < 1e-5 and med_rel < 1e-6:
        verdict = "REPRODUCIBLE"
    elif min_corr > 0.999 and med_rel < 0.01:
        # FIRE/R1 on B can move a few K aggregates by O(1e-1) while remaining highly correlated
        verdict = "NUMERICALLY_CLOSE"
    else:
        verdict = "REPRODUCIBILITY_CONCERN"
    return df, summ, verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bak = CACHE / f"cpu_cuda_audit/{TAG}/_prod_backup"
    if bak.exists():
        shutil.rmtree(bak)
    bak.mkdir(parents=True)

    prod = pd.read_csv(CACHE / "features/features_partial_all.csv")
    prod = prod[prod.id.isin(IDS) & (prod.extraction_status == "SUCCESS")].copy()

    print("backing up production r1/matched for", IDS, flush=True)
    for ab in IDS:
        backup_paths(ab, bak)

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    # keep <=4 cores for side work? user said fennix done - use modest affinity
    cmd = [
        "taskset",
        "-c",
        "16-23",
        str(FENNOL),
        str(SCRIPT),
        "--ids",
        *IDS,
        "--audit-tag",
        TAG,
        "--fresh",
    ]
    print("RUNNING", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    print("restore production artifacts", flush=True)
    restore(bak)
    if rc != 0:
        print("FeNNix audit exit", rc, file=sys.stderr)
        # still try compare if partial
    fresh_path = CACHE / f"cpu_cuda_audit/{TAG}/features.csv"
    if not fresh_path.exists():
        raise SystemExit(f"missing fresh features {fresh_path}")
    fresh = pd.read_csv(fresh_path)
    df, summ, verdict = compare(prod, fresh)
    df.to_csv(OUT / "FENNIX_FRESH_WORKER_REPRODUCIBILITY.csv", index=False)
    summ.to_csv(OUT / "FENNIX_FRESH_WORKER_REPRO_SUMMARY.csv", index=False)
    lines = [
        "# FeNNix fresh-worker reproducibility",
        "",
        f"UTC: {datetime.now(timezone.utc).isoformat()}",
        f"IDs: {', '.join(IDS)}",
        f"audit-tag: {TAG}",
        f"exit_code: {rc}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        summ.to_markdown(index=False) if hasattr(summ, "to_markdown") else summ.to_string(index=False),
        "",
        "Production r1/matched artifacts were restored after the audit run.",
        "Accepted production features are NOT replaced by fresh runs.",
    ]
    (OUT / "FENNIX_FRESH_WORKER_REPRODUCIBILITY.md").write_text("\n".join(map(str, lines)) + "\n")
    print("VERDICT", verdict)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
