#!/usr/bin/env python3
"""Replay EXP-T030 (sequence-only AbLingua FULL CONCAT) with current implementation.

Does not overwrite historical predictions. Writes results/EXP-T030_REPLAY_AUDIT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import mae  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
    load_annotations,
    load_dev_test,
    load_folds,
    load_residue_bundle,
)
from antibody_transformer.training import run_transformer_cv  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CONTROL = "EXP-T030"
VARIANT = "TMF2"
OUT = ROOT / "results" / "reproduction" / "EXP-T030"
AUDIT = ROOT / "results" / "EXP-T030_REPLAY_AUDIT.md"
TARGET = "TmApp"
# Established Transformer OOF reproduction tolerance (T037/T065 used exact/near-exact).
PRED_TOL = 1e-5
MAE_TOL = 1e-6


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def oof_to_csv(ids, pred, path: Path) -> None:
    pd.DataFrame({"id": ids, TARGET: pred}).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    hist_pred = ROOT / "experiments" / "predictions" / CONTROL
    assert hist_pred.exists(), "historical EXP-T030 predictions required"

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == CONTROL].iloc[0]
    hist = {
        "cv_primary_mae": float(row["cv_primary_mae"]),
        "cv_shadow_mae": float(row["cv_shadow_mae"]),
        "cv_mean_mae": float(row["cv_mean_mae"]),
        "cv_worst_mae": float(row["cv_worst_mae"]),
        "reproduction_status": str(row["reproduction_status"]),
    }

    print("=== EXP-T030 replay ===", flush=True)
    print("git:", git_rev(), flush=True)
    t0 = time.time()
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    summary = run_transformer_cv(
        target=TARGET,
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        variant_id=VARIANT,
        dev=dev,
        rb=rb,
        folds=folds,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        out_dir=OUT,
        quick=args.quick,
        recipe_id=None,
        plm_source="ablingua",
        pooling_mode="reg",
    )
    z = np.load(OUT / f"oof_{VARIANT}_{summary['config_hash']}.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    oof_to_csv(ids, z["primary_oof"], OUT / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], OUT / "oof_shadow.csv")

    hp = pd.read_csv(hist_pred / "oof_primary.csv")
    hs = pd.read_csv(hist_pred / "oof_shadow.csv")
    rp = pd.read_csv(OUT / "oof_primary.csv")
    rs = pd.read_csv(OUT / "oof_shadow.csv")
    for df in (hp, hs, rp, rs):
        df["id"] = df["id"].astype(str)
    hp = hp.set_index("id").loc[ids]
    hs = hs.set_index("id").loc[ids]
    rp = rp.set_index("id").loc[ids]
    rs = rs.set_index("id").loc[ids]
    col = TARGET if TARGET in hp.columns else [c for c in hp.columns if c != "id"][0]
    d_p = float(np.max(np.abs(hp[col].to_numpy(float) - rp[TARGET].to_numpy(float))))
    d_s = float(np.max(np.abs(hs[col].to_numpy(float) - rs[TARGET].to_numpy(float))))

    y = dev.set_index("id")[TARGET]
    recompute_p = mae(y.loc[ids].to_numpy(float), rp[TARGET].to_numpy(float))
    recompute_s = mae(y.loc[ids].to_numpy(float), rs[TARGET].to_numpy(float))

    ok = (
        d_p <= PRED_TOL
        and d_s <= PRED_TOL
        and abs(summary["primary_mae"] - hist["cv_primary_mae"]) <= MAE_TOL
        and abs(summary["shadow_mae"] - hist["cv_shadow_mae"]) <= MAE_TOL
    )
    verdict = "REPRODUCED" if ok else "FAIL_STOP"
    evidence = {
        "git_rev": git_rev(),
        "elapsed_sec": time.time() - t0,
        "quick": args.quick,
        "historical": hist,
        "replay": {
            "cv_primary_mae": summary["primary_mae"],
            "cv_shadow_mae": summary["shadow_mae"],
            "cv_mean_mae": summary["cv_mean_mae"],
            "cv_worst_mae": summary["cv_worst_mae"],
            "config_hash": summary["config_hash"],
            "recompute_primary_mae": recompute_p,
            "recompute_shadow_mae": recompute_s,
        },
        "max_pred_delta": {"primary": d_p, "shadow": d_s},
        "tolerances": {"pred": PRED_TOL, "mae": MAE_TOL},
        "verdict": verdict,
        "note": "Historical EXP-T030 predictions untouched; replay under results/reproduction/EXP-T030/",
    }
    (OUT / "reproduction_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    AUDIT.write_text(
        "\n".join(
            [
                "# EXP-T030 replay audit",
                "",
                f"- git: `{evidence['git_rev']}`",
                f"- verdict: **{verdict}**",
                "",
                "## Canonical (experiments.csv)",
                f"- Primary: {hist['cv_primary_mae']:.12f}",
                f"- Shadow: {hist['cv_shadow_mae']:.12f}",
                f"- prior status: {hist['reproduction_status']}",
                "",
                "## Replay (current AnnotatedTransformer, no fusion)",
                f"- Primary: {summary['primary_mae']:.12f}",
                f"- Shadow: {summary['shadow_mae']:.12f}",
                f"- config_hash: `{summary['config_hash']}`",
                "",
                "## Prediction deltas vs historical OOF",
                f"- primary max |Δ|: {d_p:.6e}",
                f"- shadow max |Δ|: {d_s:.6e}",
                f"- tolerances: pred≤{PRED_TOL}, mae≤{MAE_TOL}",
                "",
                f"## Reproducibility verdict: {verdict}",
                "",
            ]
        )
        + "\n"
    )
    print(json.dumps(evidence["replay"], indent=2), flush=True)
    print("max_pred_delta", evidence["max_pred_delta"], flush=True)
    print("VERDICT", verdict, flush=True)
    if not ok:
        raise SystemExit(1)

    # Promote registry status if policy allows
    if ok and str(row["reproduction_status"]) != "REPRODUCED":
        exp.loc[exp["experiment_code"] == CONTROL, "reproduction_status"] = "REPRODUCED"
        if "reproducibility_status_v2" in exp.columns:
            exp.loc[exp["experiment_code"] == CONTROL, "reproducibility_status_v2"] = "REPRODUCED"
        if "training_reproduction_attempted" in exp.columns:
            exp.loc[exp["experiment_code"] == CONTROL, "training_reproduction_attempted"] = True
        if "prediction_reproduction_max_delta" in exp.columns:
            exp.loc[exp["experiment_code"] == CONTROL, "prediction_reproduction_max_delta"] = max(d_p, d_s)
        exp.to_csv(ROOT / "results" / "experiments.csv", index=False)
        print("Promoted EXP-T030 reproduction_status -> REPRODUCED", flush=True)


if __name__ == "__main__":
    main()
