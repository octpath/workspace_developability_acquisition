#!/usr/bin/env python3
"""Register EXP-T030-REPLAY-001 as CONTEMPORARY_MATCHED_CONTROL and full-Dev Test.

Does NOT overwrite historical EXP-T030 artifacts/scores.
Does NOT issue a new EXP code.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import file_sha256, load_solution, mae  # noqa: E402
from antibody_transformer.data import load_dev_test, load_residue_bundle  # noqa: E402
from antibody_transformer.training import full_dev_transformer_predict  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_ID = "EXP-T030-REPLAY-001"
SRC = ROOT / "results" / "reproduction" / "EXP-T030"
DST = ROOT / "experiments" / "replays" / "EXP-T030" / RUN_ID
RUNS_CSV = ROOT / "results" / "experiment_runs.csv"
AUDIT = ROOT / "results" / "EXP-T030_REPLAY_AUDIT.md"
TARGET = "TmApp"
HIST_CODE = "EXP-T030"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def ensure_oof_copied() -> dict:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("oof_primary.csv", "oof_shadow.csv"):
        src = SRC / name
        if not src.exists():
            raise SystemExit(f"missing replay OOF: {src}")
        shutil.copy2(src, DST / name)
    cache = SRC / "cache_TMF2_0814cb40363d8eef.json"
    best_epochs = {}
    if cache.exists():
        best_epochs = json.loads(cache.read_text()).get("best_epochs_primary", {})
        shutil.copy2(cache, DST / "cv_cache.json")
    return best_epochs


def score_test(pred: pd.DataFrame) -> dict:
    sol = load_solution()
    if sol is None:
        raise SystemExit("solution.csv required for Test scoring")
    sol = sol.set_index("id")
    p = pred.copy()
    p["id"] = p["id"].astype(str)
    p = p.set_index("id")
    y = sol["TmApp"]
    pub_ids = sol.index[sol["is_public"].astype(bool)].tolist()
    priv_ids = sol.index[sol["is_private"].astype(bool)].tolist()
    pub_ids = [i for i in pub_ids if i in p.index]
    priv_ids = [i for i in priv_ids if i in p.index]
    ids = sorted(set(pub_ids) | set(priv_ids))
    return {
        "public_mae": float(mae(y.loc[pub_ids].to_numpy(float), p.loc[pub_ids, TARGET].to_numpy(float))),
        "private_mae": float(mae(y.loc[priv_ids].to_numpy(float), p.loc[priv_ids, TARGET].to_numpy(float))),
        "overall_mae": float(mae(y.loc[ids].to_numpy(float), p.loc[ids, TARGET].to_numpy(float))),
    }


def run_full_dev(best_epochs: dict) -> pd.DataFrame:
    print("=== contemporary T030 full-Dev Test ===", flush=True)
    # Prefer cache best epochs; fall back to historical ADVANCED results
    if not best_epochs:
        best_epochs = {"101": [18, 6, 11, 8, 20], "202": [14, 10, 36, 9, 20], "303": [11, 8, 8, 8, 13]}
    # normalize keys to str
    be = {str(k): list(v) for k, v in best_epochs.items()}
    ckpt = DST / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    pred = full_dev_transformer_predict(
        target=TARGET,
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        best_epochs_primary=be,
        dev=dev,
        test=test,
        rb=rb,
        device="cuda" if torch.cuda.is_available() else "cpu",
        recipe_id=None,
        plm_source="ablingua",
        pooling_mode="reg",
        joint_hl_single_reg=False,
        use_ca_distance_bias=False,
        checkpoint_dir=ckpt,
    )
    if "prediction" in pred.columns and TARGET not in pred.columns:
        pred = pred.rename(columns={"prediction": TARGET})
    out = DST / "test.csv"
    pred[["id", TARGET]].to_csv(out, index=False)
    return pred


def append_run_row(row: dict) -> None:
    cols = [
        "run_id",
        "experiment_code",
        "role",
        "source_git_commit",
        "runtime_git_commit",
        "environment",
        "gpu",
        "prediction_primary_path",
        "prediction_shadow_path",
        "prediction_test_path",
        "cv_primary_mae",
        "cv_shadow_mae",
        "public_mae",
        "private_mae",
        "overall_mae",
        "historical_match",
        "max_prediction_delta_vs_historical",
        "notes",
    ]
    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)
        df = df[df["run_id"] != RUN_ID]
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        row.setdefault(c, "")
    df = pd.concat([df, pd.DataFrame([{c: row.get(c, "") for c in cols}])], ignore_index=True)
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS_CSV, index=False)


def write_manifest(best_epochs: dict, scores: dict, test_scores: dict, deltas: dict) -> None:
    man = {
        "experiment_code": HIST_CODE,
        "run_id": RUN_ID,
        "control_role": "CONTEMPORARY_MATCHED_CONTROL",
        "git_commit": git_rev(),
        "python_version": str(platform.python_version()),
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "gpu": str(gpu_name()),
        "folds": "canonical_simple_tvt_primary_shadow",
        "seeds": [101, 202, 303],
        "training_hparams": {
            "optimizer": "AdamW",
            "lr": 0.0003,
            "weight_decay": 0.01,
            "batch_size": 16,
            "max_epochs": 300,
            "patience": 30,
            "loss": "SmoothL1Loss(beta=0.5)",
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "ff_dim": 256,
            "dropout": 0.2,
        },
        "best_epochs_primary": best_epochs,
        "prediction_sha256": {
            "oof_primary.csv": file_sha256(DST / "oof_primary.csv"),
            "oof_shadow.csv": file_sha256(DST / "oof_shadow.csv"),
            "test.csv": file_sha256(DST / "test.csv") if (DST / "test.csv").exists() else None,
        },
        "cv_scores": scores,
        "test_scores": test_scores,
        "vs_historical": deltas,
        "HISTORICAL_REPRODUCTION": "FAIL",
        "CONTEMPORARY_MATCHED_CONTROL": "ACCEPTED",
        "note": "Do not treat as REPRODUCED historical EXP-T030",
    }
    # Force plain JSON-serializable types for YAML safety
    man = json.loads(json.dumps(man, default=str))
    (DST / "run_manifest.yaml").write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")


def update_audit(scores: dict, test_scores: dict, deltas: dict) -> None:
    AUDIT.write_text(
        "\n".join(
            [
                "# EXP-T030 replay audit",
                "",
                f"- git: `{git_rev()}`",
                f"- run_id: `{RUN_ID}`",
                "- HISTORICAL_REPRODUCTION: **FAIL**",
                "- CONTEMPORARY_MATCHED_CONTROL: **ACCEPTED**",
                "",
                "## Historical canonical EXP-T030 (unchanged)",
                "- Primary: 3.317769289998",
                "- Shadow: 3.214610237153",
                "- reproducibility_status: RESULT_VERIFIED (not promoted to REPRODUCED)",
                "",
                "## Contemporary matched control (EXP-T030-REPLAY-001)",
                f"- Primary: {scores['cv_primary_mae']:.12f}",
                f"- Shadow: {scores['cv_shadow_mae']:.12f}",
                f"- Public: {test_scores['public_mae']:.12f}",
                f"- Private: {test_scores['private_mae']:.12f}",
                f"- Overall: {test_scores['overall_mae']:.12f}",
                f"- primary max |Δ| vs historical OOF: {deltas['primary_max_abs']:.6e}",
                f"- shadow max |Δ| vs historical OOF: {deltas['shadow_max_abs']:.6e}",
                "",
                "## Conclusion",
                "",
                "Historical T030 is NOT reproduced under the current environment.",
                "The contemporary run is accepted as the matched scientific control for T068/T069.",
                "Historical predictions/scores/config remain untouched.",
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    # Verify historical T030 not mutated
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == HIST_CODE].iloc[0]
    assert abs(float(row["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
    assert str(row["reproduction_status"]) == "RESULT_VERIFIED"

    best_epochs = ensure_oof_copied()
    ev = json.loads((SRC / "reproduction_evidence.json").read_text())
    scores = {
        "cv_primary_mae": float(ev["replay"]["cv_primary_mae"]),
        "cv_shadow_mae": float(ev["replay"]["cv_shadow_mae"]),
    }
    # deltas vs historical predictions
    hp = pd.read_csv(ROOT / "experiments" / "predictions" / HIST_CODE / "oof_primary.csv")
    hs = pd.read_csv(ROOT / "experiments" / "predictions" / HIST_CODE / "oof_shadow.csv")
    rp = pd.read_csv(DST / "oof_primary.csv")
    rs = pd.read_csv(DST / "oof_shadow.csv")
    for d in (hp, hs, rp, rs):
        d["id"] = d["id"].astype(str)
    ids = rp["id"].tolist()
    hp = hp.set_index("id").loc[ids]
    hs = hs.set_index("id").loc[ids]
    rp_i = rp.set_index("id").loc[ids]
    rs_i = rs.set_index("id").loc[ids]
    col = TARGET if TARGET in hp.columns else [c for c in hp.columns if c != "id"][0]
    deltas = {
        "primary_max_abs": float(np.max(np.abs(hp[col].to_numpy(float) - rp_i[TARGET].to_numpy(float)))),
        "shadow_max_abs": float(np.max(np.abs(hs[col].to_numpy(float) - rs_i[TARGET].to_numpy(float)))),
    }

    if not (DST / "test.csv").exists() or "prediction" in pd.read_csv(DST / "test.csv", nrows=1).columns:
        # regenerate if missing or legacy column name
        if (DST / "test.csv").exists():
            (DST / "test.csv").unlink()
        pred = run_full_dev(best_epochs)
    else:
        pred = pd.read_csv(DST / "test.csv")
    if "prediction" in pred.columns and TARGET not in pred.columns:
        pred = pred.rename(columns={"prediction": TARGET})
    test_scores = score_test(pred)

    write_manifest(best_epochs, scores, test_scores, deltas)
    append_run_row(
        {
            "run_id": RUN_ID,
            "experiment_code": HIST_CODE,
            "role": "CONTEMPORARY_MATCHED_CONTROL",
            "source_git_commit": "historical_TMF2_a1022f580b4768c0",
            "runtime_git_commit": git_rev(),
            "environment": f"python={platform.python_version()};torch={torch.__version__};cuda={torch.version.cuda}",
            "gpu": gpu_name(),
            "prediction_primary_path": f"experiments/replays/EXP-T030/{RUN_ID}/oof_primary.csv",
            "prediction_shadow_path": f"experiments/replays/EXP-T030/{RUN_ID}/oof_shadow.csv",
            "prediction_test_path": f"experiments/replays/EXP-T030/{RUN_ID}/test.csv",
            "cv_primary_mae": scores["cv_primary_mae"],
            "cv_shadow_mae": scores["cv_shadow_mae"],
            "public_mae": test_scores["public_mae"],
            "private_mae": test_scores["private_mae"],
            "overall_mae": test_scores["overall_mae"],
            "historical_match": "NO",
            "max_prediction_delta_vs_historical": max(deltas["primary_max_abs"], deltas["shadow_max_abs"]),
            "notes": "Contemporary matched control for T068/T069; HISTORICAL_REPRODUCTION=FAIL; do not overwrite historical T030",
        }
    )
    update_audit(scores, test_scores, deltas)
    print(json.dumps({"run_id": RUN_ID, **scores, **test_scores, **deltas}, indent=2), flush=True)
    print("DONE register contemporary T030 control", flush=True)


if __name__ == "__main__":
    main()
