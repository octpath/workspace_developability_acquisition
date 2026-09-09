#!/usr/bin/env python3
"""Rebuild / normalize XGBoost OOF under frozen advanced-suite protocol.

Artifact completion only — no hyperparameter search beyond the original
per-fold early-stopping preset selection encoded in advanced_models.cv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BUNDLE = REPO / "top_models_feature_bundle"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BUNDLE))

from _lib import (  # noqa: E402
    XGB_MAP,
    XGB_RECIPE,
    XGB_SOURCE_BY_EXP,
    mae,
    normalize_prediction_csv,
)
from experiment_codes import id_to_code  # noqa: E402

from advanced_models.data import load_dev_test, load_folds, tvt_split  # noqa: E402
from advanced_models.features import build_recipe_parts, preprocess_parts  # noqa: E402
from advanced_models.models.xgboost_model import fit_xgb_early, fit_xgb_rounds  # noqa: E402
from advanced_models.config import load_presets  # noqa: E402


EXISTING_NPZ = {
    "XGB_TM_BIOEMU_MPNN": BUNDLE
    / "results"
    / "cross_family_ensemble"
    / "xgb_oof_TmApp_TM_BASE_BIOEMU_MPNN__RIDGE.npz",
    "XGB_HIC_CONTINUOUS_SURFACE": BUNDLE
    / "results"
    / "cross_family_ensemble"
    / "xgb_oof_HIC_HIC_ARO_CONTINUOUS_SURFACE__LASSO.npz",
}


def target_of(experiment_id: str) -> str:
    return "TmApp" if "_TM_" in experiment_id else "HIC"


def write_oof_csvs(experiment_id: str, target: str, ids: list[str], primary, shadow) -> None:
    code = id_to_code(experiment_id)
    dest = ROOT / "experiments" / "predictions" / code
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, target: np.asarray(primary, float)}).to_csv(
        dest / "oof_primary.csv", index=False
    )
    pd.DataFrame({"id": ids, target: np.asarray(shadow, float)}).to_csv(
        dest / "oof_shadow.csv", index=False
    )


def from_npz(experiment_id: str, path: Path, dev: pd.DataFrame) -> dict:
    target = target_of(experiment_id)
    z = np.load(path, allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    primary = z["primary_oof"]
    shadow = z["shadow_oof"]
    # align to current dev order
    ymap = dict(zip(dev["id"].astype(str), dev[target].astype(float)))
    order = dev["id"].astype(str).tolist()
    pser = pd.Series(primary, index=ids).reindex(order)
    sser = pd.Series(shadow, index=ids).reindex(order)
    if pser.isna().any() or sser.isna().any():
        raise RuntimeError(f"npz OOF misaligned for {experiment_id}")
    write_oof_csvs(experiment_id, target, order, pser.to_numpy(), sser.to_numpy())
    y = np.asarray([ymap[i] for i in order], float)
    return {
        "experiment_id": experiment_id,
        "source": "tracked_npz",
        "primary_reconstructed": mae(y, pser.to_numpy()),
        "shadow_reconstructed": mae(y, sser.to_numpy()),
    }


def run_oof_cv(experiment_id: str, device: str, dev: pd.DataFrame, folds) -> dict:
    """Same protocol as advanced_models.cv.run_xgboost_cv, but persist OOF."""
    target = target_of(experiment_id)
    recipe_id = XGB_RECIPE[experiment_id]
    presets = load_presets()
    preset_names = list(presets["xgboost"]["presets"].keys())
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
    dev_ids = dev["id"].astype(str).tolist()
    parts = build_recipe_parts(recipe_id, dev_ids)

    oofs = {}
    rotations = {}
    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        oof = pd.Series(index=dev_ids, dtype=float)
        chosen = []
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            Xtr, Xva, Xte = preprocess_parts(parts, tr, [va, te])
            ytr = np.asarray([y_map[a] for a in tr], float)
            yva = np.asarray([y_map[a] for a in va], float)
            best_name, best_val, best_iter = None, float("inf"), 0
            for pname in preset_names:
                _, bit, vmae = fit_xgb_early(Xtr, ytr, Xva, yva, pname, device=device)
                if vmae < best_val - 1e-15 or (
                    abs(vmae - best_val) <= 1e-15 and (best_name is None or pname < best_name)
                ):
                    best_val, best_name, best_iter = vmae, pname, bit
            Xtv, Xte2 = preprocess_parts(parts, tr + va, [te])
            ytv = np.asarray([y_map[a] for a in tr + va], float)
            model = fit_xgb_rounds(Xtv, ytv, best_name, best_iter + 1, device=device)
            oof.loc[te] = model.predict(Xte2)
            chosen.append({"preset": best_name, "best_iteration": int(best_iter), "val_mae": float(best_val)})
        oofs[scheme_name] = oof.loc[dev_ids].to_numpy(float)
        rotations[scheme_name] = chosen

    write_oof_csvs(experiment_id, target, dev_ids, oofs["primary"], oofs["shadow"])
    y = np.asarray([y_map[a] for a in dev_ids], float)
    summary = {
        "experiment_id": experiment_id,
        "source": "frozen_protocol_rerun",
        "primary_reconstructed": mae(y, oofs["primary"]),
        "shadow_reconstructed": mae(y, oofs["shadow"]),
        "rotations": rotations,
    }
    log = ROOT / "results" / "xgb_rebuild_logs" / f"{experiment_id}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def copy_test_prediction(experiment_id: str, dev: pd.DataFrame, test: pd.DataFrame) -> Path:
    target = target_of(experiment_id)
    recipe = XGB_RECIPE[experiment_id]
    src = BUNDLE / "advanced_outputs" / "predictions" / f"{target}__xgboost__{recipe}__test.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing XGB test prediction {src}; regenerate with frozen full_dev_xgb_predict before continuing."
        )
    dest = ROOT / "experiments" / "predictions" / id_to_code(experiment_id) / "test.csv"
    normalize_prediction_csv(src, target, test["id"].astype(str).tolist(), dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-rerun", action="store_true", help="only convert existing npz + copy tests")
    args = ap.parse_args()

    dev, test = load_dev_test(BUNDLE / "dev.csv", BUNDLE / "test.csv")
    folds = load_folds(BUNDLE / "folds.csv")

    experiments = list(XGB_RECIPE.keys())
    if args.only:
        experiments = [e for e in experiments if e in set(args.only)]

    results = []
    for eid in experiments:
        print(f"=== {eid} ===", flush=True)
        copy_test_prediction(eid, dev, test)
        if eid in EXISTING_NPZ and EXISTING_NPZ[eid].exists():
            r = from_npz(eid, EXISTING_NPZ[eid], dev)
            print(f"  reused npz primary={r['primary_reconstructed']:.10f} shadow={r['shadow_reconstructed']:.10f}")
        elif args.skip_rerun:
            raise SystemExit(f"no OOF for {eid} and --skip-rerun set")
        else:
            r = run_oof_cv(eid, args.device, dev, folds)
            print(f"  reran primary={r['primary_reconstructed']:.10f} shadow={r['shadow_reconstructed']:.10f}")
        results.append(r)

    out = ROOT / "results" / "xgb_oof_rebuild_summary.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
