#!/usr/bin/env python3
"""EXP-H340–H343: prospective SURFACE SHAM35 vs REAL F1_SURFACE35 (no external scoring).

Phases: verify | smoke | train | analyze | all

EMBARGO: never load solution.csv / never compute Public/Private/Test Overall MAE.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import EXPERIMENTS_COLUMNS  # noqa: E402
from experiment_codes import issue_code, load_codes  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import run_protocol_v3_ext  # noqa: E402
from antibody_transformer.sham_f1_aux import (  # noqa: E402
    BUNDLE_REAL,
    BUNDLE_SHAM,
    TOTAL_DIM,
    make_surface_prospective_aux,
)

PREREG_SHA = "993b8a9e268f65ed8d203949c34cbe3ef87446e6"
PARENTS = {
    "ablang1": "EXP-H187",
    "ablingua": "EXP-H167",
}
SERIES = [
    {
        "key": "ablang1_sham",
        "parent": "EXP-H187",
        "representation": "ablang1",
        "plm_source": "ablang1",
        "arm": "SHAM35",
        "fusion_bundle_id": BUNDLE_SHAM,
        "experiment_id": "TRF_HIC_ABLANG1_JOINT_FULL_SHAM35_V3",
        "description": "Prospective AbLang1 JOINT FULL + SHAM35",
    },
    {
        "key": "ablang1_real",
        "parent": "EXP-H187",
        "representation": "ablang1",
        "plm_source": "ablang1",
        "arm": "REAL_F1_SURFACE35",
        "fusion_bundle_id": BUNDLE_REAL,
        "experiment_id": "TRF_HIC_ABLANG1_JOINT_FULL_REAL_F1_SURFACE35_V3",
        "description": "Prospective AbLang1 JOINT FULL + REAL F1_SURFACE35",
    },
    {
        "key": "ablingua_sham",
        "parent": "EXP-H167",
        "representation": "ablingua",
        "plm_source": "ablingua",
        "arm": "SHAM35",
        "fusion_bundle_id": BUNDLE_SHAM,
        "experiment_id": "TRF_HIC_ABLINGUA_JOINT_FULL_SHAM35_V3",
        "description": "Prospective AbLingua JOINT FULL + SHAM35",
    },
    {
        "key": "ablingua_real",
        "parent": "EXP-H167",
        "representation": "ablingua",
        "plm_source": "ablingua",
        "arm": "REAL_F1_SURFACE35",
        "fusion_bundle_id": BUNDLE_REAL,
        "experiment_id": "TRF_HIC_ABLINGUA_JOINT_FULL_REAL_F1_SURFACE35_V3",
        "description": "Prospective AbLingua JOINT FULL + REAL F1_SURFACE35",
    },
]
N_BOOT = 10_000
BOOT_SEED = 101
HIGH_THR = 11.5
ASSET = {
    "ablang1": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_cuda() -> None:
    import torch

    if ".venv_b1" not in str(Path(sys.executable).resolve()) and ".venv_b1" not in str(Path(torch.__file__).resolve()):
        raise SystemExit(f"must use .venv_b1 (got {sys.executable})")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")


def save_pred(ids, vals, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(ids), "y_pred": np.asarray(vals, float)}).to_csv(path, index=False)


def verify_parents() -> dict:
    exp = pd.read_csv(ROOT / "results/experiments.csv").set_index("experiment_code")
    out = {"ok": True, "details": []}
    for rep, code in PARENTS.items():
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        checks = {
            "representation": cfg.get("representation") == rep,
            "topology_id_JOINT": cfg.get("topology_id") == "JOINT",
            "topology_B1": cfg.get("topology") == "B1",
            "annotation_full": str(cfg.get("annotation_mode")).lower() == "full",
            "dual_reg": cfg.get("arch_joint_hl_dual_reg") is True,
            "share_hl": cfg.get("arch_share_hl_encoder") is True or cfg.get("share_hl_encoder") is True,
            "pooling_REG": cfg.get("pooling_mode") == "REG",
            "merge_mean": cfg.get("merge_mode") == "mean",
            "seed_101": int(cfg.get("seed", -1)) == 101,
            "frozen": cfg.get("content_mode") == "frozen",
            "context_SEPARATE": cfg.get("representation_context") == "SEPARATE_CHAIN",
            "surface_off": cfg.get("surface_features_enabled") in (False, None),
        }
        ok = all(checks.values())
        out["details"].append({"parent": code, "representation": rep, "checks": checks, "ok": ok})
        if not ok:
            out["ok"] = False
        # registry notes
        notes = str(exp.loc[code].get("notes", ""))
        if "JOINT" not in notes or "FULL" not in notes:
            out["details"][-1]["notes_warn"] = notes
    return out


def config_from_parent(parent_code: str, spec: dict, code: str) -> dict:
    parent = yaml.safe_load((ROOT / "experiments/configs" / f"{parent_code}.yaml").read_text())
    cfg = dict(parent)
    cfg.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "transformer_type": "LATE_FUSION",
            "fusion_bundle_id": spec["fusion_bundle_id"],
            "fusion_mode": "late_concat_aux32",
            "aux_dim": TOTAL_DIM,
            "surface_arm": spec["arm"],
            "surface_features_enabled": spec["arm"] == "REAL_F1_SURFACE35",
            "control_experiment_code": parent_code,
            "batch_id": "H340_H343_SURFACE_PROSPECTIVE",
            "prereg_sha": PREREG_SHA,
            "description": spec["description"],
            "input_asset_ref": ASSET[spec["representation"]],
            "no_external_scoring": True,
            "embargo_public_private_test": True,
        }
    )
    return cfg


def intended_config_diffs(parent: dict, child: dict) -> list[str]:
    ignore = {
        "experiment_code",
        "experiment_id",
        "description",
        "transformer_type",
        "fusion_bundle_id",
        "fusion_mode",
        "aux_dim",
        "surface_arm",
        "surface_features_enabled",
        "control_experiment_code",
        "batch_id",
        "prereg_sha",
        "no_external_scoring",
        "embargo_public_private_test",
        "input_space",
    }
    diffs = []
    keys = sorted(set(parent) | set(child))
    for k in keys:
        if k in ignore:
            continue
        if parent.get(k) != child.get(k):
            diffs.append(k)
    return diffs


def write_cfg(code: str, cfg: dict) -> None:
    path = ROOT / "experiments/configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def issue_series() -> list[dict]:
    """Issue codes once; persist mapping."""
    map_path = ROOT / "results/H340_H343_SURFACE_PROSPECTIVE_SERIES.json"
    if map_path.exists():
        return json.loads(map_path.read_text())
    issued = []
    for spec in SERIES:
        codes = load_codes()
        if spec["experiment_id"] in set(codes["experiment_id"].astype(str)):
            code = codes.loc[codes.experiment_id == spec["experiment_id"], "experiment_code"].iloc[0]
        else:
            code = issue_code(
                target="HIC",
                experiment_id=spec["experiment_id"],
                notes=spec["description"],
            )
        row = {**spec, "experiment_code": code}
        cfg = config_from_parent(spec["parent"], spec, code)
        parent = yaml.safe_load((ROOT / "experiments/configs" / f"{spec['parent']}.yaml").read_text())
        extra = intended_config_diffs(parent, cfg)
        if extra:
            raise SystemExit(f"unexpected config diffs for {code}: {extra}")
        write_cfg(code, cfg)
        issued.append(row)
    map_path.write_text(json.dumps(issued, indent=2) + "\n")
    return issued


def already_complete(code: str) -> bool:
    oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    pred = ROOT / "experiments/predictions" / code
    needed = [oof, pred / "oof_primary.csv", pred / "oof_shadow.csv", pred / "test.csv"]
    if not all(p.exists() for p in needed):
        return False
    doc = yaml.safe_load(oof.read_text())
    if doc.get("quick") or doc.get("technical_smoke"):
        return False
    if doc.get("external_scored"):
        raise SystemExit(f"{code} unexpectedly has external_scored=True under embargo")
    return True


def register_internal(code: str, spec: dict, summary: dict) -> None:
    exp_path = ROOT / "results/experiments.csv"
    exp = pd.read_csv(exp_path)
    row = {c: "" for c in EXPERIMENTS_COLUMNS if c in exp.columns}
    # fill carefully — leave pub/priv/test empty
    oof = summary["scores"]["oof_test"]
    vals = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "target": "HIC",
        "family": "TRANSFORMER",
        "notes": spec["description"] + "; EMBARGO_NO_EXTERNAL_SCORE",
        "cv_primary_mae": oof["primary"],
        "cv_shadow_mae": oof["shadow"],
        "cv_mean_mae": oof["mean"],
        "cv_worst_mae": oof.get("worst", max(oof["primary"], oof["shadow"])),
        "public_mae": np.nan,
        "private_mae": np.nan,
        "test_overall_mae": np.nan,
        "status": "INTERNAL_COMPLETE_EMBARGO",
    }
    for k, v in vals.items():
        if k in exp.columns:
            row[k] = v
    if code in set(exp.experiment_code.astype(str)):
        idx = exp.index[exp.experiment_code == code][0]
        for k, v in vals.items():
            if k in exp.columns:
                exp.at[idx, k] = v
    else:
        exp = pd.concat([exp, pd.DataFrame([row])], ignore_index=True)
    exp.to_csv(exp_path, index=False)


def run_one(spec: dict, *, quick: bool) -> dict:
    code = spec["experiment_code"]
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        return {"code": code, "spec": spec, "summary": summary, "skipped": True}

    cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments/predictions" / code
    aux_store = make_surface_prospective_aux(cfg["fusion_bundle_id"])
    arch = {
        "joint_hl_single_reg": bool(cfg["arch_joint_hl_single_reg"]),
        "joint_hl_dual_reg": bool(cfg["arch_joint_hl_dual_reg"]),
        "joint_hl_chain_specific_dual_reg": bool(cfg["arch_joint_hl_chain_specific_dual_reg"]),
        "use_cross_attention_bridge": bool(cfg["arch_use_cross_attention_bridge"]),
        "use_reg_only_cross_attention": bool(cfg["arch_use_reg_only_cross_attention"]),
        "use_within_chain_extra_attention": bool(cfg["arch_use_within_chain_extra_attention"]),
        "cross_gate_mode": cfg.get("arch_cross_gate_mode", "learned"),
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
    }
    plm = cfg["representation"]
    merge = cfg["merge_mode"]

    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    folds = load_folds(ROOT / "data/folds.csv")
    need = {"need_ablang1": plm == "ablang1", "need_ablingua": plm == "ablingua"}
    rb = load_residue_bundle(dev, test, **need)

    result = run_protocol_v3_ext(
        experiment_code=code,
        target="HIC",
        dev=dev,
        test=test,
        folds=folds,
        rb=rb,
        device=device_str(),
        out_dir=out,
        seed=DEFAULT_SEED,
        quick=quick,
        arch=arch,
        content_mode="frozen",
        merge_mode=merge,
        plm_source=plm,
        chain_mode=cfg.get("chain_mode", "HL"),
        capacity=None,
        aux_store=aux_store,
        fusion_mode="late_concat_aux32",
    )
    summary = result["summary"]
    hist_df = result["history_df"]
    sel_df = result["selected_df"]
    hist_df.to_csv(ROOT / "results" / f"{code}_TRAINING_HISTORY.csv", index=False)
    sel_df.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)

    pred.mkdir(parents=True, exist_ok=True)
    for scheme in ("primary", "shadow"):
        save_pred(result["dev_ids"], result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_val_{scheme}.csv")
        save_pred(result["dev_ids"], result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_test_{scheme}.csv")
    save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_primary.csv")
    save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_shadow.csv")
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv")
    for scheme in ("primary", "shadow"):
        for k in range(5):
            save_pred(result["test_ids"], result["ext"][f"{scheme}_folds"][k], pred / f"test_{scheme}_fold{k}.csv")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv")

    # EMBARGO: no solution load / no external MAE
    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "config_hash": summary["config_hash"],
        "fusion_bundle_id": cfg["fusion_bundle_id"],
        "surface_arm": spec["arm"],
        "parent": spec["parent"],
        "prereg_sha": PREREG_SHA,
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_scored": False,
        "external": None,
        "n_trainable": summary["n_trainable"],
        "quick": bool(quick),
        "embargo": "NO_PUBLIC_PRIVATE_TEST_SCORING",
        "code_sha": git_rev(),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps(
            {
                "config_hash": summary["config_hash"],
                "external_scored": False,
                "embargo": True,
                "n_trainable": summary["n_trainable"],
            },
            indent=2,
        )
        + "\n"
    )
    if not quick:
        register_internal(code, spec, summary)
    print(f"DONE {code} (embargoed; test preds saved unscored)", flush=True)
    return {"code": code, "spec": spec, "summary": summary, "skipped": False}


def smoke_sham() -> None:
    store = make_surface_prospective_aux(BUNDLE_SHAM)
    ids = store.ids[:40]
    prep = store.fit(ids)
    X = store.transform(prep, ids)
    assert X.shape == (40, TOTAL_DIM)
    assert np.allclose(X, 0.0)
    real = make_surface_prospective_aux(BUNDLE_REAL)
    prep_r = real.fit(ids)
    Xr = real.transform(prep_r, ids)
    assert Xr.shape == (40, TOTAL_DIM)
    assert not np.allclose(Xr, 0.0)
    print("SMOKE_SHAM_OK", flush=True)


def load_oof_series(code: str, name: str) -> pd.Series:
    df = pd.read_csv(ROOT / "experiments/predictions" / code / name)
    return df.set_index("id")["y_pred"].astype(float)


def paired_boot(d: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(d)
    boots = np.empty(N_BOOT, float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = float(d[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n": n, "mean": float(d.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def analyze(series: list[dict]) -> dict:
    by_key = {s["key"]: s for s in series}
    train = pd.read_csv(ROOT / "data/dev.csv")
    y = train.set_index("id")["HIC"].astype(float)

    rows = []
    boot_rows = []
    scheme_improve = []
    rng = np.random.default_rng(BOOT_SEED)

    for rep in ("ablang1", "ablingua"):
        sham = by_key[f"{rep}_sham"]["experiment_code"]
        real = by_key[f"{rep}_real"]["experiment_code"]
        oof_s = yaml.safe_load((ROOT / "results" / f"{sham}_OOF_EVALUATION.yaml").read_text())
        oof_r = yaml.safe_load((ROOT / "results" / f"{real}_OOF_EVALUATION.yaml").read_text())
        for scheme in ("primary", "shadow"):
            cv_s = float(oof_s["oof_test"][scheme])
            cv_r = float(oof_r["oof_test"][scheme])
            d = cv_r - cv_s
            scheme_improve.append(d < 0)
            # antibody bootstrap
            ps = load_oof_series(sham, f"oof_test_{scheme}.csv")
            pr = load_oof_series(real, f"oof_test_{scheme}.csv")
            ids = sorted(set(ps.index) & set(pr.index) & set(y.index))
            ae_s = np.abs(ps.loc[ids].to_numpy() - y.loc[ids].to_numpy())
            ae_r = np.abs(pr.loc[ids].to_numpy() - y.loc[ids].to_numpy())
            delta = ae_r - ae_s
            b = paired_boot(delta, rng)
            boot_rows.append({"representation": rep, "scheme": scheme, **b})
            # HIGH-tail
            high_ids = [i for i in ids if y.loc[i] > HIGH_THR]
            non_ids = [i for i in ids if y.loc[i] <= HIGH_THR]

            def mae_signed(pred_code, id_list):
                p = load_oof_series(pred_code, f"oof_test_{scheme}.csv")
                yy = y.loc[id_list].to_numpy()
                pp = p.loc[id_list].to_numpy()
                return float(np.abs(pp - yy).mean()), float((pp - yy).mean())

            if high_ids:
                mae_hs, sig_hs = mae_signed(sham, high_ids)
                mae_hr, sig_hr = mae_signed(real, high_ids)
                mae_ns, _ = mae_signed(sham, non_ids)
                mae_nr, _ = mae_signed(real, non_ids)
            else:
                mae_hs = mae_hr = sig_hs = sig_hr = mae_ns = mae_nr = float("nan")

            rows.append(
                {
                    "representation": rep,
                    "scheme": scheme,
                    "sham_code": sham,
                    "real_code": real,
                    "cv_sham": cv_s,
                    "cv_real": cv_r,
                    "delta_cv": d,
                    "boot_mean": b["mean"],
                    "boot_ci_lo": b["ci95_lo"],
                    "boot_ci_hi": b["ci95_hi"],
                    "n_high": len(high_ids),
                    "mae_high_sham": mae_hs,
                    "mae_high_real": mae_hr,
                    "delta_mae_high": mae_hr - mae_hs if high_ids else np.nan,
                    "signed_high_sham": sig_hs,
                    "signed_high_real": sig_hr,
                    "delta_signed_high": sig_hr - sig_hs if high_ids else np.nan,
                    "delta_mae_nonhigh": mae_nr - mae_ns if high_ids else np.nan,
                }
            )

        mean_s = float(oof_s["oof_test"]["mean"])
        mean_r = float(oof_r["oof_test"]["mean"])
        worst_s = float(oof_s["oof_test"].get("worst", max(oof_s["oof_test"]["primary"], oof_s["oof_test"]["shadow"])))
        worst_r = float(oof_r["oof_test"].get("worst", max(oof_r["oof_test"]["primary"], oof_r["oof_test"]["shadow"])))
        abs_ps_s = abs(float(oof_s["oof_test"]["primary"]) - float(oof_s["oof_test"]["shadow"]))
        abs_ps_r = abs(float(oof_r["oof_test"]["primary"]) - float(oof_r["oof_test"]["shadow"]))
        rows.append(
            {
                "representation": rep,
                "scheme": "mean",
                "sham_code": sham,
                "real_code": real,
                "cv_sham": mean_s,
                "cv_real": mean_r,
                "delta_cv": mean_r - mean_s,
                "delta_cv_worst": worst_r - worst_s,
                "abs_ps_sham": abs_ps_s,
                "abs_ps_real": abs_ps_r,
                "delta_abs_ps": abs_ps_r - abs_ps_s,
            }
        )

    # STRONG aggregation: Primary antibody-averaged across representations
    d_maps = {}
    for rep in ("ablang1", "ablingua"):
        sham = by_key[f"{rep}_sham"]["experiment_code"]
        real = by_key[f"{rep}_real"]["experiment_code"]
        ps = load_oof_series(sham, "oof_test_primary.csv")
        pr = load_oof_series(real, "oof_test_primary.csv")
        ids = sorted(set(ps.index) & set(pr.index) & set(y.index))
        d_maps[rep] = pd.Series(
            np.abs(pr.loc[ids].to_numpy() - y.loc[ids].to_numpy())
            - np.abs(ps.loc[ids].to_numpy() - y.loc[ids].to_numpy()),
            index=ids,
        )
    inter = sorted(set(d_maps["ablang1"].index) & set(d_maps["ablingua"].index))
    t = 0.5 * (d_maps["ablang1"].loc[inter].to_numpy() + d_maps["ablingua"].loc[inter].to_numpy())
    agg = paired_boot(t, rng)

    # verdict
    mean_deltas = {
        rep: next(r["delta_cv"] for r in rows if r["representation"] == rep and r["scheme"] == "mean")
        for rep in ("ablang1", "ablingua")
    }
    n_mean_improve = sum(1 for v in mean_deltas.values() if v < 0)
    n_scheme_improve = sum(1 for x in scheme_improve if x)
    if (
        mean_deltas["ablang1"] < 0
        and mean_deltas["ablingua"] < 0
        and n_scheme_improve >= 3
        and agg["ci95_hi"] < 0
    ):
        verdict = "INTERNAL_SURFACE_REPLICATION_STRONG"
    elif mean_deltas["ablang1"] < 0 and mean_deltas["ablingua"] < 0:
        verdict = "INTERNAL_SURFACE_REPLICATION_DIRECTIONAL"
    elif n_mean_improve == 1:
        verdict = "INTERNAL_SURFACE_REPLICATION_MIXED"
    else:
        verdict = "INTERNAL_SURFACE_REPLICATION_NOT_SUPPORTED"

    # secondary vs parents
    secondary = []
    for rep, parent in PARENTS.items():
        real = by_key[f"{rep}_real"]["experiment_code"]
        pe = pd.read_csv(ROOT / "results/experiments.csv").set_index("experiment_code").loc[parent]
        oof_r = yaml.safe_load((ROOT / "results" / f"{real}_OOF_EVALUATION.yaml").read_text())
        for scheme, col in (("primary", "cv_primary_mae"), ("shadow", "cv_shadow_mae"), ("mean", "cv_mean_mae")):
            if scheme == "mean":
                cv_r = float(oof_r["oof_test"]["mean"])
                cv_p = float(pe[col])
            else:
                cv_r = float(oof_r["oof_test"][scheme])
                cv_p = float(pe[col])
            secondary.append(
                {
                    "contrast": "SECONDARY_REAL_vs_SEQONLY_PARENT",
                    "representation": rep,
                    "scheme": scheme,
                    "parent": parent,
                    "real_code": real,
                    "cv_parent": cv_p,
                    "cv_real": cv_r,
                    "delta_real_minus_parent": cv_r - cv_p,
                }
            )

    return {
        "rows": rows,
        "boot_rows": boot_rows,
        "agg_primary": agg,
        "scheme_improve_count": n_scheme_improve,
        "mean_deltas": mean_deltas,
        "verdict": verdict,
        "secondary": secondary,
        "series": series,
    }


def write_internal_freeze(analysis: dict, verify: dict) -> None:
    REPORTS = REPO / "reports"
    pairs = pd.DataFrame([r for r in analysis["rows"] if r.get("scheme") != "mean" or "boot_mean" in r or True])
    # cleaner tables
    primary_tbl = []
    for rep in ("ablang1", "ablingua"):
        sham = next(s for s in analysis["series"] if s["key"] == f"{rep}_sham")
        real = next(s for s in analysis["series"] if s["key"] == f"{rep}_real")
        oof_s = yaml.safe_load((ROOT / "results" / f"{sham['experiment_code']}_OOF_EVALUATION.yaml").read_text())
        oof_r = yaml.safe_load((ROOT / "results" / f"{real['experiment_code']}_OOF_EVALUATION.yaml").read_text())
        primary_tbl.append(
            {
                "representation": rep,
                "sham_code": sham["experiment_code"],
                "real_code": real["experiment_code"],
                "cv_primary_sham": oof_s["oof_test"]["primary"],
                "cv_primary_real": oof_r["oof_test"]["primary"],
                "delta_cv_primary": oof_r["oof_test"]["primary"] - oof_s["oof_test"]["primary"],
                "cv_shadow_sham": oof_s["oof_test"]["shadow"],
                "cv_shadow_real": oof_r["oof_test"]["shadow"],
                "delta_cv_shadow": oof_r["oof_test"]["shadow"] - oof_s["oof_test"]["shadow"],
                "cv_mean_sham": oof_s["oof_test"]["mean"],
                "cv_mean_real": oof_r["oof_test"]["mean"],
                "delta_cv_mean": oof_r["oof_test"]["mean"] - oof_s["oof_test"]["mean"],
            }
        )
    pdf = pd.DataFrame(primary_tbl)
    pdf.to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_INTERNAL_RESULTS.csv", index=False)
    pd.DataFrame(analysis["boot_rows"]).to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_INTERNAL_BOOTSTRAP.csv", index=False)
    pd.DataFrame(analysis["secondary"]).to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_SECONDARY_VS_PARENT.csv", index=False)

    # HIGH rows from analysis.rows
    high_df = pd.DataFrame([r for r in analysis["rows"] if r.get("scheme") in ("primary", "shadow")])
    high_df.to_csv(REPORTS / "HIC_SURFACE_PROSPECTIVE_INTERNAL_HIGHTAIL.csv", index=False)

    lines = []
    A = lines.append
    A("# HIC SURFACE Prospective Replication — Internal Results Freeze")
    A("")
    A("**STATUS: `HIC_SURFACE_PROSPECTIVE_INTERNAL_FROZEN`**")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Prereg SHA | `{PREREG_SHA}` |")
    A(f"| Code/config SHA at freeze | `{git_rev()}` |")
    A("| External scoring | **NONE (embargo)** |")
    A("| solution.csv loaded | **NO** |")
    A("| Public/Private consulted | **NO** |")
    A("")
    A("## Parent verification")
    A("")
    A(f"- ok: **{verify['ok']}**")
    for d in verify["details"]:
        A(f"- {d['parent']} ({d['representation']}): ok={d['ok']}")
    A("")
    A("## Experiment codes")
    A("")
    for s in analysis["series"]:
        A(f"- `{s['experiment_code']}` — {s['description']}")
    A("")
    A("## PRIMARY contrast (REAL − SHAM)")
    A("")
    A("| Rep | ΔCV_P | ΔCV_S | ΔCV_mean |")
    A("|-----|-------|-------|----------|")
    for _, r in pdf.iterrows():
        A(f"| {r.representation} | {r.delta_cv_primary:+.6f} | {r.delta_cv_shadow:+.6f} | {r.delta_cv_mean:+.6f} |")
    A("")
    A(f"- Scheme-level improve count: **{analysis['scheme_improve_count']}/4**")
    A(f"- Aggregated Primary antibody treatment effect mean={analysis['agg_primary']['mean']:+.6f} "
      f"CI=[{analysis['agg_primary']['ci95_lo']:+.6f}, {analysis['agg_primary']['ci95_hi']:+.6f}]")
    A("")
    A(f"**Preregistered internal verdict:** `{analysis['verdict']}`")
    A("")
    A("## HIGH-tail diagnostic (HIC>11.5; not a success criterion)")
    A("")
    for _, r in high_df.iterrows():
        A(
            f"- {r.representation}/{r.scheme}: n_high={r.n_high}, "
            f"ΔMAE_high={r.delta_mae_high:+.4f}, Δsigned={r.delta_signed_high:+.4f}, "
            f"ΔMAE_nonHIGH={r.delta_mae_nonhigh:+.4f}"
        )
    A("")
    A("## Secondary (descriptive): REAL vs sequence-only parent")
    A("")
    A("See `HIC_SURFACE_PROSPECTIVE_SECONDARY_VS_PARENT.csv` (not primary causal contrast).")
    A("")
    A("## Embargo confirmation")
    A("")
    A("- Test predictions generated: **YES** (unscored)")
    A("- Public/Private/Test Overall in this report: **ABSENT**")
    A("")
    (REPORTS / "HIC_SURFACE_PROSPECTIVE_INTERNAL_RESULTS.md").write_text("\n".join(lines) + "\n")
    (REPORTS / "HIC_SURFACE_PROSPECTIVE_INTERNAL_SUMMARY.json").write_text(
        json.dumps(
            {
                "verdict": analysis["verdict"],
                "mean_deltas": analysis["mean_deltas"],
                "scheme_improve_count": analysis["scheme_improve_count"],
                "agg_primary": analysis["agg_primary"],
                "series": analysis["series"],
                "prereg_sha": PREREG_SHA,
                "public_private_consulted": False,
                "test_predictions_generated": True,
                "test_scored": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["verify", "smoke", "issue", "train", "analyze", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("verify", "all"):
        v = verify_parents()
        (ROOT / "results/H340_H343_PARENT_VERIFY.json").write_text(json.dumps(v, indent=2) + "\n")
        print("VERIFY", v["ok"], flush=True)
        if not v["ok"]:
            raise SystemExit("PARENT_VERIFY_FAILED — STOP before training")

    if args.phase in ("smoke", "all"):
        smoke_sham()

    if args.phase in ("issue", "train", "analyze", "all"):
        series = issue_series()
        print("SERIES", json.dumps([{k: s[k] for k in ('experiment_code','key','arm')} for s in series]), flush=True)

    if args.phase in ("train", "all"):
        require_cuda()
        series = issue_series()
        for spec in series:
            run_one(spec, quick=args.quick)

    if args.phase in ("analyze", "all"):
        if args.quick:
            print("SKIP analyze for --quick", flush=True)
            return 0
        series = issue_series()
        for s in series:
            if not already_complete(s["experiment_code"]):
                raise SystemExit(f"incomplete: {s['experiment_code']}")
        verify = json.loads((ROOT / "results/H340_H343_PARENT_VERIFY.json").read_text())
        analysis = analyze(series)
        write_internal_freeze(analysis, verify)
        print("VERDICT", analysis["verdict"], flush=True)
        print("INTERNAL_FREEZE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
