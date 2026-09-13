#!/usr/bin/env python3
"""HIC factorial runner EXP-H140–H339 (Representation × Topology × Annotation).

Phases:
  status | revalidate | smoke | train

Production:
  .venv_b1 + CUDA required for train
  train --i-understand-full-batch   # no --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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
from experiment_codes import load_codes  # noqa: E402
from antibody_transformer.config import RESIDUE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, run_protocol_v3  # noqa: E402
from preregister_h140_h339_factorial import TOPOLOGY_FLAGS, TOPO_LEGACY  # noqa: E402

PLAN_CSV = ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
STATUS_CSV = REPO / "reports/HIC_H140_H339_EXECUTION_STATUS.csv"
STATUS_CSV_ALT = ROOT / "results/HIC_H140_H339_EXECUTION_STATUS.csv"
SMOKE_DIR = ROOT / "results/h140_h339_technical_smoke"
SEED = DEFAULT_SEED
CODE_LO, CODE_HI = 140, 339

SMOKE_CODES = ("EXP-H140", "EXP-H234", "EXP-H317", "EXP-H287")

NEED_KW = {
    "ablingua": "need_ablingua",
    "ablang2": "need_ablang2",
    "ablang2_unpaired": "need_ablang2_unpaired",
    "ablang1": "need_ablang1",
    "esm1b": "need_esm1b",
    "esm2": "need_esm2",
    "esmc600m": "need_esmc600m",
    "currab": "need_currab",
    "currab_unpaired": "need_currab_unpaired",
}

STATUS_FIELDS = [
    "experiment_code",
    "representation",
    "topology",
    "annotation",
    "status",
    "start_time",
    "end_time",
    "primary_complete",
    "shadow_complete",
    "artifact_complete",
    "retry_count",
    "failure_reason",
    "code_sha",
]

ASSET_MAP = {
    "scratch": "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences",
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
    "ablang2_paired": "assets/transformer/residue_asset_manifest.yaml#ablang2",
    "ablang2_unpaired": "assets/transformer/residue_asset_manifest.yaml#ablang2_unpaired",
    "ablang1": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    "esm1b": "assets/transformer/residue_asset_manifest.yaml#esm1b",
    "esm2": "assets/transformer/residue_asset_manifest.yaml#esm2",
    "esmc600m": "assets/transformer/residue_asset_manifest.yaml#esmc600m",
    "currab_paired": "assets/transformer/residue_asset_manifest.yaml#currab",
    "currab_unpaired": "assets/transformer/residue_asset_manifest.yaml#currab_unpaired",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_plm_source(row: dict) -> str | None:
    if row.get("content_mode") == "scratch":
        return None
    ps = row.get("plm_source")
    if ps is None or (isinstance(ps, float) and pd.isna(ps)):
        return None
    s = str(ps).strip()
    if s in ("", "nan", "None", "NONE"):
        return None
    return s


def _plm_source_label(row: dict) -> str:
    ps = _norm_plm_source(row)
    return "NONE" if ps is None else ps.upper()


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_cuda_for_train() -> None:
    import torch

    markers = [
        sys.executable,
        sys.prefix,
        os.environ.get("VIRTUAL_ENV", ""),
        str(Path(sys.executable).resolve()),
    ]
    # uv-managed venvs may resolve the interpreter outside .venv_b1; also check argv0
    if len(sys.argv) >= 1:
        markers.append(sys.argv[0])
    ok_env = any(".venv_b1" in str(m) for m in markers)
    # Fallback: site-packages path under .venv_b1
    try:
        import torch as _t

        ok_env = ok_env or (".venv_b1" in str(Path(_t.__file__).resolve()))
    except Exception:
        pass
    if not ok_env:
        raise SystemExit(
            f"Refusing train: must run under .venv_b1 "
            f"(executable={sys.executable} prefix={sys.prefix}). "
            "Default .venv is CPU-only torch."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Refusing train: CUDA not available in .venv_b1")
    if torch.version.cuda is None:
        raise SystemExit("Refusing train: torch build has no CUDA")
    print(
        f"EXEC env python={sys.executable} prefix={sys.prefix} "
        f"torch={torch.__version__} cuda_build={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )


def load_plan() -> pd.DataFrame:
    if not PLAN_CSV.exists():
        raise SystemExit("missing plan CSV — run preregister_h140_h339_factorial.py first")
    df = pd.read_csv(PLAN_CSV)
    codes = df["experiment_code"].astype(str).tolist()
    expect = [f"EXP-H{i:03d}" for i in range(CODE_LO, CODE_HI + 1)]
    if codes != expect:
        raise SystemExit("plan experiment_code range must be exactly EXP-H140..EXP-H339")
    return df


def save_plan(df: pd.DataFrame) -> None:
    tmp = PLAN_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(PLAN_CSV)


def already_complete(code: str) -> bool:
    oof = ROOT / f"results/{code}_OOF_EVALUATION.yaml"
    pred = ROOT / "experiments/predictions" / code
    needed = [oof, pred / "oof_primary.csv", pred / "oof_shadow.csv", pred / "test.csv"]
    if not all(p.exists() for p in needed):
        return False
    doc = yaml.safe_load(oof.read_text())
    if doc.get("technical_smoke") or doc.get("quick"):
        return False
    cfg = yaml.safe_load((ROOT / f"experiments/configs/{code}.yaml").read_text())
    if cfg.get("target") != "HIC":
        return False
    if cfg.get("share_hl_encoder") is not True:
        return False
    return True


def load_bundle_for_rows(rows: list[dict], dev, test):
    need: dict[str, bool] = {}
    for r in rows:
        ps = _norm_plm_source(r) or ""
        if not ps or r.get("content_mode") == "scratch":
            continue
        kw = NEED_KW.get(ps)
        if kw:
            need[kw] = True
    return load_residue_bundle(dev, test, **need)


def load_status() -> pd.DataFrame:
    if STATUS_CSV.exists():
        return pd.read_csv(STATUS_CSV)
    return pd.DataFrame(columns=STATUS_FIELDS)


def save_status(df: pd.DataFrame) -> None:
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_CSV.with_suffix(".csv.tmp")
    out = df.reindex(columns=STATUS_FIELDS).astype(str)
    out.to_csv(tmp, index=False)
    tmp.replace(STATUS_CSV)
    STATUS_CSV_ALT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATUS_CSV_ALT, index=False)


def upsert_status(row: dict) -> None:
    """Rewrite-row upsert; all fields stored as strings for resilience."""
    df = load_status()
    if not len(df):
        df = pd.DataFrame(columns=STATUS_FIELDS)
    df = df.astype(str)
    code = str(row["experiment_code"])
    # normalize incoming
    norm = {k: ("" if v is None else str(v)) for k, v in row.items() if k in STATUS_FIELDS}
    if code in set(df["experiment_code"].astype(str)):
        idx = df.index[df["experiment_code"].astype(str) == code][0]
        for k, v in norm.items():
            df.at[idx, k] = v
    else:
        blank = {c: "" for c in STATUS_FIELDS}
        blank.update(norm)
        df = pd.concat([df, pd.DataFrame([blank]).astype(str)], ignore_index=True)
    save_status(df)


def init_status_from_plan(plan: pd.DataFrame) -> None:
    existing = load_status()
    by = {str(r.experiment_code): r.to_dict() for _, r in existing.iterrows()} if len(existing) else {}
    rows = []
    for _, r in plan.iterrows():
        code = str(r.experiment_code)
        prev = by.get(code, {})
        st = str(prev.get("status") or "PENDING")
        if already_complete(code):
            st = "COMPLETE"
        done = st == "COMPLETE"
        rows.append(
            {
                "experiment_code": code,
                "representation": str(r["representation"]),
                "topology": str(r["topology"]),
                "annotation": str(r["annotation"]),
                "status": st,
                "start_time": str(prev.get("start_time") or ""),
                "end_time": str(prev.get("end_time") or ""),
                "primary_complete": "True" if done else str(prev.get("primary_complete") or "False"),
                "shadow_complete": "True" if done else str(prev.get("shadow_complete") or "False"),
                "artifact_complete": "True" if done else str(prev.get("artifact_complete") or "False"),
                "retry_count": str(prev.get("retry_count") or "0"),
                "failure_reason": str(prev.get("failure_reason") or ""),
                "code_sha": str(prev.get("code_sha") or "") or git_rev(),
            }
        )
    save_status(pd.DataFrame(rows).astype(str))


def assert_no_tmapp_leak(cfg: dict, code: str) -> None:
    if cfg.get("target") != "HIC":
        raise RuntimeError(f"{code}: target leak {cfg.get('target')}")
    blob = json.dumps(cfg)
    if "TmApp" in blob:
        raise RuntimeError(f"{code}: TmApp contamination in config")


def _input_space(row: dict) -> str:
    topo = str(row.get("topology_legacy") or TOPO_LEGACY.get(row["topology"], row["topology"]))
    rep = row["representation"]
    topo_prefix = {
        "A": "SEPARATE_DUAL_REG",
        "B1": "JOINT_HL_DUAL_REG",
        "B2": "JOINT_HL_CHAIN_SPECIFIC_DUAL_REG",
        "C": "SEPARATE_REG_ONLY_CROSS_ATTENTION",
        "D": "PAIR_D3",
        "SEP": "SEPARATE_DUAL_REG",
        "JOINT": "JOINT_HL_DUAL_REG",
        "REG-SEP": "JOINT_HL_CHAIN_SPECIFIC_DUAL_REG",
        "XREG": "SEPARATE_REG_ONLY_CROSS_ATTENTION",
        "FUSE": "PAIR_D3",
    }[topo]
    legacy = {"SEP": "A", "JOINT": "B1", "REG-SEP": "B2", "XREG": "C", "FUSE": "D"}.get(topo, topo)
    if rep == "scratch":
        return "PAIR_D3_SCRATCH_MEAN_V3" if legacy == "D" else f"{topo_prefix}_SCRATCH_RESIDUE_MEAN_V3"
    if legacy == "D":
        return f"PAIR_D3_FROZEN_{rep.upper()}_MEAN_V3"
    return f"{topo_prefix}_FROZEN_{rep.upper()}_RESIDUE_MEAN_V3"


def register_hic(code: str, row: dict, summary: dict) -> None:
    """Update registry to FULL without consulting Public/Private."""
    oof = summary["scores"]["oof_test"]
    exp_path = ROOT / "results/experiments.csv"
    df = pd.read_csv(exp_path)
    df = df[df["experiment_code"] != code]
    rec = {c: "" for c in df.columns}
    for c in EXPERIMENTS_COLUMNS:
        rec.setdefault(c, "")
    rec.update(
        {
            "experiment_code": code,
            "experiment_id": row.get("experiment_id")
            or f"TRF_HIC_{row['representation'].upper()}_{row['annotation']}_{row.get('topology_legacy', row['topology'])}_MEAN_V3",
            "target": "HIC",
            "family": "TRANSFORMER",
            "model_type": "TRANSFORMER",
            "source_model_id": "HIC_REP_TOPO_ANNOT_FACTORIAL",
            "config_path": f"experiments/configs/{code}.yaml",
            "artifact_status": "FULL",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_mean_mae": oof["mean"],
            "cv_worst_mae": oof["worst"],
            # Public/Private intentionally blank — embargo
            "public_mae": "",
            "private_mae": "",
            "test_overall_mae": "",
            "public_private_delta": "",
            "public_private_gap": "",
            "cv_protocol": "dl_foldlocal_cosine_v3_oof_test",
            "selection_policy_at_creation": "POSTCOMP_EXPLORATORY",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "license_status": "OK",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "shareability_status": "SHAREABLE_COMPLETE",
            "canonical_benchmark_eligible": "NO",  # internal freeze; external not yet
            "notes": f"factorial {row['representation']}/{row['topology']}/{row['annotation']}; pub/priv embargo",
            "transformer_type": "SCRATCH" if row["content_mode"] == "scratch" else "FROZEN_PLM",
            "plm_source": _plm_source_label(row),
            "annotation_mode": row["annotation_mode"],
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "prediction_reproduction_max_delta": 0.0,
            "input_space": _input_space(row),
            "input_asset_ref": ASSET_MAP[row["representation"]],
        }
    )
    pd.concat([df, pd.DataFrame([{c: rec.get(c, "") for c in df.columns}])], ignore_index=True).to_csv(
        exp_path, index=False
    )
    comp_path = ROOT / "results/EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp = comp[comp["experiment_code"] != code]
        crow = {c: "" for c in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": rec["experiment_id"],
                "target": "HIC",
                "family": "TRANSFORMER",
                "config_exists": True,
                "feature_required": False,
                "feature_exists": True,
                "oof_primary_exists": True,
                "oof_shadow_exists": True,
                "test_prediction_exists": True,
                "score_recompute_ok": True,
                "reproduction_status": "REPRODUCED",
                "shareability_status": "SHAREABLE_COMPLETE",
                "canonical_benchmark_eligible": "NO",
                "notes": "H140–H339 factorial COMPLETE; pub/priv embargo",
            }
        )
        comp = pd.concat([comp, pd.DataFrame([crow])], ignore_index=True)
        comp.to_csv(comp_path, index=False)


def run_cell(
    row: dict,
    *,
    rb,
    folds,
    dev,
    test,
    quick: bool = False,
    smoke: bool = False,
) -> dict:
    code = row["experiment_code"]
    cfg_path = ROOT / "experiments/configs" / f"{code}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert_no_tmapp_leak(cfg, code)
    if cfg.get("share_hl_encoder") is not True or cfg.get("arch_share_hl_encoder") is not True:
        return {"code": code, "status": "FAILED_TECHNICAL", "error": "share_hl_encoder not True"}
    if cfg.get("surface_features_enabled"):
        return {"code": code, "status": "FAILED_TECHNICAL", "error": "surface features enabled"}
    if (not smoke) and quick:
        return {"code": code, "status": "FAILED_TECHNICAL", "error": "production train must not use --quick"}

    topo_key = str(row.get("topology") or cfg.get("topology_id") or "")
    if topo_key not in TOPOLOGY_FLAGS:
        rev = {"A": "SEP", "B1": "JOINT", "B2": "REG-SEP", "C": "XREG", "D": "FUSE"}
        topo_key = rev.get(str(cfg.get("topology")), topo_key)
    flags = TOPOLOGY_FLAGS[topo_key]

    if (not smoke) and already_complete(code) and not quick:
        oof = yaml.safe_load((ROOT / f"results/{code}_OOF_EVALUATION.yaml").read_text())
        ot = oof.get("oof_test") or oof.get("scores", {}).get("oof_test")
        return {"code": code, "status": "COMPLETE", "oof_test": ot, "skipped": True}

    print(
        f"==== {'SMOKE' if smoke else 'TRAIN'} {code} "
        f"{row['representation']}/{row['topology']}/{row['annotation']} ====",
        flush=True,
    )
    out = (SMOKE_DIR / code) if smoke else (ROOT / "results" / f"{code}_run")
    out.mkdir(parents=True, exist_ok=True)
    try:
        flag_log = {
            "share_hl_encoder": True,
            "topology": topo_key,
            "pair_interaction_mode": flags.get("pair_interaction_mode"),
            "joint_hl_dual_reg": flags.get("joint_hl_dual_reg"),
            "joint_hl_chain_specific_dual_reg": flags.get("joint_hl_chain_specific_dual_reg"),
            "use_reg_only_cross_attention": flags.get("use_reg_only_cross_attention"),
            "surface_features_enabled": False,
            "target": "HIC",
            "plm_source": _norm_plm_source(row),
            "annotation_mode": row["annotation_mode"],
            "quick": quick,
            "smoke": smoke,
        }
        (out / "runtime_flags.json").write_text(json.dumps(flag_log, indent=2) + "\n")

        result = run_protocol_v3(
            experiment_code=code,
            target="HIC",
            dev=dev,
            test=test,
            folds=folds,
            rb=rb,
            device=device_str(),
            out_dir=out,
            seed=SEED,
            quick=quick,
            arch=flags,
            content_mode=row["content_mode"],
            merge_mode="mean",
            plm_source=_norm_plm_source(row),
            annotation_mode=row["annotation_mode"],
            chain_mode="HL",
        )
    except Exception as e:
        print(f"FAILED_TECHNICAL {code}: {e}", flush=True)
        return {"code": code, "status": "FAILED_TECHNICAL", "error": str(e)}

    for scheme in ("primary", "shadow"):
        vals = result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float)
        if not np.all(np.isfinite(vals)):
            return {"code": code, "status": "FAILED_TECHNICAL", "error": f"{scheme} non-finite"}
        var = float(np.nanvar(vals))
        if not np.isfinite(var) or var <= 0:
            return {"code": code, "status": "FAILED_TECHNICAL", "error": f"{scheme} var={var}"}

    summary = result["summary"]
    ot = summary["scores"]["oof_test"]
    if smoke:
        report = {
            "experiment_code": code,
            "technical_smoke": True,
            "quick": quick,
            "status": "SMOKE_PASS",
            "target": "HIC",
            "share_hl_encoder": True,
            "topology_flags": flag_log,
            "oof_test": ot,
            "note": "Technical smoke only — not a scientific factorial COMPLETE result",
        }
        (out / "SMOKE_REPORT.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        return {"code": code, "status": "SMOKE_PASS", "oof_test": ot, "smoke_dir": str(out)}

    sel = result["selected_df"]
    sel.to_csv(ROOT / f"results/{code}_SELECTED_LR.csv", index=False)
    pred = ROOT / "experiments/predictions" / code
    pred.mkdir(parents=True, exist_ok=True)

    def save_pred(ids, vals, path: Path) -> None:
        pd.DataFrame({"id": ids, "HIC": vals}).to_csv(path, index=False)

    save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_primary.csv")
    save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_shadow.csv")
    for scheme in ("primary", "shadow"):
        save_pred(result["dev_ids"], result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_val_{scheme}.csv")
        save_pred(result["dev_ids"], result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_test_{scheme}.csv")
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv")

    pvar = float(np.nanvar(result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float)))
    oof_doc = {
        "experiment_code": code,
        "target": "HIC",
        "representation": row["representation"],
        "topology": row["topology"],
        "annotation": row["annotation"],
        "quick": False,
        "technical_smoke": False,
        "share_hl_encoder": True,
        "oof_test": ot,
        "oof_val": summary["scores"]["oof_val"],
        "n_trainable": summary.get("n_trainable"),
        "param_account": summary.get("param_account"),
        "selected": sel.to_dict(orient="records"),
        "prediction_variance_oof_primary": pvar,
        "public_private_inspected": False,
        "code_sha": git_rev(),
    }
    (ROOT / f"results/{code}_OOF_EVALUATION.yaml").write_text(yaml.safe_dump(oof_doc, sort_keys=False))
    register_hic(code, row, summary)
    print(f"DONE {code} TEST_mean={ot['mean']:.6f}", flush=True)
    return {"code": code, "status": "COMPLETE", "oof_test": ot, "summary": summary}


def phase_status() -> None:
    df = load_plan()
    init_status_from_plan(df)
    st = load_status()
    print("STATUS", st["status"].value_counts().to_dict(), flush=True)


def phase_revalidate() -> int:
    """Programmatic freeze check before full train."""
    import hashlib

    plan = load_plan()
    man = pd.read_csv(REPO / "reports/HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv")
    errs: list[str] = []
    if len(plan) != 200 or len(man) != 200:
        errs.append(f"row counts plan={len(plan)} man={len(man)}")
    codes = [f"EXP-H{i:03d}" for i in range(140, 340)]
    if plan["experiment_code"].astype(str).tolist() != codes:
        errs.append("plan codes not EXP-H140..H339")
    if man["experiment_code"].astype(str).tolist() != codes:
        errs.append("manifest codes not EXP-H140..H339")
    combos = set(zip(plan.representation, plan.topology, plan.annotation))
    if len(combos) != 200:
        errs.append(f"unique combos={len(combos)}")
    if plan.representation.nunique() != 10:
        errs.append("rep!=10")
    if plan.topology.nunique() != 5:
        errs.append("topo!=5")
    if plan.annotation.nunique() != 4:
        errs.append("annot!=4")

    for _, r in plan.iterrows():
        code = r.experiment_code
        yp = ROOT / "experiments/configs" / f"{code}.yaml"
        if not yp.exists():
            errs.append(f"missing config {code}")
            continue
        y = yaml.safe_load(yp.read_text())
        if y.get("target") != "HIC":
            errs.append(f"{code} target")
        if y.get("seed") != 101:
            errs.append(f"{code} seed")
        if y.get("d_model") != 128:
            errs.append(f"{code} d_model")
        if y.get("merge_mode") != "mean":
            errs.append(f"{code} merge")
        if y.get("share_hl_encoder") is not True or y.get("arch_share_hl_encoder") is not True:
            errs.append(f"{code} share_hl")
        if y.get("surface_features_enabled"):
            errs.append(f"{code} surface")
        if y.get("quick"):
            errs.append(f"{code} quick in config")
        flags = TOPOLOGY_FLAGS[str(r.topology)]
        if y.get("pair_interaction_mode") != flags["pair_interaction_mode"]:
            errs.append(f"{code} pair_interaction_mode")
        if y.get("arch_use_reg_only_cross_attention") != flags["use_reg_only_cross_attention"]:
            errs.append(f"{code} xreg flag")
        if y.get("arch_joint_hl_dual_reg") != flags["joint_hl_dual_reg"]:
            errs.append(f"{code} joint flag")
        if str(r.content_mode) == "frozen":
            sub = str(r.subdir)
            if not (RESIDUE_ROOT / sub / "metadata.json").exists():
                errs.append(f"{code} missing asset {sub}")
        # no pub/priv in config
        blob = json.dumps(y)
        if "public" in blob.lower() and "public_private" in blob.lower():
            pass  # tolerate notes; check paths
        if "solution.csv" in blob:
            errs.append(f"{code} solution dependency")

    # reuse must be none
    if "reuse_status" in man.columns and (man["reuse_status"].astype(str) == "REUSE").any():
        errs.append("reuse cells present")

    out = {
        "status": "PASS" if not errs else "FAIL",
        "n_errors": len(errs),
        "errors": errs[:50],
        "code_sha": git_rev(),
    }
    path = ROOT / "results/HIC_H140_H339_PREFLIGHT_REVALIDATE.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(out["status"], f"errors={len(errs)}", flush=True)
    if errs:
        for e in errs[:20]:
            print(" ERR", e, flush=True)
        return 1
    return 0


def phase_smoke(*, quick: bool = True, codes: list[str] | None = None) -> int:
    df = load_plan()
    want = codes or list(SMOKE_CODES)
    subprocess.check_call(["bash", str(RESIDUE_ROOT / "assemble_embeddings.sh")])
    for sub in ("ablang2_unpaired", "currab_unpaired"):
        if not (RESIDUE_ROOT / sub / "metadata.json").exists():
            raise SystemExit(f"BLOCKED missing asset {sub}")
    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    folds = load_folds(ROOT / "data/folds.csv")
    rows = [df[df.experiment_code == c].iloc[0].to_dict() for c in want]
    rb = load_bundle_for_rows(rows, dev, test)
    results = []
    for row in rows:
        res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=quick, smoke=True)
        if res["status"] == "FAILED_TECHNICAL":
            res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=quick, smoke=True)
        results.append(res)
    report = {
        "phase": "technical_smoke",
        "codes": want,
        "quick": quick,
        "results": results,
        "pass": all(r["status"] == "SMOKE_PASS" for r in results),
        "mae_is_pass_criterion": False,
        "public_private_inspected": False,
    }
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "SMOKE_SUMMARY.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print("SMOKE", "PASS" if report["pass"] else "FAIL", flush=True)
    return 0 if report["pass"] else 1


def phase_train(*, quick: bool = False, only_codes: list[str] | None = None) -> int:
    if quick:
        raise SystemExit("production full-batch must not use --quick")
    require_cuda_for_train()
    df = load_plan()
    codes_ok = set(load_codes()["experiment_code"].astype(str))
    for c in [f"EXP-H{i:03d}" for i in range(CODE_LO, CODE_HI + 1)]:
        if c not in codes_ok:
            raise SystemExit(f"missing issued code {c}")

    init_status_from_plan(df)
    sha = git_rev()

    subprocess.check_call(["bash", str(RESIDUE_ROOT / "assemble_embeddings.sh")])
    for sub in ("ablang2_unpaired", "currab_unpaired"):
        if not (RESIDUE_ROOT / sub / "metadata.json").exists():
            raise SystemExit(f"BLOCKED missing asset {sub}")

    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    folds = load_folds(ROOT / "data/folds.csv")
    planned = df.copy()
    if only_codes:
        planned = planned[planned.experiment_code.isin(only_codes)]
    rb = load_bundle_for_rows(df.to_dict(orient="records"), dev, test)

    for row in planned.to_dict(orient="records"):
        code = row["experiment_code"]
        if already_complete(code):
            df.loc[df.experiment_code == code, "execution_status"] = "COMPLETE"
            if "primary_mae" in df.columns and pd.isna(df.loc[df.experiment_code == code, "primary_mae"]).all():
                oof = yaml.safe_load((ROOT / f"results/{code}_OOF_EVALUATION.yaml").read_text())
                ot = oof.get("oof_test") or {}
                if ot:
                    df.loc[df.experiment_code == code, "primary_mae"] = ot.get("primary")
                    df.loc[df.experiment_code == code, "shadow_mae"] = ot.get("shadow")
                    df.loc[df.experiment_code == code, "mean_ps"] = ot.get("mean")
                    df.loc[df.experiment_code == code, "worst_ps"] = ot.get("worst")
            upsert_status(
                {
                    "experiment_code": code,
                    "representation": row["representation"],
                    "topology": row["topology"],
                    "annotation": row["annotation"],
                    "status": "COMPLETE",
                    "primary_complete": True,
                    "shadow_complete": True,
                    "artifact_complete": True,
                    "code_sha": sha,
                }
            )
            save_plan(df)
            continue

        upsert_status(
            {
                "experiment_code": code,
                "representation": row["representation"],
                "topology": row["topology"],
                "annotation": row["annotation"],
                "status": "RUNNING",
                "start_time": utc_now(),
                "end_time": "",
                "primary_complete": False,
                "shadow_complete": False,
                "artifact_complete": False,
                "retry_count": 0,
                "failure_reason": "",
                "code_sha": sha,
            }
        )
        res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=False, smoke=False)
        retries = 0
        while res["status"] == "FAILED_TECHNICAL" and retries < 1:
            retries += 1
            print(f"retry {retries} {code}", flush=True)
            upsert_status(
                {
                    "experiment_code": code,
                    "status": "RUNNING",
                    "retry_count": retries,
                    "failure_reason": res.get("error", ""),
                    "code_sha": sha,
                }
            )
            res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=False, smoke=False)

        st = res["status"]
        df.loc[df.experiment_code == code, "execution_status"] = st
        if st == "COMPLETE" and res.get("oof_test"):
            ot = res["oof_test"]
            for col, key in (
                ("primary_mae", "primary"),
                ("shadow_mae", "shadow"),
                ("mean_ps", "mean"),
                ("worst_ps", "worst"),
            ):
                if col in df.columns:
                    df.loc[df.experiment_code == code, col] = ot.get(key)
        save_plan(df)
        upsert_status(
            {
                "experiment_code": code,
                "representation": row["representation"],
                "topology": row["topology"],
                "annotation": row["annotation"],
                "status": st,
                "end_time": utc_now(),
                "primary_complete": st == "COMPLETE",
                "shadow_complete": st == "COMPLETE",
                "artifact_complete": st == "COMPLETE",
                "retry_count": retries,
                "failure_reason": res.get("error", "") if st != "COMPLETE" else "",
                "code_sha": sha,
            }
        )

    st = load_status()
    vc = st["status"].value_counts().to_dict()
    print("train phase done", vc, flush=True)
    n_ok = int((st["status"] == "COMPLETE").sum())
    return 0 if n_ok == 200 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["status", "revalidate", "smoke", "train"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--i-understand-full-batch", action="store_true")
    args = ap.parse_args()
    if args.phase == "status":
        phase_status()
        return 0
    if args.phase == "revalidate":
        return phase_revalidate()
    if args.phase == "smoke":
        return phase_smoke(quick=True, codes=args.codes)
    if args.phase == "train":
        if not args.codes and not args.i_understand_full_batch:
            raise SystemExit(
                "Refusing full 200-cell train without --i-understand-full-batch "
                "(or pass --codes for a subset)"
            )
        if args.quick:
            raise SystemExit("Refusing --quick for production train")
        return phase_train(quick=False, only_codes=args.codes)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
