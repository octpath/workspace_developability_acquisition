#!/usr/bin/env python3
"""HIC factorial runner EXP-H140–H339 (Representation × Topology × Annotation).

Phases:
  status | smoke | train

- Consumes frozen plan/manifest
- Resumes safely; skips only COMPLETE cells with valid artifacts
- Retries technical failures without changing scientific config
- Never skips on poor MAE; never inspects Public/Private
- Distinguishes COMPLETE / FAILED_TECHNICAL / BLOCKED

Full batch requires separate human approval. Prefer `smoke` for wiring checks.
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

from experiment_codes import load_codes  # noqa: E402
from antibody_transformer.config import RESIDUE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, run_protocol_v3  # noqa: E402
from preregister_h140_h339_factorial import TOPOLOGY_FLAGS  # noqa: E402

PLAN_CSV = ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
STATUS_CSV = ROOT / "results/HIC_H140_H339_EXECUTION_STATUS.csv"
SMOKE_DIR = ROOT / "results/h140_h339_technical_smoke"
SEED = DEFAULT_SEED
CODE_LO, CODE_HI = 140, 339

# Suggested technical smoke set (wiring only; may use --quick)
SMOKE_CODES = (
    "EXP-H140",  # Scratch × SEP × BASE
    "EXP-H234",  # AbLang2 PAIRED_NATIVE × XREG × REGION
    "EXP-H317",  # CurrAb SEPARATE_CHAIN × FUSE × IMGT
    "EXP-H287",  # ESM-C 600M × JOINT × FULL
)

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


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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
    needed = [
        oof,
        pred / "oof_primary.csv",
        pred / "oof_shadow.csv",
        pred / "test.csv",
    ]
    if not all(p.exists() for p in needed):
        return False
    # smoke quick runs must not count as scientific COMPLETE
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
    kwargs = {v: False for v in NEED_KW.values()}
    kwargs["need_annotations"] = True
    for r in rows:
        ps = _norm_plm_source(r)
        if ps and ps in NEED_KW:
            kwargs[NEED_KW[ps]] = True
    return load_residue_bundle(dev, test, **kwargs)


def update_status(rows: list[dict]) -> None:
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATUS_CSV, index=False)


def assert_no_tmapp_leak(cfg: dict, code: str) -> None:
    if cfg.get("target") != "HIC":
        raise RuntimeError(f"{code}: target leak {cfg.get('target')}")
    blob = json.dumps(cfg)
    if "TmApp" in blob or "/EXP-T" in blob:
        raise RuntimeError(f"{code}: TmApp contamination in config")


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

    topo_key = str(row.get("topology") or cfg.get("topology_id") or "")
    if topo_key not in TOPOLOGY_FLAGS:
        # accept legacy A/B1 keys via reverse map
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
        # Log explicit flags for smoke audit
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

    # Finite + nonzero variance (Primary and Shadow)
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
            "prediction_variance_oof_primary": float(
                np.nanvar(result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float))
            ),
            "prediction_variance_oof_shadow": float(
                np.nanvar(result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float))
            ),
            "artifacts": {
                "summary": str(out / "summary.json"),
                "runtime_flags": str(out / "runtime_flags.json"),
            },
            "note": "Technical smoke only — not a scientific factorial COMPLETE result",
        }
        (out / "SMOKE_REPORT.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        return {"code": code, "status": "SMOKE_PASS", "oof_test": ot, "smoke_dir": str(out)}

    # Production path: write scientific artifacts (no Public/Private inspection)
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
    }
    (ROOT / f"results/{code}_OOF_EVALUATION.yaml").write_text(yaml.safe_dump(oof_doc, sort_keys=False))
    return {"code": code, "status": "COMPLETE", "oof_test": ot}


def phase_status() -> None:
    df = load_plan()
    rows = []
    for _, r in df.iterrows():
        code = r["experiment_code"]
        if already_complete(code):
            st = "COMPLETE"
        elif r.get("execution_status") == "BLOCKED":
            st = "BLOCKED"
        elif str(r.get("execution_status", "")).startswith("FAILED"):
            st = "FAILED_TECHNICAL"
        else:
            st = str(r.get("execution_status") or "PLANNED")
        rows.append(
            {
                "experiment_code": code,
                "representation": r["representation"],
                "topology": r["topology"],
                "annotation": r["annotation"],
                "status": st,
            }
        )
    update_status(rows)
    vc = pd.Series([x["status"] for x in rows]).value_counts().to_dict()
    print("STATUS", vc, flush=True)


def phase_smoke(*, quick: bool = True, codes: list[str] | None = None) -> int:
    df = load_plan()
    want = codes or list(SMOKE_CODES)
    for c in want:
        if c not in set(df.experiment_code.astype(str)):
            raise SystemExit(f"smoke code not in plan: {c}")
    # Asset assemble (same as TmApp) — do not regenerate embeddings
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
            print(f"retry once {row['experiment_code']}", flush=True)
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


def phase_train(*, quick: bool = False, only_codes: list[str] | None = None) -> None:
    """Full scientific train. Do not call without human approval for the 200-cell batch."""
    df = load_plan()
    codes_ok = set(load_codes()["experiment_code"].astype(str))
    for c in [f"EXP-H{i:03d}" for i in range(CODE_LO, CODE_HI + 1)]:
        if c not in codes_ok:
            raise SystemExit(f"missing issued code {c}")

    subprocess.check_call(["bash", str(RESIDUE_ROOT / "assemble_embeddings.sh")])
    for sub in ("ablang2_unpaired", "currab_unpaired"):
        if not (RESIDUE_ROOT / sub / "metadata.json").exists():
            raise SystemExit(f"BLOCKED missing asset {sub}")

    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    folds = load_folds(ROOT / "data/folds.csv")
    planned = df[df.execution_status.isin(["PLANNED", "FAILED", "FAILED_TECHNICAL"])].copy()
    if only_codes:
        planned = planned[planned.experiment_code.isin(only_codes)]
    rb = load_bundle_for_rows(df.to_dict(orient="records"), dev, test)

    status_rows = []
    for row in planned.to_dict(orient="records"):
        code = row["experiment_code"]
        if already_complete(code):
            df.loc[df.experiment_code == code, "execution_status"] = "COMPLETE"
            status_rows.append({"experiment_code": code, "status": "COMPLETE", "skipped": True})
            continue
        res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=quick, smoke=False)
        if res["status"] == "FAILED_TECHNICAL":
            print(f"retry once {code}", flush=True)
            res = run_cell(row, rb=rb, folds=folds, dev=dev, test=test, quick=quick, smoke=False)
        df.loc[df.experiment_code == code, "execution_status"] = res["status"]
        status_rows.append(res)
        save_plan(df)
    update_status(status_rows)
    print("train phase done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["status", "smoke", "train"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument(
        "--i-understand-full-batch",
        action="store_true",
        help="Required to run train without --codes (full 200-cell batch)",
    )
    args = ap.parse_args()
    if args.phase == "status":
        phase_status()
        return 0
    if args.phase == "smoke":
        return phase_smoke(quick=True if not args.quick else True, codes=args.codes)
    if args.phase == "train":
        if not args.codes and not args.i_understand_full_batch:
            raise SystemExit(
                "Refusing full 200-cell train without --i-understand-full-batch "
                "(or pass --codes for a subset)"
            )
        phase_train(quick=args.quick, only_codes=args.codes)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
