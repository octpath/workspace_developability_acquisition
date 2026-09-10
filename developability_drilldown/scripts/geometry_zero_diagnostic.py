#!/usr/bin/env python3
"""Inference geometry-zero ablation for ARCH-6G (NO retrain).

For each selected fold checkpoint of each geometry experiment:
  - normal inference
  - inference with cross_geom_weight forced to 0
Compare VAL OOF / TEST OOF / external fold MAE.

Codes: T105–T108, T118–T119, H064–H065, H078–H079.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import mae  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
    load_dev_test,
    load_folds,
    load_solution,
    tvt_split,
)
from antibody_transformer.protocol_v3 import (  # noqa: E402
    BATCH_SIZE,
    PLATFORM_ID,
    build_platform_model,
    normalize_arch_flags,
)
from antibody_transformer.training import AbDataset, _batch_to_device, collate_batch  # noqa: E402
from classical_features.ca_cache import build_or_load_ca_cache  # noqa: E402
from antibody_transformer.data import attach_ca_coords, load_residue_bundle  # noqa: E402
from preregister_t105_hic_batch import SERIES  # noqa: E402

GEOM_CODES = [
    "EXP-T105",
    "EXP-T106",
    "EXP-T107",
    "EXP-T108",
    "EXP-T118",
    "EXP-T119",
    "EXP-H064",
    "EXP-H065",
    "EXP-H078",
    "EXP-H079",
]


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def spec_for(code: str) -> dict:
    for s in SERIES:
        if s["code"] == code:
            return s
    raise KeyError(code)


def prepare_rb(spec: dict, dev, test):
    need_ablingua = spec["plm_source"] == "ablingua"
    need_ablang2 = spec["plm_source"] == "ablang2"
    need_esm2 = spec["plm_source"] == "esm2"
    rb = load_residue_bundle(
        dev,
        test,
        need_ablingua=need_ablingua,
        need_ablang2=need_ablang2,
        need_esm2=need_esm2,
    )
    ca = build_or_load_ca_cache(dev, test)
    seqs = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    attach_ca_coords(
        rb,
        ca_heavy=ca["H"],
        ca_light=ca["L"],
        ca_ids=ca["ids"],
        seqs=seqs,
    )
    return rb


@torch.no_grad()
def predict_ckpt(
    ckpt_path: Path,
    ids: list[str],
    rb,
    device: torch.device,
    *,
    force_geom_zero: bool = False,
    y_placeholder: np.ndarray | None = None,
    content_mode: str,
    plm_source: str | None,
) -> np.ndarray:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flags = normalize_arch_flags(blob.get("arch") or {})
    cm = blob.get("content_mode", content_mode)
    mm = blob.get("merge_mode", "concat")
    ps = blob.get("plm_source", plm_source)
    model = build_platform_model(
        rb, flags, content_mode=cm, merge_mode=mm, plm_source=ps
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    if force_geom_zero:
        if model.cross_geom_weight is None:
            raise RuntimeError(f"{ckpt_path}: expected cross_geom_weight")
        model.cross_geom_weight.data.zero_()
    mu = float(blob["mu"])
    sd = float(blob["sd"])
    y_arr = y_placeholder if y_placeholder is not None else np.zeros(len(ids), dtype=float)
    ds = AbDataset(
        ids,
        y_arr,
        rb,
        content_mode=cm,
        plm_source=ps or "ablingua",
        use_cross_geometry_bias=True,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    preds = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        batch.pop("y", None)
        batch.pop("fixed", None)
        pred_z = model(batch)
        preds.append((pred_z * sd + mu).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def run_one(code: str, device: torch.device) -> pd.DataFrame:
    spec = spec_for(code)
    target = spec["target"]
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    print(f"==== geom-zero {code} ({spec['description']}) ====", flush=True)

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(spec, dev, test)
    sol = load_solution(BUNDLE_ROOT / "solution.csv").set_index("id")
    y_map = {str(r["id"]): float(r[target]) for _, r in dev.iterrows()}
    dev_ids = dev["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()
    y_dev = pd.Series(y_map)
    y_test = sol.loc[test_ids, target].astype(float)

    oof_val_n = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_val_z = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_te_n = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_te_z = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}

    rows = []
    cm = spec["content_mode"]
    ps = spec["plm_source"]
    for _, row in sel.iterrows():
        scheme = str(row["scheme"])
        k = int(row["fold"])
        ckpt = Path(str(row["checkpoint"]))
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        fmap = folds.primary if scheme == "primary" else folds.shadow
        tr, va, te = tvt_split(fmap, k, dev_ids)
        y_va = np.asarray([y_map[a] for a in va], float)
        y_te = np.asarray([y_map[a] for a in te], float)

        pred_va_n = predict_ckpt(
            ckpt, va, rb, device, y_placeholder=y_va, content_mode=cm, plm_source=ps
        )
        pred_va_z = predict_ckpt(
            ckpt,
            va,
            rb,
            device,
            force_geom_zero=True,
            y_placeholder=y_va,
            content_mode=cm,
            plm_source=ps,
        )
        pred_te_n = predict_ckpt(
            ckpt, te, rb, device, y_placeholder=y_te, content_mode=cm, plm_source=ps
        )
        pred_te_z = predict_ckpt(
            ckpt,
            te,
            rb,
            device,
            force_geom_zero=True,
            y_placeholder=y_te,
            content_mode=cm,
            plm_source=ps,
        )
        pred_ext_n = predict_ckpt(
            ckpt, test_ids, rb, device, content_mode=cm, plm_source=ps
        )
        pred_ext_z = predict_ckpt(
            ckpt,
            test_ids,
            rb,
            device,
            force_geom_zero=True,
            content_mode=cm,
            plm_source=ps,
        )

        oof_val_n[scheme].loc[va] = pred_va_n
        oof_val_z[scheme].loc[va] = pred_va_z
        oof_te_n[scheme].loc[te] = pred_te_n
        oof_te_z[scheme].loc[te] = pred_te_z

        mae_va_n = float(mae(y_va, pred_va_n))
        mae_va_z = float(mae(y_va, pred_va_z))
        mae_te_n = float(mae(y_te, pred_te_n))
        mae_te_z = float(mae(y_te, pred_te_z))
        mae_ext_n = float(mae(y_test.to_numpy(float), pred_ext_n))
        mae_ext_z = float(mae(y_test.to_numpy(float), pred_ext_z))

        # weight norms
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        w = None
        for kname, t in blob["model"].items():
            if kname.endswith("cross_geom_weight"):
                w = t.detach().float().numpy()
                break
        w_norm = float(np.linalg.norm(w)) if w is not None else float("nan")

        rows.append(
            {
                "code": code,
                "target": target,
                "representation": spec["representation"],
                "merge": spec["merge_mode"],
                "scheme": scheme,
                "fold": k,
                "w_frobenius": w_norm,
                "val_mae_normal": mae_va_n,
                "val_mae_geom0": mae_va_z,
                "val_mae_delta": mae_va_z - mae_va_n,
                "test_oof_mae_normal": mae_te_n,
                "test_oof_mae_geom0": mae_te_z,
                "test_oof_mae_delta": mae_te_z - mae_te_n,
                "ext_fold_mae_normal": mae_ext_n,
                "ext_fold_mae_geom0": mae_ext_z,
                "ext_fold_mae_delta": mae_ext_z - mae_ext_n,
                "mean_abs_pred_delta_val": float(np.mean(np.abs(pred_va_n - pred_va_z))),
                "mean_abs_pred_delta_test_oof": float(np.mean(np.abs(pred_te_n - pred_te_z))),
                "mean_abs_pred_delta_ext": float(np.mean(np.abs(pred_ext_n - pred_ext_z))),
            }
        )
        print(
            f"  {scheme} k={k} ΔMAE_test={mae_te_z - mae_te_n:+.5f} "
            f"|Δpred|_val={np.mean(np.abs(pred_va_n - pred_va_z)):.5g} ||w||={w_norm:.4g}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    # append aggregate OOF rows
    agg = []
    for scheme in ("primary", "shadow"):
        yn = y_dev.loc[dev_ids].to_numpy(float)
        for split, sn, sz in (
            ("VAL_OOF", oof_val_n, oof_val_z),
            ("TEST_OOF", oof_te_n, oof_te_z),
        ):
            mn = float(mae(yn, sn[scheme].loc[dev_ids].to_numpy(float)))
            mz = float(mae(yn, sz[scheme].loc[dev_ids].to_numpy(float)))
            agg.append(
                {
                    "code": code,
                    "target": target,
                    "representation": spec["representation"],
                    "merge": spec["merge_mode"],
                    "scheme": scheme,
                    "fold": -1,
                    "split": split,
                    "mae_normal": mn,
                    "mae_geom0": mz,
                    "mae_delta": mz - mn,
                }
            )
    return df, pd.DataFrame(agg)


def main() -> int:
    device = torch.device(device_str())
    print("device", device, "platform", PLATFORM_ID, flush=True)
    all_fold = []
    all_agg = []
    for code in GEOM_CODES:
        fold_df, agg_df = run_one(code, device)
        all_fold.append(fold_df)
        all_agg.append(agg_df)
    fold = pd.concat(all_fold, ignore_index=True)
    agg = pd.concat(all_agg, ignore_index=True)
    fold_path = ROOT / "results" / "TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.csv"
    agg_path = ROOT / "results" / "TM_HIC_GEOMETRY_ZERO_AGGREGATE.csv"
    fold.to_csv(fold_path, index=False)
    agg.to_csv(agg_path, index=False)

    md_lines = [
        "# Geometry-zero inference ablation (ARCH-6G, no retrain)",
        "",
        f"- platform: `{PLATFORM_ID}`",
        "- definition: set `cross_geom_weight = 0` at inference; attention path otherwise unchanged.",
        "- ΔMAE = MAE(geom0) − MAE(normal); positive ⇒ zeroing geometry worsens MAE (geometry used).",
        "",
        "## Aggregate OOF",
        "",
        "| code | scheme | split | MAE normal | MAE geom0 | Δ |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        md_lines.append(
            f"| {r['code']} | {r['scheme']} | {r['split']} | "
            f"{r['mae_normal']:.6f} | {r['mae_geom0']:.6f} | {r['mae_delta']:.6g} |"
        )
    md_lines += [
        "",
        f"Per-fold detail: `{fold_path.name}`",
        "",
    ]
    (ROOT / "results" / "TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # Patch geometry reports with D section summary
    for report, codes in (
        (
            ROOT / "results" / "TM_GEOMETRY_REPORT.md",
            [c for c in GEOM_CODES if c.startswith("EXP-T")],
        ),
        (
            ROOT / "results" / "HIC_GEOMETRY_REPORT.md",
            [c for c in GEOM_CODES if c.startswith("EXP-H")],
        ),
    ):
        text = report.read_text(encoding="utf-8")
        block = [
            "",
            "## D. Inference geometry-zero ablation",
            "",
            "Source: `TM_HIC_GEOMETRY_ZERO_AGGREGATE.csv` / `TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.md`.",
            "",
            "| code | Primary TEST_OOF Δ | Shadow TEST_OOF Δ |",
            "|---|---:|---:|",
        ]
        for code in codes:
            sub = agg[(agg.code == code) & (agg.split == "TEST_OOF")]
            p = sub[sub.scheme == "primary"]["mae_delta"]
            s = sub[sub.scheme == "shadow"]["mae_delta"]
            pdlt = float(p.iloc[0]) if len(p) else float("nan")
            sdlt = float(s.iloc[0]) if len(s) else float("nan")
            block.append(f"| {code} | {pdlt:.6g} | {sdlt:.6g} |")
        marker = "## Diagnostics checklist"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n" + "\n".join(block) + "\n\n" + marker + text.split(marker, 1)[1]
        else:
            text = text.rstrip() + "\n" + "\n".join(block) + "\n"
        # replace placeholder D line
        text = text.replace(
            "- D. inference geometry-zero ablation: re-score with `cross_geom_weight=0` from checkpoints if needed",
            "- D. inference geometry-zero ablation: see section above (`TM_HIC_GEOMETRY_ZERO_*`)",
        )
        report.write_text(text, encoding="utf-8")

    print("wrote", fold_path, agg_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
