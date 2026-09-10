#!/usr/bin/env python3
"""Cheap T079 gate-zero diagnostic (NO retrain).

For each selected fold checkpoint of EXP-T079:
  - normal inference
  - inference with g_H=g_L forced to 0
Compare VAL OOF / TEST OOF / external fold MAE; optionally measure
||g * CrossAttn|| / ||H|| residual ratio on Dev batches.

Does not modify T080–T104 configs or retrain anything.
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
    load_residue_bundle,
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

TARGET = "TmApp"
CODE = "EXP-T079"
SEL_CSV = ROOT / "results" / f"{CODE}_SELECTED_LR.csv"
OUT_MD = ROOT / "results" / "T079_GATE_ZERO_DIAGNOSTIC.md"
OUT_CSV = ROOT / "results" / "T079_GATE_ZERO_DIAGNOSTIC.csv"

ARCH = normalize_arch_flags(
    {
        "use_cross_attention_bridge": True,
        "cross_gate_mode": "learned",
    }
)


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_ckpt(
    ckpt_path: Path,
    ids: list[str],
    rb,
    device: torch.device,
    *,
    force_gates_zero: bool = False,
    y_placeholder: np.ndarray | None = None,
) -> np.ndarray:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flags = normalize_arch_flags(blob.get("arch") or ARCH)
    cm = blob.get("content_mode", "frozen")
    mm = blob.get("merge_mode", "concat")
    ps = blob.get("plm_source", "ablingua")
    model = build_platform_model(
        rb, flags, content_mode=cm, merge_mode=mm, plm_source=ps
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    if force_gates_zero:
        if model.cross_gate_h is None or model.cross_gate_l is None:
            raise RuntimeError(f"{ckpt_path}: expected learned cross gates")
        model.cross_gate_h.fill_(0.0)
        model.cross_gate_l.fill_(0.0)
    mu = float(blob["mu"])
    sd = float(blob["sd"])
    y_arr = y_placeholder if y_placeholder is not None else np.zeros(len(ids), dtype=float)
    ds = AbDataset(ids, y_arr, rb, content_mode=cm, plm_source=ps or "ablingua")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    preds = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        batch.pop("y", None)
        batch.pop("fixed", None)
        pred_z = model(batch)
        preds.append((pred_z * sd + mu).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


@torch.no_grad()
def residual_ratio_for_ckpt(
    ckpt_path: Path,
    ids: list[str],
    rb,
    device: torch.device,
    max_batches: int = 4,
) -> dict:
    """Mean ||g * CrossAttn|| / ||pre-residual H|| over residue tokens.

    Definition: after layer-1 self-attn, residue streams H_res / L_res receive
    residual g_H * CrossAttn(H←L) and g_L * CrossAttn(L←H). Ratio is
    ||g * cross||_F / (||H_res||_F + eps) averaged over batches (and similarly L).
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flags = normalize_arch_flags(blob.get("arch") or ARCH)
    cm = blob.get("content_mode", "frozen")
    mm = blob.get("merge_mode", "concat")
    ps = blob.get("plm_source", "ablingua")
    model = build_platform_model(
        rb, flags, content_mode=cm, merge_mode=mm, plm_source=ps
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    g_h = float(model.cross_gate_h.detach().cpu())
    g_l = float(model.cross_gate_l.detach().cpu())

    y_arr = np.zeros(len(ids), dtype=float)
    ds = AbDataset(ids, y_arr, rb, content_mode=cm, plm_source=ps or "ablingua")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    ratios_h: list[float] = []
    ratios_l: list[float] = []
    layers = model.encoder.layers
    n_seen = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        mh = batch["heavy_mask"]
        ml = batch["light_mask"]
        x_h, pad_h = model._embed_chain_with_reg(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=mh,
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        x_l, pad_l = model._embed_chain_with_reg(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            mask=ml,
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        h1_h = layers[0](model.dropout(x_h), src_key_padding_mask=pad_h)
        h1_l = layers[0](model.dropout(x_l), src_key_padding_mask=pad_l)
        h_res = h1_h[:, 1:]
        l_res = h1_l[:, 1:]
        cross_h, _ = model.cross_attn(
            h_res, l_res, l_res, key_padding_mask=~ml, need_weights=False
        )
        cross_l, _ = model.cross_attn(
            l_res, h_res, h_res, key_padding_mask=~mh, need_weights=False
        )
        # Masked Frobenius over valid residues
        wh = mh.unsqueeze(-1).to(h_res.dtype)
        wl = ml.unsqueeze(-1).to(l_res.dtype)
        num_h = (model.cross_gate_h * cross_h * wh).norm()
        den_h = (h_res * wh).norm().clamp_min(1e-8)
        num_l = (model.cross_gate_l * cross_l * wl).norm()
        den_l = (l_res * wl).norm().clamp_min(1e-8)
        ratios_h.append(float(num_h / den_h))
        ratios_l.append(float(num_l / den_l))
        n_seen += 1
        if n_seen >= max_batches:
            break
    return {
        "g_H": g_h,
        "g_L": g_l,
        "residual_ratio_H_mean": float(np.mean(ratios_h)) if ratios_h else float("nan"),
        "residual_ratio_L_mean": float(np.mean(ratios_l)) if ratios_l else float("nan"),
        "n_batches_ratio": n_seen,
    }


def main() -> int:
    if not SEL_CSV.exists():
        raise SystemExit(f"missing {SEL_CSV}")
    # Protect T080–T104 configs from accidental mutation: this script never writes them.
    sel = pd.read_csv(SEL_CSV)
    device = torch.device(device_str())
    print("device", device, flush=True)

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    sol = load_solution(BUNDLE_ROOT / "solution.csv").set_index("id")
    y_map = {str(r["id"]): float(r[TARGET]) for _, r in dev.iterrows()}
    dev_ids = dev["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()
    y_dev = pd.Series(y_map)
    y_test = sol.loc[test_ids, TARGET].astype(float)

    oof_val_n = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_val_z = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_te_n = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}
    oof_te_z = {s: pd.Series(np.nan, index=dev_ids, dtype=float) for s in ("primary", "shadow")}

    rows = []
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

        pred_va_n = predict_ckpt(ckpt, va, rb, device, y_placeholder=y_va)
        pred_va_z = predict_ckpt(
            ckpt, va, rb, device, force_gates_zero=True, y_placeholder=y_va
        )
        pred_te_n = predict_ckpt(ckpt, te, rb, device, y_placeholder=y_te)
        pred_te_z = predict_ckpt(
            ckpt, te, rb, device, force_gates_zero=True, y_placeholder=y_te
        )
        pred_ext_n = predict_ckpt(ckpt, test_ids, rb, device)
        pred_ext_z = predict_ckpt(ckpt, test_ids, rb, device, force_gates_zero=True)

        oof_val_n[scheme].loc[va] = pred_va_n
        oof_val_z[scheme].loc[va] = pred_va_z
        oof_te_n[scheme].loc[te] = pred_te_n
        oof_te_z[scheme].loc[te] = pred_te_z

        ratio = residual_ratio_for_ckpt(ckpt, va, rb, device)

        mae_va_n = float(mae(y_va, pred_va_n))
        mae_va_z = float(mae(y_va, pred_va_z))
        mae_te_n = float(mae(y_te, pred_te_n))
        mae_te_z = float(mae(y_te, pred_te_z))
        mae_ext_n = float(mae(y_test.to_numpy(float), pred_ext_n))
        mae_ext_z = float(mae(y_test.to_numpy(float), pred_ext_z))
        delta_pred_va = float(np.mean(np.abs(pred_va_n - pred_va_z)))
        delta_pred_te = float(np.mean(np.abs(pred_te_n - pred_te_z)))
        delta_pred_ext = float(np.mean(np.abs(pred_ext_n - pred_ext_z)))

        rows.append(
            {
                "scheme": scheme,
                "fold": k,
                "checkpoint": str(ckpt),
                "g_H": ratio["g_H"],
                "g_L": ratio["g_L"],
                "residual_ratio_H_mean": ratio["residual_ratio_H_mean"],
                "residual_ratio_L_mean": ratio["residual_ratio_L_mean"],
                "val_mae_normal": mae_va_n,
                "val_mae_gate0": mae_va_z,
                "val_mae_delta": mae_va_z - mae_va_n,
                "test_oof_mae_normal": mae_te_n,
                "test_oof_mae_gate0": mae_te_z,
                "test_oof_mae_delta": mae_te_z - mae_te_n,
                "ext_fold_mae_normal": mae_ext_n,
                "ext_fold_mae_gate0": mae_ext_z,
                "ext_fold_mae_delta": mae_ext_z - mae_ext_n,
                "mean_abs_pred_delta_val": delta_pred_va,
                "mean_abs_pred_delta_test_oof": delta_pred_te,
                "mean_abs_pred_delta_ext": delta_pred_ext,
            }
        )
        print(
            f"{scheme} fold={k} ΔMAE_val={mae_va_z - mae_va_n:.6g} "
            f"ratio_H={ratio['residual_ratio_H_mean']:.3g} g_H={ratio['g_H']:.4g}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Aggregate OOF slice MAEs
    summary_lines = [
        "# T079 gate-zero diagnostic (no retrain)",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- source: `{SEL_CSV.relative_to(ROOT)}`",
        f"- definition residual ratio: `||g * CrossAttn||_F / ||H_pre||_F` on residue tokens "
        "(after layer-1; REG excluded), averaged over up to 4 VAL batches.",
        "- delta MAE = MAE(gate0) − MAE(normal); positive ⇒ zeroing gates worsens MAE.",
        "",
        "## Aggregate OOF MAE",
        "",
        "| scheme | split | MAE normal | MAE gate0 | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for scheme in ("primary", "shadow"):
        for name, sn, sz in (
            ("VAL OOF", oof_val_n, oof_val_z),
            ("TEST OOF", oof_te_n, oof_te_z),
        ):
            yn = y_dev.loc[dev_ids].to_numpy(float)
            mn = float(mae(yn, sn[scheme].loc[dev_ids].to_numpy(float)))
            mz = float(mae(yn, sz[scheme].loc[dev_ids].to_numpy(float)))
            summary_lines.append(
                f"| {scheme} | {name} | {mn:.6f} | {mz:.6f} | {mz - mn:.6g} |"
            )

    cols = [
        "scheme",
        "fold",
        "g_H",
        "g_L",
        "residual_ratio_H_mean",
        "residual_ratio_L_mean",
        "val_mae_delta",
        "test_oof_mae_delta",
        "ext_fold_mae_delta",
        "mean_abs_pred_delta_val",
    ]
    summary_lines += [
        "",
        "## Per-fold deltas",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        summary_lines.append("| " + " | ".join(cells) + " |")
    summary_lines += [
        "",
        "## Interpretation notes",
        "",
        "- Learned gates on T079 are typically near zero (see `EXP-T079_CROSS_GATES.csv`); "
        "small residual ratios and small prediction deltas are expected.",
        "- Residual ratio uses the trained g values (not forced to zero). "
        "If instrumentation fails on a fold, rely on mean absolute prediction delta columns.",
        "- This diagnostic does **not** alter T080–T104 configs or checkpoints.",
        "",
        f"CSV: `{OUT_CSV.relative_to(ROOT)}`",
        "",
    ]
    OUT_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("wrote", OUT_MD, OUT_CSV, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
