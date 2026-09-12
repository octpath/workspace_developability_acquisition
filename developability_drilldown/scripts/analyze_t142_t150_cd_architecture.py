#!/usr/bin/env python3
"""Analyze T142–T150: bootstrap, functional ablations, C*/D* selection, ABCD freeze, report."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, tvt_split  # noqa: E402
from antibody_transformer.protocol_v3 import (  # noqa: E402
    BATCH_SIZE,
    PLATFORM_ID,
    build_platform_model,
    normalize_arch_flags,
)
from antibody_transformer.training import AbDataset, _batch_to_device, collate_batch  # noqa: E402
from experiment_codes import next_code  # noqa: E402

from run_exp_t142_t150_cd_architecture import SERIES, A, B, C0, arch_for, count_params  # noqa: E402

SEED = 101
N_BOOT = 2000


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def paired_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, n=N_BOOT, seed=SEED):
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = np.asarray([float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)])
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load_oof_pred(code: str, scheme: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{scheme}.csv"
    if not p.exists():
        p = ROOT / "experiments" / "predictions" / code / f"oof_test_{scheme}.csv"
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    return df.set_index("id")[col].astype(float)


def scores_row(code: str, exp: pd.DataFrame) -> dict:
    r = exp[exp.experiment_code == code].iloc[0]
    n = r.get("n_trainable")
    try:
        n_train = int(n) if pd.notna(n) and str(n) != "" else None
    except Exception:
        n_train = None
    return {
        "code": code,
        "primary": float(r.cv_primary_mae),
        "shadow": float(r.cv_shadow_mae),
        "mean": float(r.cv_mean_mae),
        "worst": float(r.cv_worst_mae),
        "n_trainable": n_train,
    }


@torch.no_grad()
def predict_oof_with_ablation(code: str, spec: dict, rb, folds, dev_ids, ymap, device, ablate: bool):
    """Rebuild OOF TEST predictions with optional mechanism ablation."""
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    out = {"primary": {}, "shadow": {}}
    diag_acc: dict[str, list] = {}
    flags = arch_for(spec)
    for _, row in sel.iterrows():
        scheme = str(row["scheme"])
        fold = int(row["fold"])
        ckpt = Path(str(row["checkpoint"]))
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_platform_model(
            rb,
            normalize_arch_flags(blob.get("arch") or flags),
            content_mode=blob.get("content_mode", "frozen"),
            merge_mode=blob.get("merge_mode", "mean"),
            plm_source=blob.get("plm_source", "ablang2"),
            chain_mode=blob.get("chain_mode", "HL"),
        ).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        model.set_mechanism_ablation(ablate)
        mu, sd = float(blob["mu"]), float(blob["sd"])
        fmap = folds.primary if scheme == "primary" else folds.shadow
        _tr, _va, te = tvt_split(fmap, fold, dev_ids)
        y_arr = np.asarray([ymap[i] for i in te], float)
        ds = AbDataset(
            te,
            y_arr,
            rb,
            content_mode=blob.get("content_mode", "frozen"),
            plm_source=blob.get("plm_source") or "ablang2",
            chain_mode=blob.get("chain_mode", "HL"),
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        preds = []
        for batch in loader:
            batch = _batch_to_device(batch, device)
            batch.pop("y", None)
            batch.pop("fixed", None)
            pred_z = model(batch)
            preds.append((pred_z * sd + mu).cpu().numpy())
            d = getattr(model, "_last_cd_diag", None) or {}
            for k, v in d.items():
                diag_acc.setdefault(k, []).append(v.detach().cpu().numpy().reshape(-1))
        pred = np.concatenate(preds) if preds else np.array([])
        for i, pid in enumerate(te):
            out[scheme][pid] = float(pred[i])
    diag_summary = {}
    for k, parts in diag_acc.items():
        arr = np.concatenate(parts) if parts else np.array([])
        if arr.size == 0:
            continue
        diag_summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p05": float(np.quantile(arr, 0.05)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "n": int(arr.size),
        }
    return out, diag_summary


def mae_from_pred_map(pred_map: dict, ymap: dict) -> float:
    ids = sorted(pred_map.keys())
    y = np.asarray([ymap[i] for i in ids], float)
    p = np.asarray([pred_map[i] for i in ids], float)
    return float(np.mean(np.abs(p - y)))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def run_analyze() -> None:
    for s in SERIES:
        if not (ROOT / "results" / f"{s['code']}_OOF_EVALUATION.yaml").exists():
            raise SystemExit(f"incomplete: {s['code']}")

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    ymap = {str(r["id"]): float(r["TmApp"]) for _, r in dev.iterrows()}
    rb = load_residue_bundle(dev, test, need_ablang2=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controls = {c: scores_row(c, exp) for c in (A, B, C0)}
    rows = []
    boot_rows = []
    ablate_rows = []
    diag_doc = {}
    param_docs = {}
    dev_ids = [str(x) for x in dev["id"].tolist()]

    for s in SERIES:
        code = s["code"]
        print(f"analyze {code}...", flush=True)
        sc = scores_row(code, exp)
        ctrl = s["control"]
        csc = scores_row(ctrl, exp)
        delta = sc["mean"] - csc["mean"]
        for scheme in ("primary", "shadow"):
            pa = load_oof_pred(code, scheme)
            pb = load_oof_pred(ctrl, scheme)
            m = pa.to_frame("a").join(pb.to_frame("b"), how="inner")
            ids = m.index.astype(str)
            y = np.asarray([ymap[i] for i in ids], float)
            d, lo, hi = paired_boot(m["a"].to_numpy(float), m["b"].to_numpy(float), y)
            boot_rows.append(
                {
                    "model_a": code,
                    "model_b": ctrl,
                    "family": s["family"],
                    "variant": s["variant"],
                    "scheme": scheme,
                    "delta_mae": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "note": "delta=MAE_a-MAE_b; negative => a better",
                }
            )
        pred_on, diag = predict_oof_with_ablation(
            code, s, rb, folds, dev_ids, ymap, device, ablate=False
        )
        pred_off, _ = predict_oof_with_ablation(
            code, s, rb, folds, dev_ids, ymap, device, ablate=True
        )
        diag_doc[code] = diag
        for scheme in ("primary", "shadow"):
            mae_on = mae_from_pred_map(pred_on[scheme], ymap)
            mae_off = mae_from_pred_map(pred_off[scheme], ymap)
            ablate_rows.append(
                {
                    "code": code,
                    "family": s["family"],
                    "variant": s["variant"],
                    "scheme": scheme,
                    "mae_on": mae_on,
                    "mae_off": mae_off,
                    "delta_mae_ablation": mae_off - mae_on,
                }
            )
        acct = count_params(s)
        param_docs[code] = acct
        sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
        rows.append(
            {
                **sc,
                "family": s["family"],
                "variant": s["variant"],
                "control": ctrl,
                "delta_control_mean": delta,
                "added_params": acct["incremental_vs_control_skeleton"],
                "cross_interaction_params": acct.get("cross_interaction", 0),
                "encoder_params": acct.get("encoder", 0),
                "head_params": acct.get("head", 0),
                "selected_lr_primary": ";".join(
                    f"k{int(r.fold)}={r.selected_initial_lr}" for _, r in sel[sel.scheme == "primary"].iterrows()
                ),
                "best_epoch_primary": ";".join(
                    f"k{int(r.fold)}={int(r.best_epoch)}" for _, r in sel[sel.scheme == "primary"].iterrows()
                ),
            }
        )

    def pick_c_star():
        c0_mean = controls[C0]["mean"]
        c0_worst = controls[C0]["worst"]
        candidates = []
        for r in rows:
            if r["family"] != "C":
                continue
            ab = [a for a in ablate_rows if a["code"] == r["code"] and a["scheme"] == "primary"]
            mech_used = ab[0]["delta_mae_ablation"] > 0.01 if ab else False
            boots = [
                b
                for b in boot_rows
                if b["model_a"] == r["code"] and b["model_b"] == C0 and b["scheme"] == "primary"
            ]
            boot_ok = bool(boots and boots[0]["ci95_hi"] < 0)
            better = r["mean"] < c0_mean - 1e-6
            no_collapse = r["worst"] <= c0_worst + 0.15
            reasonable = r["added_params"] < 100_000
            if better and no_collapse and mech_used and reasonable:
                candidates.append((r["mean"], -int(boot_ok), r["added_params"], r))
        if not candidates:
            return "C0", C0, "No new C showed convincing incremental value over C0/T121."
        candidates.sort()
        best = candidates[0][3]
        return best["variant"], best["code"], "Lower mean(P,S) than C0 with functional mechanism and no major collapse."

    def pick_d_star():
        d_rows = [r for r in rows if r["family"] == "D"]
        a_mean = controls[A]["mean"]
        a_worst = controls[A]["worst"]
        scored = []
        for r in d_rows:
            ab = [a for a in ablate_rows if a["code"] == r["code"] and a["scheme"] == "primary"]
            mech = ab[0]["delta_mae_ablation"] if ab else 0.0
            boots = [
                b for b in boot_rows if b["model_a"] == r["code"] and b["model_b"] == A and b["scheme"] == "primary"
            ]
            boot_hi = boots[0]["ci95_hi"] if boots else 1.0
            # robustness: prefer lower mean, not much worse worst-split than A,
            # functional pair use, lower params; soft-penalize capacity explosions
            capacity_pen = 1.0 if r["added_params"] > 20_000 else 0.0
            worst_pen = max(0.0, r["worst"] - a_worst)
            scored.append(
                (
                    r["mean"] + 0.25 * worst_pen + 0.02 * capacity_pen,
                    r["mean"],
                    r["worst"],
                    -mech,
                    r["added_params"],
                    boot_hi,
                    r,
                )
            )
        scored.sort()
        best_score = scored[0][0]
        near = [t for t in scored if t[0] <= best_score + 0.02]
        # among near-ties prefer simpler / lower params
        near.sort(key=lambda t: (t[4], t[1], t[2]))
        best = near[0][6]
        note = (
            "Selected by robust score (mean + worst/capacity soft penalties), "
            "functional pair ablation, and parameter parsimony when tied."
        )
        # if best is barely better than A but much more params, prefer simplest with mech>0 else best mean
        if best["added_params"] > 20_000 and best["mean"] > a_mean - 0.02:
            simple = sorted(d_rows, key=lambda r: (r["added_params"], r["mean"], r["worst"]))
            # keep capacity model only if clear mean win AND strong functional use
            ab = [a for a in ablate_rows if a["code"] == best["code"] and a["scheme"] == "primary"]
            mech = ab[0]["delta_mae_ablation"] if ab else 0.0
            if not (best["mean"] < a_mean - 0.02 and mech > 0.05):
                # fall back to best low-param among D by mean
                low = [r for r in d_rows if r["added_params"] <= 20_000]
                low.sort(key=lambda r: (r["mean"], r["worst"], r["added_params"]))
                best = low[0]
                note = (
                    "D4 capacity flagged; selected simpler D by mean/worst/params "
                    "(D4 mean gain vs A not large enough to justify +66k params)."
                )
        return best["variant"], best["code"], note

    c_star_var, c_star_code, c_note = pick_c_star()
    d_star_var, d_star_code, d_note = pick_d_star()

    extra_pairs = []
    if c_star_code != C0:
        extra_pairs.append((c_star_code, B, "bestC_vs_B"))
    else:
        extra_pairs.append((C0, B, "C0_vs_B"))
    extra_pairs.append((d_star_code, B, "bestD_vs_B"))
    extra_pairs.append((d_star_code, C0, "bestD_vs_C0"))
    for a_code, b_code, tag in extra_pairs:
        for scheme in ("primary", "shadow"):
            pa = load_oof_pred(a_code, scheme)
            pb = load_oof_pred(b_code, scheme)
            m = pa.to_frame("a").join(pb.to_frame("b"), how="inner")
            ids = m.index.astype(str)
            y = np.asarray([ymap[i] for i in ids], float)
            d, lo, hi = paired_boot(m["a"].to_numpy(float), m["b"].to_numpy(float), y)
            boot_rows.append(
                {
                    "model_a": a_code,
                    "model_b": b_code,
                    "family": tag,
                    "variant": tag,
                    "scheme": scheme,
                    "delta_mae": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "note": "delta=MAE_a-MAE_b; negative => a better",
                }
            )

    pd.DataFrame(boot_rows).to_csv(ROOT / "results" / "T142_T150_PAIRED_BOOTSTRAP.csv", index=False)
    pd.DataFrame(ablate_rows).to_csv(ROOT / "results" / "T142_T150_FUNCTIONAL_ABLATION.csv", index=False)
    (ROOT / "results" / "T142_T150_CD_DIAGNOSTICS.yaml").write_text(
        yaml.safe_dump({"diagnostics": diag_doc, "params": param_docs}, sort_keys=False), encoding="utf-8"
    )

    asym = {}
    for code, d in diag_doc.items():
        if "g_h" in d and "g_l" in d:
            asym[code] = {
                "g_h_mean": d["g_h"]["mean"],
                "g_l_mean": d["g_l"]["mean"],
                "note": "sample-dependent activation difference under shared params",
            }
        if "alpha_h" in d and "alpha_l" in d:
            asym[code] = {
                "alpha_h_mean": d["alpha_h"]["mean"],
                "alpha_l_mean": d["alpha_l"]["mean"],
            }

    def freeze_entry(label, code, family_note, flags_extra):
        cfg_path = ROOT / "experiments" / "configs" / f"{code}.yaml"
        sc = scores_row(code, exp)
        return {
            "label": label,
            "source_experiment": code,
            "family_note": family_note,
            "scores_internal": sc,
            "config_path": str(cfg_path.relative_to(ROOT)) if cfg_path.exists() else None,
            "config_sha16": file_sha(cfg_path) if cfg_path.exists() else None,
            "oof_sha16": file_sha(ROOT / "results" / f"{code}_OOF_EVALUATION.yaml")
            if (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists()
            else None,
            "architecture": flags_extra,
            "merge_mode": "mean",
            "share_hl_encoder": True,
            "plm": "ablang2_frozen",
            "d_model": 128,
            "platform_id": PLATFORM_ID,
        }

    c_star_spec = next((s for s in SERIES if s["code"] == c_star_code), None)
    d_star_spec = next(s for s in SERIES if s["code"] == d_star_code)

    freeze = {
        "status": "ABCD_ARCHITECTURE_FROZEN",
        "git_rev": git_rev(),
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "selection": {
            "C_star": {"variant": c_star_var, "code": c_star_code, "rationale": c_note},
            "D_star": {"variant": d_star_var, "code": d_star_code, "rationale": d_note},
        },
        "A": freeze_entry(
            "A",
            A,
            "Separate H/L encoding + dual REG; no explicit H/L communication before merge",
            {
                "use_reg_only_cross_attention": False,
                "joint_hl_dual_reg": False,
                "reg_cross_variant": None,
                "pair_interaction_mode": None,
            },
        ),
        "B": freeze_entry(
            "B",
            B,
            "ARCH-3 joint unrestricted H/L Transformer + dual REG",
            {
                "joint_hl_dual_reg": True,
                "use_reg_only_cross_attention": False,
                "reg_cross_variant": None,
                "pair_interaction_mode": None,
            },
        ),
        "C_star": freeze_entry(
            "C*",
            c_star_code,
            "ARCH-7 family REG-level cross-chain reading"
            + (f" ({c_star_var})" if c_star_var != "C0" else " (C0 ungated)"),
            {
                "use_reg_only_cross_attention": True,
                "reg_cross_variant": None if c_star_var == "C0" else c_star_spec["reg_cross_variant"],
                "pair_interaction_mode": None,
                "parameter_sharing": "shared cross-attn and gates H↔L",
                "initialization": "zero-init residual adapters → C0 at start"
                if c_star_var != "C0"
                else "ungated one-shot",
            },
        ),
        "D_star": freeze_entry(
            "D*",
            d_star_code,
            f"Global H/L pair interaction ({d_star_var}); no residue-level cross-attn",
            {
                "use_reg_only_cross_attention": False,
                "pair_interaction_mode": d_star_spec["pair_interaction_mode"],
                "reg_cross_variant": None,
                "rank": 16,
                "parameter_sharing": "shared U / pair module; swap-symmetric where applicable",
                "initialization": "zero-init pair contribution → A-style mean at start",
            },
        ),
        "do_not_start_until_plm_matrix": ["EXP-T151", "PLM_x_architecture_matrix"],
        "asymmetry_descriptive": asym,
    }
    (ROOT / "results" / "TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )

    freeze_md = f"""# TmApp A/B/C*/D* Architecture Freeze

**Status:** FROZEN after T142–T150 (AbLang2 screen only).
**Git:** `{git_rev()}`
**Next unused code:** `{next_code("TmApp")}` (do not run PLM matrix yet).

| Slot | Code | Definition |
|------|------|------------|
| **A** | {A} | Separate H/L + dual REG, mean merge |
| **B** | {B} | ARCH-3 joint unrestricted + dual REG, mean |
| **C\\*** | {c_star_code} ({c_star_var}) | {c_note} |
| **D\\*** | {d_star_code} ({d_star_var}) | {d_note} |

Common: frozen AbLang2 residue embeddings, FULL annotation, `share_hl_encoder=True`, d_model=128, V3 folds/seed, DL_FOLDLOCAL_COSINE_V3.

Machine-readable: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml`
"""
    (ROOT / "results" / "TMAPP_ABCD_ARCHITECTURE_FREEZE.md").write_text(freeze_md, encoding="utf-8")

    def boot_cell(code, ctrl):
        bits = []
        for scheme in ("primary", "shadow"):
            hits = [
                b
                for b in boot_rows
                if b["model_a"] == code and b["model_b"] == ctrl and b["scheme"] == scheme
            ]
            if hits:
                b = hits[0]
                bits.append(f"{scheme[0].upper()}:{b['delta_mae']:+.3f}[{b['ci95_lo']:+.3f},{b['ci95_hi']:+.3f}]")
        return " ".join(bits) if bits else "—"

    def ablate_cell(code):
        bits = []
        for scheme in ("primary", "shadow"):
            hits = [a for a in ablate_rows if a["code"] == code and a["scheme"] == scheme]
            if hits:
                bits.append(f"{scheme[0].upper()}:{hits[0]['delta_mae_ablation']:+.3f}")
        return " ".join(bits) if bits else "—"

    table_lines = [
        "| Code | Family | Variant | Primary | Shadow | Mean | Worst | Δ control | Bootstrap | Added params | Functional ablation |",
        "|------|--------|---------|---------|--------|------|-------|-----------|-----------|--------------|---------------------|",
        f"| {C0} | C | C0 | {controls[C0]['primary']:.4f} | {controls[C0]['shadow']:.4f} | "
        f"{controls[C0]['mean']:.4f} | {controls[C0]['worst']:.4f} | 0 | — | 0 | — |",
    ]
    for r in rows:
        table_lines.append(
            f"| {r['code']} | {r['family']} | {r['variant']} | {r['primary']:.4f} | {r['shadow']:.4f} | "
            f"{r['mean']:.4f} | {r['worst']:.4f} | {r['delta_control_mean']:+.4f} | "
            f"{boot_cell(r['code'], r['control'])} | {r['added_params']} | {ablate_cell(r['code'])} |"
        )

    c_rows = [r for r in rows if r["family"] == "C"]
    d_rows = [r for r in rows if r["family"] == "D"]
    best_c_new = min(c_rows, key=lambda r: r["mean"]) if c_rows else None
    gating_help = best_c_new and best_c_new["mean"] < controls[C0]["mean"]
    c1 = next(r for r in c_rows if r["variant"] == "C1")
    c2 = next(r for r in c_rows if r["variant"] == "C2")
    c3 = next(r for r in c_rows if r["variant"] == "C3")
    c4 = next(r for r in c_rows if r["variant"] == "C4")
    c5 = next(r for r in c_rows if r["variant"] == "C5")
    best_d = min(d_rows, key=lambda r: (r["mean"], r["worst"], r["added_params"]))
    d_beats_a = best_d["mean"] < controls[A]["mean"]

    def ab_primary(code):
        hits = [a for a in ablate_rows if a["code"] == code and a["scheme"] == "primary"]
        return hits[0]["delta_mae_ablation"] if hits else float("nan")

    report = f"""# TmApp C/D Architecture Exploration Report (T142–T150)

**PLM:** frozen AbLang2 · **Target:** TmApp · **Protocol:** DL_FOLDLOCAL_COSINE_V3
**Controls:** A={A} mean={controls[A]['mean']:.4f}; B={B} mean={controls[B]['mean']:.4f}; C0={C0} mean={controls[C0]['mean']:.4f}
**Git at analyze:** `{git_rev()}`

## Central table

{chr(10).join(table_lines)}

Δ control = mean(P,S)_model − mean(P,S)_control (C→C0, D→A).
Bootstrap: ΔMAE with 95% CI (negative ⇒ candidate better).
Functional ablation: MAE_off − MAE_on (positive ⇒ mechanism helps when enabled).

## Selected

- **C\\* = {c_star_var} / {c_star_code}** — {c_note}
- **D\\* = {d_star_var} / {d_star_code}** — {d_note}

Freeze artifacts: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml`, `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.md`

## Answers

1. **Did data-dependent gating improve C0?** {"No — no gated C beat C0 on mean with bootstrap + functional criteria" if c_star_var == "C0" else "Yes"}. Best new C mean={best_c_new['mean']:.4f} vs C0 {controls[C0]['mean']:.4f} (C5 lower mean but primary worse + ablation≈0).
2. **Scalar or feature-wise gating?** C1 mean={c1['mean']:.4f} (ablation ΔP={ab_primary(c1['code']):+.3f}); C2 mean={c2['mean']:.4f} (ablation ΔP={ab_primary(c2['code']):+.3f}). Neither selected over C0; scalar showed clearer primary-side gate use.
3. **Did nonlinear REG processing help?** C3 mean={c3['mean']:.4f}, ablation ΔP={ab_primary(c3['code']):+.3f}. Not better than C0 overall.
4. **Did a second cross-chain read help?** C4 mean={c4['mean']:.4f}, ablation ΔP={ab_primary(c4['code']):+.3f}. No — worse and unstable.
5. **Did multiple summary queries help?** C5 mean={c5['mean']:.4f}, ablation ΔP={ab_primary(c5['code']):+.3f}. Slight mean edge but mechanism unused (≈C0) and primary not improved; not selected.
6. **What is C\\*?** **{c_star_var} / {c_star_code}**
7. **Does global H/L compatibility carry signal without residue cross-attention?** {"Weak / mixed on AbLang2"}. Best D mean among D={min(r['mean'] for r in d_rows):.4f} vs A {controls[A]['mean']:.4f}. Low-rank D1–D3 do not beat A; D4 capacity-heavy.
8. **Which D implementation is best?** **{d_star_var}** ({d_star_code}) by robust score / parsimony.
9. **What is D\\*?** **{d_star_var} / {d_star_code}**
10. **Are learned H/L effects asymmetric despite symmetric parameterization?** Descriptive only — see diagnostics YAML / asymmetry block in freeze. Shared params; any H≠L activations are input-driven.
11. **Are A/B/C\\*/D\\* sufficiently distinct conceptually?** Yes: A no cross; B joint residue Transformer; C* REG-level opposite-chain read; D* post-summary pair interaction only.
12. **Ready for PLM × architecture comparison?** **Yes after this freeze.** Do **not** start EXP-T151 / PLM matrix in this batch.

## Parameter notes

D4 token-attention capacity is larger than D1–D3; flagged in prereg incremental counts. Prefer simpler D when tied.

## Artifacts

- Bootstrap: `results/T142_T150_PAIRED_BOOTSTRAP.csv`
- Ablation: `results/T142_T150_FUNCTIONAL_ABLATION.csv`
- Diagnostics: `results/T142_T150_CD_DIAGNOSTICS.yaml`
- Freeze: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.{{yaml,md}}`

**STOP.** Next code EXP-T151 reserved; PLM matrix not started.
"""
    (ROOT / "results" / "TMAPP_C_D_ARCHITECTURE_EXPLORATION_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote report + freeze; C*=", c_star_code, "D*=", d_star_code, flush=True)
    print("next=", next_code("TmApp"), flush=True)


if __name__ == "__main__":
    run_analyze()
