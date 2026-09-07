#!/usr/bin/env python3
"""Simple Train/Val/Test CV: BASE vs BASE+STRUCTURE (orthogonal to STRICT nested).

Does not touch FeNNix production workers/caches.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
GAP = FP / "structure_gap_closure"
OUT = FP / "fennix_fab_context_interim_audit"
RES = OUT / "results"
OOF_DIR = ROOT / "virtual_participant/stage5_integration/oof"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
ARO = GAP / "cache/aromatic_features_esmfold.csv"
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
REGIONS = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"
STRICT_CSV = OUT / "STRICT_NESTED_INCREMENT_RESULTS.csv"
TMAPP_REF = "TmApp__META_performance__ridge_100.0"
HIC_REF = "HIC__SIMPLE_blend_seq_surf_adv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
B_BOOT = 10000
RNG = np.random.default_rng(42)
MIN_TR, MIN_VA, MIN_TE = 20, 5, 5

FENNIX_FAMS = [
    "DELTA_GEOM",
    "DELTA_ENV",
    "CONSTANT",
    "INTERFACE",
    "FULL_FAB_NORMALIZED",
    "COMBINED_PREDECLARED",
]

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
from run_stage1 import build_all_feature_tables, make_xy  # noqa: E402


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def numeric_frame(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    drop_cols = drop_cols or []
    X = df.set_index("id").copy() if "id" in df.columns else df.copy()
    X.index = X.index.astype(str)
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    return X


def load_y(target: str) -> pd.Series:
    name = TMAPP_REF if target == "TmApp" else HIC_REF
    s = pd.read_csv(OOF_DIR / f"{name}.csv").set_index("id")["y_true"].astype(float)
    s.index = s.index.astype(str)
    return s


def load_bases():
    dev = pd.read_csv(DEV)
    ids = [str(x) for x in dev["id"]]
    emb = np.load(EMB, allow_pickle=True)
    emb_ids = [str(x) for x in emb["ids"]]
    assert emb_ids == ids, "stage2 embedding order must match DEV"

    ablang = pd.DataFrame(np.asarray(emb["ablang2__HL_paired"], float), index=ids)
    ablang.columns = [f"ablang2_{i}" for i in range(ablang.shape[1])]
    esm2h = pd.DataFrame(np.asarray(emb["esm2__H"], float), index=ids)
    esm2h.columns = [f"esm2h_{i}" for i in range(esm2h.shape[1])]

    ann = pd.read_csv(ANN)
    regions = pd.read_csv(REGIONS)
    tables = build_all_feature_tables(dev, ann, regions)
    seq_b, _ = make_xy(tables, "SEQ_BASIC")
    seq_a, _ = make_xy(tables, "SEQ_ALL")
    seq_b.index = ids
    seq_a.index = ids
    seq_b.columns = [f"seqB_{c}" for c in seq_b.columns]
    seq_a.columns = [f"seqA_{c}" for c in seq_a.columns]

    tmapp_base = pd.concat([ablang, seq_b], axis=1)
    hic_base = pd.concat([esm2h, seq_a], axis=1)

    aro = numeric_frame(pd.read_csv(ARO)) if ARO.exists() else pd.DataFrame()
    aro = aro.reindex(ids)
    aro.columns = [f"aro_{c}" for c in aro.columns]
    hic_base_aro = pd.concat([hic_base, aro], axis=1)

    meta = {
        "TmApp_BASE": {
            "name": "ABLANG2_HL_PAIRED+SEQ_BASIC",
            "dim": int(tmapp_base.shape[1]),
            "paths": [
                str(EMB) + "::ablang2__HL_paired",
                "stage1 make_xy SEQ_BASIC",
            ],
        },
        "HIC_BASE": {
            "name": "ESM2_H+SEQ_ALL",
            "dim": int(hic_base.shape[1]),
            "paths": [str(EMB) + "::esm2__H", "stage1 make_xy SEQ_ALL"],
        },
        "HIC_BASE_ARO": {
            "name": "ESM2_H+SEQ_ALL+AROMATIC_TOPO",
            "dim": int(hic_base_aro.shape[1]),
            "paths": [
                str(EMB) + "::esm2__H",
                "stage1 make_xy SEQ_ALL",
                str(ARO),
            ],
        },
    }
    return tmapp_base, hic_base, hic_base_aro, meta


def load_structure_families():
    pack = numeric_frame(pd.read_csv(GAP / "results/TMAPP_PACKING_CAVITY_FEATURES.csv"))
    unsat = numeric_frame(pd.read_csv(GAP / "results/TMAPP_BURIED_UNSAT_FEATURES.csv"))
    iface = numeric_frame(
        pd.read_csv(GAP / "results/TMAPP_INTERFACE_FEATURES.csv"),
        drop_cols=["shape_complementarity_status"],
    )
    core = pd.concat([pack, unsat], axis=1, join="inner")
    core = core.loc[:, ~core.columns.duplicated()]
    gap_all = pd.concat([core, iface], axis=1, join="inner")
    gap_all = gap_all.loc[:, ~gap_all.columns.duplicated()]

    surf = pd.read_csv(GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv")
    scols = [c for c in surf.columns if c.startswith("fv_esmfold__")]
    cont = numeric_frame(surf[["id"] + scols])
    aro = numeric_frame(pd.read_csv(ARO)) if ARO.exists() else pd.DataFrame()
    hic_all = pd.concat([cont, aro], axis=1, join="inner") if len(aro) else cont
    hic_all = hic_all.loc[:, ~hic_all.columns.duplicated()]

    fams = [
        ("GapClosure", "TmApp", "PACKING_CAVITY", pack, "TmApp_BASE"),
        ("GapClosure", "TmApp", "BURIED_UNSAT", unsat, "TmApp_BASE"),
        ("GapClosure", "TmApp", "FAB_INTERFACE", iface, "TmApp_BASE"),
        ("GapClosure", "TmApp", "CORE_DEFECT", core, "TmApp_BASE"),
        ("GapClosure", "TmApp", "GAP_ALL", gap_all, "TmApp_BASE"),
        ("GapClosure", "HIC", "CONTINUOUS_SURFACE", cont, "HIC_BASE_ARO"),
        # Aggregate package vs PLM+seq only (aromatic lives in STRUCTURE, not BASE)
        ("GapClosure", "HIC", "HIC_SURFACE_ALL", hic_all, "HIC_BASE"),
    ]
    complete = pd.read_csv(OUT / "INTERIM_FENNIX_COMPLETE_SET.csv")
    complete["id"] = complete.id.astype(str)
    usable = set(
        complete.loc[complete.usable_interim.astype(bool) & complete.is_dev.astype(bool), "id"]
    )
    for name in FENNIX_FAMS:
        path = RES / f"INTERIM_{name}_FEATURES.csv"
        X = numeric_frame(pd.read_csv(path))
        cols = [c for c in X.columns if not str(c).endswith("_n") and not str(c).endswith("_n_sites")]
        X = X[cols]
        fams.append(("InterimFeNNix", "TmApp", name, X, "TmApp_BASE", usable))
    # normalize tuple length
    out = []
    for row in fams:
        if len(row) == 5:
            out.append((*row, None))
        else:
            out.append(row)
    return out


def rotation_splits(folds: pd.DataFrame, ids: list[str]):
    """Yield (k, train_ids, val_ids, test_ids) for TEST=k, VAL=(k+1)%5."""
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in ids if i in fmap]
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        yield k, tr, va, te


def fit_impute_median(X_tr: pd.DataFrame):
    med = X_tr.median(numeric_only=True)
    return med.fillna(0.0)


def apply_impute(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    return X.fillna(med).fillna(0.0)


def select_alpha_val(X_tr, y_tr, X_va, y_va):
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        sc = StandardScaler()
        Xt = sc.fit_transform(X_tr)
        Xv = sc.transform(X_va)
        m = Ridge(alpha=a, random_state=0)
        m.fit(Xt, y_tr)
        score = mae(y_va, m.predict(Xv))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def predict_tvt(X: pd.DataFrame, y: pd.Series, tr, va, te):
    """Train-only impute/scale/alpha; refit on train+val; predict test."""
    Xtr0, Xva0, Xte0 = X.loc[tr], X.loc[va], X.loc[te]
    ytr, yva, yte = y.loc[tr], y.loc[va], y.loc[te]
    med = fit_impute_median(Xtr0)
    Xtr = apply_impute(Xtr0, med)
    Xva = apply_impute(Xva0, med)
    a = select_alpha_val(Xtr, ytr, Xva, yva)
    # refit on train+val
    Xtv0 = X.loc[tr + va]
    ytv = y.loc[tr + va]
    med2 = fit_impute_median(Xtv0)
    Xtv = apply_impute(Xtv0, med2)
    Xte = apply_impute(Xte0, med2)
    sc = StandardScaler()
    Ztv = sc.fit_transform(Xtv)
    Zte = sc.transform(Xte)
    m = Ridge(alpha=a, random_state=0)
    m.fit(Ztv, ytv)
    pred = pd.Series(m.predict(Zte), index=te)
    return pred, a, len(tr), len(va), len(te)


def run_family(base: pd.DataFrame, struct: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, ids: list[str]):
    common = sorted(set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    coverage = []
    for k, tr, va, te in rotation_splits(folds, common):
        coverage.append(
            {
                "test_fold": k,
                "n_train": len(tr),
                "n_val": len(va),
                "n_test": len(te),
                "ok": len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE,
            }
        )
    if not all(c["ok"] for c in coverage):
        return None, coverage

    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    fold_rows = []
    alphas_b, alphas_p = [], []
    Xp = pd.concat([base.loc[common], struct.loc[common]], axis=1)
    Xp = Xp.loc[:, ~Xp.columns.duplicated()]
    Xb = base.loc[common]
    yy = y.loc[common]

    for k, tr, va, te in rotation_splits(folds, common):
        pb, ab, ntr, nva, nte = predict_tvt(Xb, yy, tr, va, te)
        pp, ap, _, _, _ = predict_tvt(Xp, yy, tr, va, te)
        base_oof.loc[te] = pb
        plus_oof.loc[te] = pp
        alphas_b.append(ab)
        alphas_p.append(ap)
        d_fold = mae(yy.loc[te], pb) - mae(yy.loc[te], pp)
        fold_rows.append(
            {
                "test_fold": k,
                "n_train": ntr,
                "n_val": nva,
                "n_test": nte,
                "base_mae": mae(yy.loc[te], pb),
                "plus_mae": mae(yy.loc[te], pp),
                "delta_mae": d_fold,
                "alpha_base": ab,
                "alpha_plus": ap,
            }
        )

    base_mae = mae(yy, base_oof)
    plus_mae = mae(yy, plus_oof)
    delta = base_mae - plus_mae
    return {
        "N": len(common),
        "base_dim": int(Xb.shape[1]),
        "struct_dim": int(struct.loc[common].shape[1]),
        "plus_dim": int(Xp.shape[1]),
        "base_mae": base_mae,
        "plus_mae": plus_mae,
        "delta_mae": delta,
        "base_pred": base_oof,
        "plus_pred": plus_oof,
        "y": yy,
        "fold_rows": fold_rows,
        "alphas_base": alphas_b,
        "alphas_plus": alphas_p,
    }, coverage


def boot_delta(y, base_pred, plus_pred, B=B_BOOT):
    idx = y.index
    y, b, p = y.loc[idx], base_pred.loc[idx], plus_pred.loc[idx]
    # delta = |y-b| mean - |y-p| mean  == base_mae - plus_mae; positive = improve
    d0 = mae(y, b) - mae(y, p)
    boots = []
    n = len(y)
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        boots.append(mae(y.iloc[ix], b.iloc[ix]) - mae(y.iloc[ix], p.iloc[ix]))
    boots = np.asarray(boots)
    return {
        "delta": float(d0),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "p_improve": float(np.mean(boots > 0)),
    }


def classify_simple(dp, ds):
    if dp > 0 and ds > 0:
        return "SIMPLE_CV_CONSISTENT_IMPROVEMENT"
    if (dp > 0) != (ds > 0):
        return "SIMPLE_CV_MIXED"
    return "SIMPLE_CV_NO_IMPROVEMENT"


def interpret(sp, ss, qp, qs):
    def pos(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and x > 0

    def neg(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and x <= 0

    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (sp, ss, qp, qs)):
        # partial
        if pos(qp) and pos(qs) and (sp is None or np.isnan(sp)):
            return "SIMPLE_ONLY_POSITIVE"
        if neg(qp) and neg(qs) and (sp is None or np.isnan(sp)):
            return "SIMPLE_ONLY_NEGATIVE"
        return "MIXED"
    both_pos = pos(sp) and pos(ss) and pos(qp) and pos(qs)
    both_neg = neg(sp) and neg(ss) and neg(qp) and neg(qs)
    strict_neg_simple_pos = neg(sp) and neg(ss) and pos(qp) and pos(qs)
    strict_pos_simple_neg = pos(sp) and pos(ss) and neg(qp) and neg(qs)
    if both_pos:
        return "BOTH_POSITIVE"
    if both_neg:
        return "BOTH_NEGATIVE"
    if strict_neg_simple_pos:
        return "STRICT_NEGATIVE_SIMPLE_POSITIVE"
    if strict_pos_simple_neg:
        return "STRICT_POSITIVE_SIMPLE_NEGATIVE"
    return "MIXED"


def load_strict_map():
    """Map family -> (primary_delta, shadow_delta) from residual_plus_oof preferably."""
    if not STRICT_CSV.exists():
        return {}
    df = pd.read_csv(STRICT_CSV)
    df = df[df.protocol == "STRICT_NESTED_INCREMENT"]
    out = {}
    for fam, g in df.groupby("family"):
        # prefer residual
        for fw in ("residual_plus_oof", "ridge_concat_oof_scalar"):
            sub = g[g.framework == fw]
            if sub.empty:
                continue
            p = sub[sub.cv == "Primary"]
            s = sub[sub.cv == "Shadow"]
            if p.empty or s.empty:
                continue
            out[fam] = {
                "strict_primary": float(p.iloc[0].delta_mae_inc_minus_comb),
                "strict_shadow": float(s.iloc[0].delta_mae_inc_minus_comb),
                "strict_framework": fw,
            }
            break
    return out


def write_report(results: pd.DataFrame, compare: pd.DataFrame, base_meta: dict):
    def ans_tmapp():
        g = results[(results.target == "TmApp") & (results.scope == "GapClosure")]
        hit = g[g.verdict == "SIMPLE_CV_CONSISTENT_IMPROVEMENT"]
        return (
            "はい — " + ", ".join(hit.family.unique())
            if len(hit)
            else "いいえ（Primary∧Shadow で一貫改善した Gap Closure TmApp family なし）"
        )

    def ans_hic():
        g = results[results.target == "HIC"]
        hit = g[g.verdict == "SIMPLE_CV_CONSISTENT_IMPROVEMENT"]
        return (
            "はい — " + ", ".join(hit.family.unique())
            if len(hit)
            else "いいえ"
        )

    def ans_gap_despite_strict():
        rows = compare[
            (compare.interpretation == "STRICT_NEGATIVE_SIMPLE_POSITIVE")
            & (compare.scope == "GapClosure")
        ]
        return ", ".join(rows.family.tolist()) if len(rows) else "なし"

    def ans_fennix():
        g = results[results.scope == "InterimFeNNix"]
        hit = g[g.verdict == "SIMPLE_CV_CONSISTENT_IMPROVEMENT"]
        return (
            "はい（暫定）— " + ", ".join(hit.family.unique())
            if len(hit)
            else "いいえ（暫定）"
        )

    consistent = (results.verdict == "SIMPLE_CV_CONSISTENT_IMPROVEMENT").any()
    mixed = (results.verdict == "SIMPLE_CV_MIXED").any()
    both_neg = (compare.interpretation == "BOTH_NEGATIVE").mean() if len(compare) else 0
    snsp = (compare.interpretation == "STRICT_NEGATIVE_SIMPLE_POSITIVE").sum()

    lines = [
        "# Simple TVT CV — 日本語報告",
        "",
        "**注意:** `STRICT_NESTED_INCREMENT` の代替ではない。直接特徴融合の直交チェック。FeNNix 本番未変更。",
        "",
        "## 冒頭 7 問",
        "",
        f"1. **直接特徴融合は TmApp を改善したか？**  \n   {ans_tmapp()}",
        "",
        f"2. **直接特徴融合は HIC を改善したか？**  \n   {ans_hic()}",
        "",
        f"3. **Strict nested で失敗しつつ Simple TVT で効いた Gap Closure は？**  \n   {ans_gap_despite_strict()}",
        "",
        f"4. **Interim FeNNix で Simple TVT 改善は？**  \n   {ans_fennix()}",
        "",
        f"5. **Primary / Shadow 一貫性は？**  \n   "
        + (
            f"一貫改善あり={consistent}; 混合あり={mixed}; 詳細は CSV。"
        ),
        "",
        f"6. **STRICT_NESTED と結論は一致するか？**  \n   "
        + (
            f"BOTH_NEGATIVE 比率≈{both_neg:.0%}; STRICT_NEGATIVE_SIMPLE_POSITIVE={int(snsp)}。"
            + (" 概ね一致（構造増分なし）。" if both_neg >= 0.7 and snsp == 0 else " 一部でプロトコル依存の可能性。")
        ),
        "",
        f"7. **プロトコル選択は科学的結論を実質変えるか？**  \n   "
        + (
            "変える（late-fusion 失敗だが直接融合は陽性）"
            if snsp > 0
            else "大きくは変えない（両プロトコルとも構造増分の強い証拠なし）。"
        ),
        "",
        "## BASE 定義",
        "",
        f"- TmApp: `{base_meta['TmApp_BASE']['name']}` dim={base_meta['TmApp_BASE']['dim']}",
        f"- HIC: `{base_meta['HIC_BASE_ARO']['name']}` dim={base_meta['HIC_BASE_ARO']['dim']}（構造比較の BASE）",
        "",
        "## 要約表",
        "",
        "| scope | family | Primary Δ | Shadow Δ | verdict |",
        "|-------|--------|-----------|----------|---------|",
    ]
    for _, r in results.sort_values(["scope", "target", "family"]).iterrows():
        lines.append(
            f"| {r.scope} | {r.family} | {r.primary_delta:+.4f} | {r.shadow_delta:+.4f} | {r.verdict} |"
        )
    lines += [
        "",
        "## vs Strict Nested",
        "",
        "詳細: `SIMPLE_VS_STRICT_COMPARISON.csv`。",
        "",
        "Interim FeNNix はすべて `PROVISIONAL_SIMPLE_CV`（完了サブセット）。",
        "",
    ]
    (OUT / "SIMPLE_TVT_CV_REPORT_JA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("Loading BASE matrices…", flush=True)
    tmapp_base, hic_base, hic_base_aro, base_meta = load_bases()
    bases = {"TmApp_BASE": tmapp_base, "HIC_BASE": hic_base, "HIC_BASE_ARO": hic_base_aro}
    print("BASE meta", base_meta, flush=True)

    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)
    y_tm = load_y("TmApp")
    y_hic = load_y("HIC")

    detail_rows = []
    summary = {}  # (scope,target,family) -> {Primary:..., Shadow:...}

    for scope, target, fam, Xstruct, base_key, id_filter in load_structure_families():
        base = bases[base_key]
        y = y_tm if target == "TmApp" else y_hic
        ids = list(base.index)
        if id_filter is not None:
            ids = [i for i in ids if i in id_filter]
        print(f"eval {scope} {target} {fam} base={base_key} N_cand={len(ids)} struct_dim={Xstruct.shape[1]}", flush=True)
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            out, cov = run_family(base, Xstruct, y, folds, ids)
            if out is None:
                print(f"  SKIP {tag} coverage={cov}", flush=True)
                detail_rows.append(
                    {
                        "scope": scope,
                        "target": target,
                        "family": fam,
                        "cv": tag,
                        "status": "UNDERPOWERED_OR_BAD_COVERAGE",
                        "coverage": str(cov),
                        "base_name": base_meta[base_key]["name"],
                        "label": "PROVISIONAL_SIMPLE_CV" if scope == "InterimFeNNix" else "SIMPLE_TVT_CV",
                    }
                )
                continue
            key = (scope, target, fam)
            summary.setdefault(key, {})[tag] = out
            for fr in out["fold_rows"]:
                detail_rows.append(
                    {
                        "scope": scope,
                        "target": target,
                        "family": fam,
                        "cv": tag,
                        "status": "OK",
                        "base_name": base_meta[base_key]["name"],
                        "N": out["N"],
                        "base_dim": out["base_dim"],
                        "struct_dim": out["struct_dim"],
                        "plus_dim": out["plus_dim"],
                        "base_mae": out["base_mae"],
                        "plus_mae": out["plus_mae"],
                        "delta_mae": out["delta_mae"],
                        "label": "PROVISIONAL_SIMPLE_CV" if scope == "InterimFeNNix" else "SIMPLE_TVT_CV",
                        **{f"fold_{k}": v for k, v in fr.items()},
                    }
                )
            # also one aggregate row marker via fold_test_fold = -1
            detail_rows.append(
                {
                    "scope": scope,
                    "target": target,
                    "family": fam,
                    "cv": tag,
                    "status": "OK_AGG",
                    "base_name": base_meta[base_key]["name"],
                    "N": out["N"],
                    "base_dim": out["base_dim"],
                    "struct_dim": out["struct_dim"],
                    "plus_dim": out["plus_dim"],
                    "base_mae": out["base_mae"],
                    "plus_mae": out["plus_mae"],
                    "delta_mae": out["delta_mae"],
                    "alphas_base": str(out["alphas_base"]),
                    "alphas_plus": str(out["alphas_plus"]),
                    "label": "PROVISIONAL_SIMPLE_CV" if scope == "InterimFeNNix" else "SIMPLE_TVT_CV",
                    "fold_test_fold": -1,
                }
            )

    # build results table (one row per family)
    res_rows = []
    for (scope, target, fam), by_cv in summary.items():
        if "Primary" not in by_cv or "Shadow" not in by_cv:
            continue
        p, s = by_cv["Primary"], by_cv["Shadow"]
        verdict = classify_simple(p["delta_mae"], s["delta_mae"])
        boot = {"ci_lo": np.nan, "ci_hi": np.nan, "p_improve": np.nan}
        if p["delta_mae"] > 0 and s["delta_mae"] > 0:
            # bootstrap on Primary held-out (and report Primary boot; Shadow optional)
            boot = boot_delta(p["y"], p["base_pred"], p["plus_pred"])
            boot_s = boot_delta(s["y"], s["base_pred"], s["plus_pred"])
        else:
            boot_s = boot
        res_rows.append(
            {
                "scope": scope,
                "target": target,
                "family": fam,
                "N_primary": p["N"],
                "N_shadow": s["N"],
                "base_dim": p["base_dim"],
                "struct_dim": p["struct_dim"],
                "primary_base_mae": p["base_mae"],
                "primary_plus_mae": p["plus_mae"],
                "primary_delta": p["delta_mae"],
                "shadow_base_mae": s["base_mae"],
                "shadow_plus_mae": s["plus_mae"],
                "shadow_delta": s["delta_mae"],
                "verdict": verdict,
                "boot_primary_ci_lo": boot.get("ci_lo", np.nan),
                "boot_primary_ci_hi": boot.get("ci_hi", np.nan),
                "boot_primary_p_improve": boot.get("p_improve", np.nan),
                "boot_shadow_ci_lo": boot_s.get("ci_lo", np.nan) if p["delta_mae"] > 0 and s["delta_mae"] > 0 else np.nan,
                "boot_shadow_ci_hi": boot_s.get("ci_hi", np.nan) if p["delta_mae"] > 0 and s["delta_mae"] > 0 else np.nan,
                "boot_shadow_p_improve": boot_s.get("p_improve", np.nan) if p["delta_mae"] > 0 and s["delta_mae"] > 0 else np.nan,
                "label": "PROVISIONAL_SIMPLE_CV" if scope == "InterimFeNNix" else "SIMPLE_TVT_CV",
                "per_fold_primary_delta": str([fr["delta_mae"] for fr in p["fold_rows"]]),
                "per_fold_shadow_delta": str([fr["delta_mae"] for fr in s["fold_rows"]]),
                "coverage_primary": str([(fr["n_train"], fr["n_val"], fr["n_test"]) for fr in p["fold_rows"]]),
                "coverage_shadow": str([(fr["n_train"], fr["n_val"], fr["n_test"]) for fr in s["fold_rows"]]),
            }
        )

    results = pd.DataFrame(res_rows)
    results_path = OUT / "SIMPLE_TVT_CV_RESULTS.csv"
    results.to_csv(results_path, index=False)
    pd.DataFrame(detail_rows).to_csv(OUT / "SIMPLE_TVT_CV_FOLD_DETAIL.csv", index=False)

    strict_map = load_strict_map()
    cmp_rows = []
    for _, r in results.iterrows():
        sm = strict_map.get(r.family, {})
        sp = sm.get("strict_primary", np.nan)
        ss = sm.get("strict_shadow", np.nan)
        cmp_rows.append(
            {
                "scope": r.scope,
                "target": r.target,
                "family": r.family,
                "strict_framework": sm.get("strict_framework", ""),
                "strict_nested_primary_delta": sp,
                "strict_nested_shadow_delta": ss,
                "simple_tvt_primary_delta": r.primary_delta,
                "simple_tvt_shadow_delta": r.shadow_delta,
                "simple_verdict": r.verdict,
                "interpretation": interpret(sp, ss, r.primary_delta, r.shadow_delta),
                "label": r.label,
            }
        )
    compare = pd.DataFrame(cmp_rows)
    compare.to_csv(OUT / "SIMPLE_VS_STRICT_COMPARISON.csv", index=False)
    write_report(results, compare, base_meta)
    print("wrote", results_path, "n_families", len(results), flush=True)
    print(results[["scope", "family", "primary_delta", "shadow_delta", "verdict"]].to_string(index=False), flush=True)
    print(compare[["family", "interpretation"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
