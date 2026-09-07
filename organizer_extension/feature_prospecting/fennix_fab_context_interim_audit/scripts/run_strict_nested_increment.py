#!/usr/bin/env python3
"""STRICT_NESTED_INCREMENT vs OLD_GLOBAL_OOF_INCREMENT (evaluation-only).

Does not touch FeNNix production caches/workers.
Reuses Stage5 frozen base configs + precomputed feature matrices / PLM embeddings.
"""
from __future__ import annotations

import json
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
ARO_CSV = GAP / "cache/aromatic_features_esmfold.csv"
BASE_CFG = ROOT / "virtual_participant/stage5_integration/stage5_base_models.json"
DEV = ROOT / "competition/data/distribution/dev.csv"

TMAPP_REF = "TmApp__META_performance__ridge_100.0"
HIC_REF = "HIC__SIMPLE_blend_seq_surf_adv"
TMAPP_BASES = [
    "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt",
    "TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt",
    "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt",
    "TmApp__ADV_TMAPP_ALL__SVROpt",
]
HIC_BASES = [
    "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt",
    "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt",
    "HIC__ADV_SURFACE_PATCH__SVROpt",
]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
B_BOOT = 10000
RNG = np.random.default_rng(42)

sys.path.insert(0, str(ROOT / "virtual_participant/stage5_integration/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage4_advanced_structure/scripts"))

from run_stage5 import (  # noqa: E402
    fit_predict,
    load_feature_store,
)

FENNIX_FAMS = [
    "DELTA_GEOM",
    "DELTA_ENV",
    "CONSTANT",
    "INTERFACE",
    "FULL_FAB_NORMALIZED",
    "COMBINED_PREDECLARED",
]


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.set_index("id").copy() if "id" in df.columns else df.copy()
    X.index = X.index.astype(str)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    keep = [c for c in X.columns if float(X[c].std(ddof=0)) > 1e-12]
    return X[keep]


def select_alpha(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        maes = []
        for f in sorted({fold_map[i] for i in train_ids}):
            te = [i for i in train_ids if fold_map[i] == f]
            tr = [i for i in train_ids if fold_map[i] != f]
            if len(te) == 0 or len(tr) < 2:
                continue
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.loc[tr]), y.loc[tr])
            maes.append(mae(y.loc[te], m.predict(sc.transform(X.loc[te]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def nested_ridge_global_ref(X, y, folds, ref):
    """OLD path: nest Ridge with globally fixed ref column / residual base."""
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof_concat = pd.Series(index=ids, dtype=float)
    oof_resid = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        # concat
        Xc = X.copy()
        Xc["__ref__"] = ref.loc[ids].astype(float)
        a = select_alpha(Xc, y, folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xc.loc[tr]), y.loc[tr])
        oof_concat.loc[te] = m.predict(sc.transform(Xc.loc[te]))
        # residual
        resid = y.loc[tr] - ref.loc[tr]
        a2 = select_alpha(X, resid, folds, tr)
        sc2 = StandardScaler()
        m2 = Ridge(alpha=a2, random_state=0)
        m2.fit(sc2.fit_transform(X.loc[tr]), resid)
        oof_resid.loc[te] = ref.loc[te].values + m2.predict(sc2.transform(X.loc[te]))
    return oof_concat, oof_resid


def boot_delta(y, base, cand, B=B_BOOT):
    idx = y.index.intersection(base.index).intersection(cand.index)
    y, base, cand = y.loc[idx], base.loc[idx], cand.loc[idx]
    d0 = mae(y, base) - mae(y, cand)  # >0 improvement
    n = len(y)
    boots = []
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        boots.append(mae(y.iloc[ix], base.iloc[ix]) - mae(y.iloc[ix], cand.iloc[ix]))
    boots = np.asarray(boots)
    return float(d0), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(np.mean(boots > 0))


def direction_label(d_pri, d_sha):
    if d_pri > 0 and d_sha > 0:
        return "CONSISTENT_IMPROVEMENT"
    if d_pri < 0 and d_sha < 0:
        return "CONSISTENT_WORSENING"
    return "MIXED_DIRECTION"


def verdict_from_delta(d, lo, hi):
    if d <= 0:
        return "NO_INCREMENT"
    if hi > 0 and lo <= 0:
        return "WEAK_INCREMENT"
    if lo > 0:
        return "CLEAR_INCREMENT"
    return "WEAK_INCREMENT"


class IncumbentEngine:
    """Reconstruct TmApp META ridge_100 / HIC equal-blend from Stage5 feature store."""

    def __init__(self):
        print("Loading Stage5 feature store (precomputed matrices / embeddings)…", flush=True)
        self.dev = pd.read_csv(DEV)
        self.ids = [str(x) for x in self.dev["id"]]
        self.id_to_pos = {i: k for k, i in enumerate(self.ids)}
        self.y_tm = self.dev["TmApp"].to_numpy(float)
        self.y_hic = self.dev["HIC"].to_numpy(float)
        with open(BASE_CFG) as f:
            specs = json.load(f)
        self.cfg_by_id = {c["experiment_id"]: c for c in specs["TmApp"] + specs["HIC"]}
        self.store = load_feature_store(self.ids)
        self._bundle_cache: dict = {}
        print("Feature store ready.", flush=True)

    def _y(self, target: str):
        return self.y_tm if target == "TmApp" else self.y_hic

    def _cfgs(self, target: str):
        eids = TMAPP_BASES if target == "TmApp" else HIC_BASES
        return [self.cfg_by_id[e] for e in eids]

    def _pos(self, id_list):
        return np.asarray([self.id_to_pos[str(i)] for i in id_list], dtype=int)

    def base_predict(self, cfg, y, tr_ids, te_ids):
        tr_idx = self._pos(tr_ids)
        te_idx = self._pos(te_ids)
        return fit_predict(cfg, self.store, y, tr_idx, te_idx)

    def fold_bundle(self, target: str, folds: pd.DataFrame, ids: list[str]):
        """Per outer fold: ref_tr (inner), ref_te (train→val), and assembled OOF."""
        ids = [str(i) for i in ids]
        folds = folds[folds.id.astype(str).isin(ids)].copy()
        folds["id"] = folds.id.astype(str)
        # cache key: target + fold assignment fingerprint
        fmap = folds.set_index("id")["fold"].astype(int).to_dict()
        key = (target, tuple(sorted((i, int(fmap[i])) for i in ids)))
        if key in self._bundle_cache:
            return self._bundle_cache[key]
        fold_map = fmap
        y = self._y(target)
        cfgs = self._cfgs(target)
        bundles = {}
        oof = pd.Series(index=ids, dtype=float)
        for f in sorted({fold_map[i] for i in ids}):
            te = [i for i in ids if fold_map[i] == f]
            tr = [i for i in ids if fold_map[i] != f]
            # inner base OOF on tr
            mat_tr = pd.DataFrame(index=tr, dtype=float)
            for c in cfgs:
                s = pd.Series(index=tr, dtype=float)
                for fi in sorted({fold_map[i] for i in tr}):
                    te_i = [i for i in tr if fold_map[i] == fi]
                    tr_i = [i for i in tr if fold_map[i] != fi]
                    if len(tr_i) < 2:
                        continue
                    s.loc[te_i] = self.base_predict(c, y, tr_i, te_i)
                mat_tr[c["experiment_id"]] = s
            # val bases: fit on full tr
            mat_te = pd.DataFrame(index=te, dtype=float)
            for c in cfgs:
                mat_te[c["experiment_id"]] = self.base_predict(c, y, tr, te)
            if target == "HIC":
                ref_tr = mat_tr.mean(axis=1)
                ref_te = mat_te.mean(axis=1)
            else:
                # train-side meta via LOF on mat_tr
                ref_tr = pd.Series(index=tr, dtype=float)
                for fi in sorted({fold_map[i] for i in tr}):
                    te_i = [i for i in tr if fold_map[i] == fi]
                    tr_i = [i for i in tr if fold_map[i] != fi]
                    if len(tr_i) < 2:
                        continue
                    model = Ridge(alpha=100.0, random_state=0)
                    model.fit(mat_tr.loc[tr_i].values, y[self._pos(tr_i)])
                    ref_tr.loc[te_i] = model.predict(mat_tr.loc[te_i].values)
                model = Ridge(alpha=100.0, random_state=0)
                model.fit(mat_tr.values, y[self._pos(tr)])
                ref_te = pd.Series(model.predict(mat_te.values), index=te)
            bundles[int(f)] = {"tr": tr, "te": te, "ref_tr": ref_tr, "ref_te": ref_te}
            oof.loc[te] = ref_te
        self._bundle_cache[key] = (bundles, oof)
        return bundles, oof


def strict_structural(X, y, folds, bundles, framework: str):
    ids = list(X.index)
    cand = pd.Series(index=ids, dtype=float)
    for f, b in bundles.items():
        tr, te = b["tr"], b["te"]
        ref_tr, ref_te = b["ref_tr"], b["ref_te"]
        if framework == "ridge_concat_oof_scalar":
            Xc_tr = X.loc[tr].copy()
            Xc_tr["__ref__"] = ref_tr.loc[tr].astype(float)
            Xc_te = X.loc[te].copy()
            Xc_te["__ref__"] = ref_te.loc[te].astype(float)
            a = select_alpha(Xc_tr, y.loc[tr], folds, tr)
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(Xc_tr), y.loc[tr])
            cand.loc[te] = m.predict(sc.transform(Xc_te))
        else:  # residual_plus_oof
            resid = y.loc[tr] - ref_tr.loc[tr]
            a = select_alpha(X.loc[tr], resid, folds, tr)
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.loc[tr]), resid)
            cand.loc[te] = ref_te.loc[te].values + m.predict(sc.transform(X.loc[te]))
    return cand


def load_gap_families():
    pack = prep_X(pd.read_csv(GAP / "results/TMAPP_PACKING_CAVITY_FEATURES.csv"))
    unsat = prep_X(pd.read_csv(GAP / "results/TMAPP_BURIED_UNSAT_FEATURES.csv"))
    iface = prep_X(
        pd.read_csv(GAP / "results/TMAPP_INTERFACE_FEATURES.csv").drop(
            columns=["shape_complementarity_status"], errors="ignore"
        )
    )
    core = pd.concat([pack, unsat], axis=1, join="inner")
    core = core.loc[:, ~core.columns.duplicated()]
    gap_all = pd.concat([core, iface], axis=1, join="inner")
    gap_all = gap_all.loc[:, ~gap_all.columns.duplicated()]
    surf = pd.read_csv(GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv")
    scols = [c for c in surf.columns if c.startswith("fv_esmfold__")]
    hic_surf = prep_X(surf[["id"] + scols])
    aro = prep_X(pd.read_csv(ARO_CSV)) if ARO_CSV.exists() else pd.DataFrame()
    hic_all = pd.concat([hic_surf, aro], axis=1, join="inner") if len(aro) else hic_surf
    hic_all = hic_all.loc[:, ~hic_all.columns.duplicated()]
    y_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)
    y_tm.index = y_tm.index.astype(str)
    inc_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_pred"].astype(float)
    inc_tm.index = inc_tm.index.astype(str)
    y_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    y_hic.index = y_hic.index.astype(str)
    inc_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_pred"].astype(float)
    inc_hic.index = inc_hic.index.astype(str)
    return [
        ("GapClosure", "TmApp", "CORE_DEFECT", core, y_tm, inc_tm),
        ("GapClosure", "TmApp", "FAB_INTERFACE", iface, y_tm, inc_tm),
        ("GapClosure", "TmApp", "GAP_ALL", gap_all, y_tm, inc_tm),
        ("GapClosure", "HIC", "HIC_SURFACE_ALL", hic_all, y_hic, inc_hic),
    ]


def load_fennix_families():
    y = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")
    y.index = y.index.astype(str)
    y_true = y["y_true"].astype(float)
    inc = y["y_pred"].astype(float)
    complete = pd.read_csv(OUT / "INTERIM_FENNIX_COMPLETE_SET.csv")
    complete["id"] = complete.id.astype(str)
    usable = set(complete.loc[complete.usable_interim.astype(bool) & complete.is_dev.astype(bool), "id"])
    fams = []
    for name in FENNIX_FAMS:
        path = RES / f"INTERIM_{name}_FEATURES.csv"
        if not path.exists():
            print("missing", path, flush=True)
            continue
        X = prep_X(pd.read_csv(path))
        cols = [c for c in X.columns if not c.endswith("_n") and not c.endswith("_n_sites")][:20]
        X = X[cols]
        fams.append(("InterimFeNNix", "TmApp", name, X, y_true, inc, usable))
    return fams


def eval_family(engine, scope, target, name, X, y, global_inc, folds, tag, id_filter=None):
    ids = sorted(set(X.index.astype(str)) & set(y.index.astype(str)) & set(folds.id.astype(str)))
    if id_filter is not None:
        ids = [i for i in ids if i in id_filter]
    if len(ids) < 30:
        return []
    X = X.loc[ids]
    y = y.loc[ids]
    ginc = global_inc.loc[ids]
    ff = folds[folds.id.isin(ids)].copy()
    ff["id"] = ff.id.astype(str)

    rows = []
    # OLD global OOF
    old_c, old_r = nested_ridge_global_ref(X, y, ff, ginc)
    for framework, cand in [
        ("ridge_concat_oof_scalar", old_c),
        ("residual_plus_oof", old_r),
    ]:
        d, lo, hi, p = boot_delta(y, ginc, cand)
        rows.append(
            {
                "scope": scope,
                "target": target,
                "family": name,
                "framework": framework,
                "protocol": "OLD_GLOBAL_OOF_INCREMENT",
                "cv": tag,
                "N": len(ids),
                "feature_dim": int(X.shape[1]),
                "incumbent_mae": mae(y, ginc),
                "combined_mae": mae(y, cand),
                "delta_mae_inc_minus_comb": d,
                "boot_ci_low": lo,
                "boot_ci_high": hi,
                "p_improve": p,
                "verdict": verdict_from_delta(d, lo, hi),
            }
        )

    # STRICT nested
    print(f"  STRICT {scope} {target} {name} {tag} N={len(ids)}…", flush=True)
    bundles, strict_inc = engine.fold_bundle(target, ff, ids)
    for framework in ("ridge_concat_oof_scalar", "residual_plus_oof"):
        cand = strict_structural(X, y, ff, bundles, framework)
        d, lo, hi, p = boot_delta(y, strict_inc, cand)
        rows.append(
            {
                "scope": scope,
                "target": target,
                "family": name,
                "framework": framework,
                "protocol": "STRICT_NESTED_INCREMENT",
                "cv": tag,
                "N": len(ids),
                "feature_dim": int(X.shape[1]),
                "incumbent_mae": mae(y, strict_inc),
                "combined_mae": mae(y, cand),
                "delta_mae_inc_minus_comb": d,
                "boot_ci_low": lo,
                "boot_ci_high": hi,
                "p_improve": p,
                "verdict": verdict_from_delta(d, lo, hi),
            }
        )
    return rows


def write_report(df: pd.DataFrame):
    lines = [
        "# Strict Nested Increment — 結果報告",
        "",
        "**汚染判定:** `CROSS_FOLD_CONTAMINATION = YES`（詳細: `STRICT_NESTED_CONTAMINATION_AUDIT.md`）。",
        "",
        "本表は同一 family / framework / CV で `OLD_GLOBAL_OOF_INCREMENT` と `STRICT_NESTED_INCREMENT` を併記する。",
        "ΔMAE = incumbent_MAE − combined_MAE（**正 = 改善**）。",
        "",
        "## Primary / Shadow 方向ラベル",
        "",
        "- `CONSISTENT_IMPROVEMENT`: Primary・Shadow とも ΔMAE > 0",
        "- `CONSISTENT_WORSENING`: Primary・Shadow とも ΔMAE < 0",
        "- `MIXED_DIRECTION`: 符号が分かれる",
        "",
        "**注意:** 悪化の一致を「良いシグナル」や「同方向改善」と読んではならない。",
        "",
    ]
    # pivot summary
    lines.append("## 要約表（framework 平均は取らない — 両 framework を並記）")
    lines.append("")
    for scope in ["GapClosure", "InterimFeNNix"]:
        sub = df[df.scope == scope]
        if sub.empty:
            continue
        lines.append(f"### {scope}")
        lines.append("")
        for (fam, fw), g in sub.groupby(["family", "framework"]):
            lines.append(f"**{fam} / {fw}**")
            for proto in ["OLD_GLOBAL_OOF_INCREMENT", "STRICT_NESTED_INCREMENT"]:
                p = g[(g.protocol == proto) & (g.cv == "Primary")]
                s = g[(g.protocol == proto) & (g.cv == "Shadow")]
                if p.empty or s.empty:
                    continue
                dp = float(p.iloc[0].delta_mae_inc_minus_comb)
                ds = float(s.iloc[0].delta_mae_inc_minus_comb)
                lab = direction_label(dp, ds)
                lines.append(
                    f"- `{proto}`: Primary Δ={dp:+.4f} ({p.iloc[0].verdict}), "
                    f"Shadow Δ={ds:+.4f} ({s.iloc[0].verdict}) → **{lab}**"
                )
            # difference note
            op = g[(g.protocol == "OLD_GLOBAL_OOF_INCREMENT") & (g.cv == "Primary")]
            sp = g[(g.protocol == "STRICT_NESTED_INCREMENT") & (g.cv == "Primary")]
            if len(op) and len(sp):
                diff = float(sp.iloc[0].delta_mae_inc_minus_comb) - float(op.iloc[0].delta_mae_inc_minus_comb)
                lines.append(f"- Primary STRICT−OLD ΔΔMAE = {diff:+.4f}")
            lines.append("")
    lines.append("## 解釈")
    lines.append("")
    lines.append(
        "OLD は train 側 incumbent 特徴が outer-val を含む base 学習由来のため、"
        "STRICT と数値がずれうる。最終 `NO_INCREMENT` 主張は **STRICT_NESTED_INCREMENT** 側で確定する。"
    )
    lines.append("")
    lines.append("FeNNix 本番ワーカー／特徴定義は未変更（評価のみ）。")
    lines.append("")
    (OUT / "STRICT_NESTED_INCREMENT_REPORT_JA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_interim_report(df: pd.DataFrame):
    """Fix direction wording in INTERIM_FENNIX_REPORT_JA.md Q8."""
    path = OUT / "INTERIM_FENNIX_REPORT_JA.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    # Build direction lines from STRICT preferred, else OLD interim results
    fenn = df[df.scope == "InterimFeNNix"]
    bits = []
    for fam in FENNIX_FAMS:
        g = fenn[(fenn.family == fam) & (fenn.framework == "ridge_concat_oof_scalar")]
        # prefer STRICT
        p = g[(g.protocol == "STRICT_NESTED_INCREMENT") & (g.cv == "Primary")]
        s = g[(g.protocol == "STRICT_NESTED_INCREMENT") & (g.cv == "Shadow")]
        if p.empty or s.empty:
            continue
        lab = direction_label(
            float(p.iloc[0].delta_mae_inc_minus_comb),
            float(s.iloc[0].delta_mae_inc_minus_comb),
        )
        bits.append(f"{fam}:{lab}")
    new_q8 = (
        "8. **Primary/Shadow 方向一致は？**  \n"
        "   （ΔMAE>0=改善。悪化の一致は陽性シグナルではない。）  \n"
        f"   {', '.join(bits)}\n"
    )
    import re

    text2, n = re.subn(
        r"8\.\s*\*\*Primary/Shadow 方向一致は？\*\*.*?(?=\n10\.|\n9\.)",
        new_q8 + "\n",
        text,
        count=1,
        flags=re.S,
    )
    if n == 0:
        # fallback: replace the known short line
        text2 = text.replace(
            "8. **Primary/Shadow 方向一致は？**  \n"
            "   DELTA_GEOM:一致, DELTA_ENV:一致, CONSTANT:一致, INTERFACE:一致, "
            "FULL_FAB_NORMALIZED:一致, PREP_RELAX_SENSITIVITY:一致, COMBINED_PREDECLARED:一致",
            new_q8.rstrip(),
        )
    # also fix "同方向改善=False" phrasing that could be misread
    text2 = text2.replace("同方向改善=False", "方向ラベルは悪化側の一致であり改善ではない")
    path.write_text(text2, encoding="utf-8")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    engine = IncumbentEngine()
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    rows = []
    print("=== Gap Closure families ===", flush=True)
    for scope, target, name, X, y, inc in load_gap_families():
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            rows.extend(eval_family(engine, scope, target, name, X, y, inc, folds, tag))

    print("=== Interim FeNNix families ===", flush=True)
    for scope, target, name, X, y, inc, usable in load_fennix_families():
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            rows.extend(
                eval_family(engine, scope, target, name, X, y, inc, folds, tag, id_filter=usable)
            )

    df = pd.DataFrame(rows)
    out_csv = OUT / "STRICT_NESTED_INCREMENT_RESULTS.csv"
    df.to_csv(out_csv, index=False)
    write_report(df)
    patch_interim_report(df)
    # also amend GAP audit
    audit = OUT / "GAP_CLOSURE_INCREMENT_AUDIT.md"
    if audit.exists():
        extra = (
            "\n---\n\n## Strict nested follow-up (2026-09-07)\n\n"
            "Prior claim of leakage-safe incremental **overstated**: global incumbent OOF "
            "as a feature inside a second outer CV causes **cross-fold contamination** on "
            "outer-train samples. See `STRICT_NESTED_CONTAMINATION_AUDIT.md` and "
            "`STRICT_NESTED_INCREMENT_RESULTS.csv`. Final NO_INCREMENT must be read from "
            "`STRICT_NESTED_INCREMENT`.\n"
        )
        t = audit.read_text(encoding="utf-8")
        if "Strict nested follow-up" not in t:
            audit.write_text(t.rstrip() + "\n" + extra, encoding="utf-8")
    print("wrote", out_csv, "rows", len(df), flush=True)


if __name__ == "__main__":
    main()
