#!/usr/bin/env python3
"""Predeclared OpenMM MD family ablation — Simple TVT only. No new MD / no fishing."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "openmm_fab_md"
RES = OUT / "results"
INTERIM = FP / "fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"
B_BOOT = 10000
RNG = np.random.default_rng(42)

spec = importlib.util.spec_from_file_location("tvt_rescreen", INTERIM / "scripts/run_simple_tvt_rescreen.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)


def run_one(base, struct, y, folds, ids, mode):
    common = sorted(set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    cov_ok = True
    for k, tr, va, te in mod.rotation_splits(folds, common):
        ok = len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        cov_ok = cov_ok and ok
    if not cov_ok or len(common) < 40:
        return None
    Xb, Xs = base.loc[common], struct.loc[common]
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    for k, tr, va, te in mod.rotation_splits(folds, common):
        a_base = mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=False)
        a_plus = (
            a_base
            if mode != "FREE_ALPHA"
            else mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=True)
        )
        if mode != "FREE_ALPHA":
            a_plus = a_base
        base_oof.loc[te] = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_base, use_struct=False)
        plus_oof.loc[te] = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_plus, use_struct=True)
    return {
        "N": len(common),
        "struct_dim": int(Xs.shape[1]),
        "base_mae": mod.mae(yy, base_oof),
        "plus_mae": mod.mae(yy, plus_oof),
        "delta": mod.mae(yy, base_oof) - mod.mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
    }


def family_verdict(dp, ds, dpf, dsf):
    if dp is None or ds is None:
        return "FAMILY_DROP"
    if dp > 0 and ds > 0:
        fixed_ok = (dpf is not None and dsf is not None and dpf >= 0 and dsf >= 0)
        if min(dp, ds) < 0.01 or not fixed_ok:
            return "FAMILY_WEAK" if fixed_ok or min(dp, ds) < 0.01 else "FAMILY_WEAK"
        return "FAMILY_KEEP"
    if dp > 0 or ds > 0:
        return "FAMILY_MIXED"
    return "FAMILY_DROP"


def bootstrap(y, b0, p0):
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(b0, float)) - np.abs(y - np.asarray(p0, float))
    boots = np.array([float(d[RNG.integers(0, len(d), len(d))].mean()) for _ in range(B_BOOT)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float((boots <= 0).mean())


def main():
    fmap = pd.read_csv(OUT / "OPENMM_MD_FAMILY_MAP.csv")
    feat = pd.read_parquet(RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet").set_index("id")
    # ensure map covers all numeric features
    feat_cols = [c for c in feat.columns if c != "id"]
    mapped = set(fmap.feature)
    assert mapped == set(feat_cols), (mapped - set(feat_cols), set(feat_cols) - mapped)

    bases, _ = mod.load_bases()
    y_tm, y_hic = mod.load_y("TmApp"), mod.load_y("HIC")
    primary, shadow = pd.read_csv(PRIMARY), pd.read_csv(SHADOW)
    # exact successful DEV intersection from authoritative table
    usable = sorted(feat.index.astype(str).tolist())
    assert len(usable) == 158, len(usable)

    bio = pd.read_csv(BIO).set_index("id")
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    X_pair = bio[pair_cols].apply(pd.to_numeric, errors="coerce")
    m1 = mod.numeric_X(pd.read_csv(M1), drop_cols=["status", "extraction_status"])
    recipe = pd.concat([bases["TmApp_BASE"], X_pair, m1], axis=1, join="inner")
    recipe = recipe.loc[:, ~recipe.columns.duplicated()]

    families = list(fmap.ablation_family.drop_duplicates())
    rows = []
    oofs = {}

    for fam in families:
        cols = fmap.loc[fmap.ablation_family == fam, "feature"].tolist()
        Xf = feat[cols].astype(float)
        specs = [
            ("TmApp", "BASE+F", bases["TmApp_BASE"], y_tm),
            ("TmApp", "BASE+BIOEMU_NEW_PAIRWISE+M1+F", recipe, y_tm),
            ("HIC", "HIC_ARO+F", bases["HIC_BASE_ARO"], y_hic),
        ]
        for target, tag, base, y in specs:
            for fold_name, folds in [("Primary", primary), ("Shadow", shadow)]:
                for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
                    out = run_one(base, Xf, y, folds, usable, mode)
                    if out is None:
                        rows.append(
                            {
                                "target": target,
                                "comparison": tag,
                                "family": fam,
                                "fold": fold_name,
                                "mode": mode,
                                "status": "SKIP",
                                "N": None,
                                "struct_dim": len(cols),
                                "base_mae": None,
                                "cand_mae": None,
                                "delta": None,
                            }
                        )
                        continue
                    rows.append(
                        {
                            "target": target,
                            "comparison": tag,
                            "family": fam,
                            "fold": fold_name,
                            "mode": mode,
                            "status": "OK",
                            "N": out["N"],
                            "struct_dim": out["struct_dim"],
                            "base_mae": out["base_mae"],
                            "cand_mae": out["plus_mae"],
                            "delta": out["delta"],
                        }
                    )
                    if mode == "FREE_ALPHA":
                        oofs[(fam, target, tag, fold_name)] = out

    df = pd.DataFrame(rows)
    df.to_csv(RES / "OPENMM_MD_FAMILY_ABLATION_RESULTS.csv", index=False)
    df.to_csv(OUT / "OPENMM_MD_FAMILY_ABLATION_RESULTS.csv", index=False)

    # verdicts per family/target
    def getd(fam, target, tag, fold, mode="FREE_ALPHA"):
        s = df[
            (df.family == fam)
            & (df.target == target)
            & (df.comparison == tag)
            & (df.fold == fold)
            & (df.mode == mode)
            & (df.status == "OK")
        ]
        if len(s) == 0:
            return None
        return float(list(s.itertuples())[0].delta)

    summary_rows = []
    boots = {}
    survivors = []
    for fam in families:
        tm_p = getd(fam, "TmApp", "BASE+F", "Primary")
        tm_s = getd(fam, "TmApp", "BASE+F", "Shadow")
        tm_pf = getd(fam, "TmApp", "BASE+F", "Primary", "BASE_FIXED_ALPHA")
        tm_sf = getd(fam, "TmApp", "BASE+F", "Shadow", "BASE_FIXED_ALPHA")
        rec_p = getd(fam, "TmApp", "BASE+BIOEMU_NEW_PAIRWISE+M1+F", "Primary")
        rec_s = getd(fam, "TmApp", "BASE+BIOEMU_NEW_PAIRWISE+M1+F", "Shadow")
        hic_p = getd(fam, "HIC", "HIC_ARO+F", "Primary")
        hic_s = getd(fam, "HIC", "HIC_ARO+F", "Shadow")
        hic_pf = getd(fam, "HIC", "HIC_ARO+F", "Primary", "BASE_FIXED_ALPHA")
        hic_sf = getd(fam, "HIC", "HIC_ARO+F", "Shadow", "BASE_FIXED_ALPHA")

        v_tm = family_verdict(tm_p, tm_s, tm_pf, tm_sf)
        v_rec = family_verdict(rec_p, rec_s, getd(fam, "TmApp", "BASE+BIOEMU_NEW_PAIRWISE+M1+F", "Primary", "BASE_FIXED_ALPHA"), getd(fam, "TmApp", "BASE+BIOEMU_NEW_PAIRWISE+M1+F", "Shadow", "BASE_FIXED_ALPHA"))
        v_hic = family_verdict(hic_p, hic_s, hic_pf, hic_sf)

        for tgt, vp, vs, tag in [
            ("TmApp_BASE", tm_p, tm_s, "BASE+F"),
            ("TmApp_RECIPE", rec_p, rec_s, "BASE+BIOEMU_NEW_PAIRWISE+M1+F"),
            ("HIC", hic_p, hic_s, "HIC_ARO+F"),
        ]:
            if vp is not None and vs is not None and vp > 0 and vs > 0:
                key = (fam, "TmApp" if tgt.startswith("Tm") else "HIC", tag, "Primary")
                # bootstrap both folds
                for fold in ("Primary", "Shadow"):
                    o = oofs.get((fam, "TmApp" if tgt.startswith("Tm") else "HIC", tag, fold))
                    if o is not None:
                        lo, hi, p = bootstrap(o["y"], o["base_oof"], o["plus_oof"])
                        boots[f"{fam}|{tgt}|{fold}"] = {"ci95": [lo, hi], "p_improve": p}
                survivors.append({"family": fam, "setting": tgt, "primary_delta": vp, "shadow_delta": vs})

        summary_rows.append(
            {
                "family": fam,
                "n_features": int((fmap.ablation_family == fam).sum()),
                "tm_base_P": tm_p,
                "tm_base_S": tm_s,
                "tm_base_fixed_P": tm_pf,
                "tm_base_fixed_S": tm_sf,
                "tm_base_verdict": v_tm,
                "tm_recipe_P": rec_p,
                "tm_recipe_S": rec_s,
                "tm_recipe_verdict": v_rec,
                "hic_P": hic_p,
                "hic_S": hic_s,
                "hic_fixed_P": hic_pf,
                "hic_fixed_S": hic_sf,
                "hic_verdict": v_hic,
            }
        )

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(RES / "OPENMM_MD_FAMILY_ABLATION_SUMMARY.csv", index=False)

    any_survive = len(survivors) > 0
    closure = "OPENMM_ONE_FAMILY_SURVIVES" if any_survive else "OPENMM_CLOSED"
    decision = {
        "families": families,
        "N": 158,
        "survivors": survivors,
        "bootstrap": boots,
        "final_openmm_verdict": closure,
        "GO_TO_TEST": False,  # organizer review required even if survivors
    }
    (RES / "OPENMM_MD_FAMILY_ABLATION_DECISION.json").write_text(json.dumps(decision, indent=2))

    # report JA
    def yn(cond):
        return "YES" if cond else "NO"

    tm_any = any(r["tm_base_verdict"] in ("FAMILY_KEEP", "FAMILY_WEAK") for r in summary_rows)
    rec_any = any(r["tm_recipe_verdict"] in ("FAMILY_KEEP", "FAMILY_WEAK") for r in summary_rows)
    aro = next(r for r in summary_rows if r["family"] == "AROMATIC_EXPOSURE")
    hyd = next(r for r in summary_rows if r["family"] == "HYDROPHOBIC_EXPOSURE")
    fixed_survive = any(
        (r["tm_base_verdict"] in ("FAMILY_KEEP", "FAMILY_WEAK") and r["tm_base_fixed_P"] is not None and r["tm_base_fixed_P"] >= 0 and r["tm_base_fixed_S"] >= 0)
        or (r["hic_verdict"] in ("FAMILY_KEEP", "FAMILY_WEAK") and r["hic_fixed_P"] is not None and r["hic_fixed_P"] >= 0 and r["hic_fixed_S"] >= 0)
        for r in summary_rows
    )

    lines = [
        "# OpenMM MD Family Ablation — 最終レポート",
        "",
        "全36特徴ブロックは有害だったため、事前凍結した family 単位で最終確認。新規 MD / FeNNix 再開なし。",
        "",
        "## 冒頭 Q&A",
        "",
        f"1. **凍結 family？** {', '.join(families)}",
        f"2. **TmApp BASE で Primary∩Shadow 改善？** {yn(tm_any)}",
        f"3. **BioEmu+MPNN で Primary∩Shadow 改善？** {yn(rec_any)}",
        f"4. **AROMATIC_EXPOSURE が HIC で両方改善？** {yn(aro['hic_P'] is not None and aro['hic_S'] is not None and aro['hic_P']>0 and aro['hic_S']>0)}（P={aro['hic_P']}, S={aro['hic_S']}）",
        f"5. **HYDROPHOBIC_EXPOSURE が HIC で両方改善？** {yn(hyd['hic_P'] is not None and hyd['hic_S'] is not None and hyd['hic_P']>0 and hyd['hic_S']>0)}（P={hyd['hic_P']}, S={hyd['hic_S']}）",
        f"6. **正の結果が BASE_FIXED_ALPHA 生存？** {yn(fixed_survive)}",
        f"7. **Test に進める family？** {('organizer確認待ち: '+str(survivors)) if survivors else 'なし（Test生成しない）'}",
        f"8. **最終 OpenMM 判定？** **{closure}**",
        "",
        "## Family map",
        "",
        fmap.groupby("ablation_family").size().to_string(),
        "",
        "## Summary",
        "",
        summ.to_string(index=False),
        "",
        "## Full results",
        "",
        df.to_string(index=False),
        "",
    ]
    (OUT / "OPENMM_MD_FAMILY_ABLATION_REPORT_JA.md").write_text("\n".join(lines) + "\n")
    (RES / "OPENMM_MD_FAMILY_ABLATION_REPORT_JA.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
