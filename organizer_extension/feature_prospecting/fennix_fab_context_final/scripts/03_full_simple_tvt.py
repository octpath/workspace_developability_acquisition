#!/usr/bin/env python3
"""Full-cohort Simple TVT replay for frozen FeNNix families (exact rescreen protocol)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FINAL = FP / "fennix_fab_context_final/results"
INTERIM = FP / "fennix_fab_context_interim_audit"
RES_I = INTERIM / "results"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
B_BOOT = 10000
RNG = np.random.default_rng(42)

# import rescreen helpers
spec = importlib.util.spec_from_file_location(
    "tvt_rescreen",
    INTERIM / "scripts/run_simple_tvt_rescreen.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)


FAMS = [
    "DELTA_GEOM",
    "DELTA_ENV",
    "CONSTANT",
    "INTERFACE",
    "FULL_FAB_NORMALIZED",
    "PREP_RELAX_SENSITIVITY",
    "COMBINED_PREDECLARED",
]


def numeric_X(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return mod.numeric_X(df)


def bootstrap_p_improve(y, base_oof, plus_oof, b=B_BOOT):
    y = np.asarray(y, float)
    e0 = np.abs(y - np.asarray(base_oof, float))
    e1 = np.abs(y - np.asarray(plus_oof, float))
    d = e0 - e1
    n = len(d)
    boots = []
    for _ in range(b):
        idx = RNG.integers(0, n, n)
        boots.append(float(d[idx].mean()))
    boots = np.asarray(boots)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    p_improve = float((boots <= 0).mean())  # one-sided: improvement means d>0
    return ci, p_improve, float(d.mean())


def run_protocol_with_oof(base, struct, y, folds, ids, mode):
    """Like rescreen run_protocol but also return OOFs for bootstrap."""
    common = sorted(set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    cov_ok = True
    for k, tr, va, te in mod.rotation_splits(folds, common):
        ok = len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        cov_ok = cov_ok and ok
    if not cov_ok or len(common) < 40:
        return None

    Xb = base.loc[common]
    Xs = struct.loc[common]
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    for k, tr, va, te in mod.rotation_splits(folds, common):
        a_base = mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=False)
        a_plus = a_base if mode != "FREE_ALPHA" else mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=True)
        if mode == "FREE_ALPHA":
            pass
        else:
            a_plus = a_base
        pb = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_base, use_struct=False)
        pp = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_plus, use_struct=True)
        base_oof.loc[te] = pb
        plus_oof.loc[te] = pp
    return {
        "N": len(common),
        "base_dim": int(Xb.shape[1]),
        "struct_dim": int(Xs.shape[1]),
        "base_mae": mod.mae(yy, base_oof),
        "plus_mae": mod.mae(yy, plus_oof),
        "delta": mod.mae(yy, base_oof) - mod.mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
    }


def replicate_label(i_dp, i_ds, f_dp, f_ds):
    if i_dp > 0 and i_ds > 0 and f_dp > 0 and f_ds > 0:
        if min(f_dp, f_ds) >= 0.7 * min(i_dp, i_ds):
            return "REPLICATED"
        return "ATTENUATED_BUT_POSITIVE"
    if i_dp > 0 and i_ds > 0 and (f_dp <= 0 or f_ds <= 0):
        if f_dp < 0 and f_ds < 0:
            return "SIGN_REVERSED"
        return "DID_NOT_REPLICATE"
    return "INCONCLUSIVE"


def main():
    print("load bases", flush=True)
    bases, _ = mod.load_bases()
    base = bases["TmApp_BASE"]
    y = mod.load_y("TmApp")
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    fennix_ids = set(pd.read_csv(FINAL / "FINAL_FENNIX_COHORT.csv").id.astype(str))
    # usable Dev = intersection folds ∩ fennix
    usable_dev = sorted(set(primary.id) & fennix_ids)
    print(f"full Dev={len(primary)} usable FeNNix Dev={len(usable_dev)} missing={sorted(set(primary.id)-fennix_ids)}", flush=True)

    rows = []
    boot_rows = []
    for fam in FAMS:
        X = numeric_X(FINAL / f"FENNIX_{fam}.parquet")
        print(f"→ {fam} dim={X.shape[1]}", flush=True)
        by = {}
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            for mode in ("FREE_ALPHA", "BASE_FIXED_ALPHA"):
                out = run_protocol_with_oof(base, X, y, folds, usable_dev, mode)
                by[(tag, mode)] = out
                if out is None:
                    print("  FAIL coverage", tag, mode)
        pf, sf = by[("Primary", "FREE_ALPHA")], by[("Shadow", "FREE_ALPHA")]
        pfix, sfix = by[("Primary", "BASE_FIXED_ALPHA")], by[("Shadow", "BASE_FIXED_ALPHA")]
        if not pf or not sf:
            continue
        row = {
            "family": fam,
            "N": pf["N"],
            "struct_dim": pf["struct_dim"],
            "primary_base_mae": pf["base_mae"],
            "primary_cand_mae": pf["plus_mae"],
            "primary_delta": pf["delta"],
            "shadow_base_mae": sf["base_mae"],
            "shadow_cand_mae": sf["plus_mae"],
            "shadow_delta": sf["delta"],
            "fixed_primary_delta": pfix["delta"] if pfix else np.nan,
            "fixed_shadow_delta": sfix["delta"] if sfix else np.nan,
            "sci_verdict": mod.scientific_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
            "comp_verdict": mod.competition_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
        }
        rows.append(row)
        if pf["delta"] > 0 and sf["delta"] > 0:
            ci, p_imp, mean_d = bootstrap_p_improve(pf["y"], pf["base_oof"], pf["plus_oof"])
            boot_rows.append(
                {
                    "family": fam,
                    "split": "Primary",
                    "mean_delta_err": mean_d,
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    "p_improve": p_imp,
                }
            )
            ci, p_imp, mean_d = bootstrap_p_improve(sf["y"], sf["base_oof"], sf["plus_oof"])
            boot_rows.append(
                {
                    "family": fam,
                    "split": "Shadow",
                    "mean_delta_err": mean_d,
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    "p_improve": p_imp,
                }
            )

    # BIOEMU_NEW_CONTACT + CONSTANT
    bio = pd.read_csv(BIO)
    bio["id"] = bio.id.astype(str)
    contact_cols = [c for c in bio.columns if "contact" in c.lower()]
    bio_x = mod.numeric_X(bio[["id"] + contact_cols])
    const = numeric_X(FINAL / "FENNIX_CONSTANT.parquet")
    combo = bio_x.join(const, how="inner")
    combo = combo.loc[:, ~combo.columns.duplicated()]
    print(f"→ BIOEMU_NEW_CONTACT+CONSTANT dim={combo.shape[1]}", flush=True)
    by = {}
    for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
        for mode in ("FREE_ALPHA", "BASE_FIXED_ALPHA"):
            by[(tag, mode)] = run_protocol_with_oof(base, combo, y, folds, usable_dev, mode)
    pf, sf = by[("Primary", "FREE_ALPHA")], by[("Shadow", "FREE_ALPHA")]
    pfix, sfix = by[("Primary", "BASE_FIXED_ALPHA")], by[("Shadow", "BASE_FIXED_ALPHA")]
    if pf and sf:
        rows.append(
            {
                "family": "BIOEMU_NEW_CONTACT+CONSTANT",
                "N": pf["N"],
                "struct_dim": pf["struct_dim"],
                "primary_base_mae": pf["base_mae"],
                "primary_cand_mae": pf["plus_mae"],
                "primary_delta": pf["delta"],
                "shadow_base_mae": sf["base_mae"],
                "shadow_cand_mae": sf["plus_mae"],
                "shadow_delta": sf["delta"],
                "fixed_primary_delta": pfix["delta"],
                "fixed_shadow_delta": sfix["delta"],
                "sci_verdict": mod.scientific_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
                "comp_verdict": mod.competition_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
            }
        )
        if pf["delta"] > 0 and sf["delta"] > 0:
            for tag, out in [("Primary", pf), ("Shadow", sf)]:
                ci, p_imp, mean_d = bootstrap_p_improve(out["y"], out["base_oof"], out["plus_oof"])
                boot_rows.append(
                    {
                        "family": "BIOEMU_NEW_CONTACT+CONSTANT",
                        "split": tag,
                        "mean_delta_err": mean_d,
                        "ci_lo": ci[0],
                        "ci_hi": ci[1],
                        "p_improve": p_imp,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FINAL / "FENNIX_FULL_SIMPLE_TVT_RESULTS.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(FINAL / "FENNIX_FULL_SIMPLE_TVT_BOOTSTRAP.csv", index=False)
    print(out_df.to_string(index=False))

    # interim vs full
    interim = pd.read_csv(INTERIM / "SIMPLE_TVT_CV_RESULTS.csv")
    # also rescreen for PREP
    rescreen = pd.read_csv(INTERIM / "SIMPLE_TVT_ALL_BLOCKS.csv") if (INTERIM / "SIMPLE_TVT_ALL_BLOCKS.csv").exists() else pd.DataFrame()
    ivf = []
    for fam in FAMS + ["BIOEMU_NEW_CONTACT+CONSTANT"]:
        fr = out_df[out_df.family == fam]
        if not len(fr):
            continue
        fr = fr.iloc[0]
        # interim lookup
        i_n = i_pd = i_sd = np.nan
        i_ver = ""
        hit = interim[interim.family.astype(str).str.replace("INTERIM_", "") == fam.replace("BIOEMU_NEW_CONTACT+CONSTANT", "___none___")]
        if fam in set(interim.family.astype(str)):
            hit = interim[interim.family == fam]
        elif fam == "CONSTANT":
            hit = interim[interim.family == "CONSTANT"]
        if len(hit):
            # SIMPLE_TVT_CV_RESULTS schema
            r = hit.iloc[0]
            i_n = r.get("N_primary", r.get("N", 100))
            i_pd = r.get("primary_delta", np.nan)
            i_sd = r.get("shadow_delta", np.nan)
            i_ver = r.get("verdict", "")
        elif fam == "PREP_RELAX_SENSITIVITY" and len(rescreen):
            hit = rescreen[rescreen.family.astype(str).str.contains("PREP_RELAX")]
            if len(hit):
                r = hit.iloc[0]
                i_n = r.get("N", 100)
                i_pd = r.get("primary_delta_free", r.get("primary_delta", np.nan))
                i_sd = r.get("shadow_delta_free", r.get("shadow_delta", np.nan))
                i_ver = r.get("sci_verdict", "")
        elif fam == "BIOEMU_NEW_CONTACT+CONSTANT" and len(rescreen):
            hit = rescreen[rescreen.family.astype(str).str.contains("BIOEMU_NEW_CONTACT") & rescreen.family.astype(str).str.contains("CONSTANT")]
            if len(hit):
                r = hit.iloc[0]
                i_n = r.get("N", 100)
                i_pd = r.get("primary_delta_free", r.get("primary_delta", np.nan))
                i_sd = r.get("shadow_delta_free", r.get("shadow_delta", np.nan))
                i_ver = r.get("sci_verdict", "")
        ivf.append(
            {
                "family": fam,
                "interim_N": i_n,
                "interim_Primary_delta": i_pd,
                "interim_Shadow_delta": i_sd,
                "interim_verdict": i_ver,
                "full_N": fr.N,
                "full_Primary_delta": fr.primary_delta,
                "full_Shadow_delta": fr.shadow_delta,
                "full_verdict": fr.sci_verdict,
                "replication": replicate_label(i_pd, i_sd, fr.primary_delta, fr.shadow_delta)
                if pd.notna(i_pd)
                else "INCONCLUSIVE",
            }
        )
    pd.DataFrame(ivf).to_csv(FINAL / "FENNIX_INTERIM_VS_FULL.csv", index=False)
    (FINAL / "USABLE_DEV.json").write_text(
        json.dumps({"full_dev": int(len(primary)), "usable_fennix_dev": len(usable_dev), "missing_dev": sorted(set(primary.id) - fennix_ids)}, indent=2)
    )
    print("wrote results", flush=True)


if __name__ == "__main__":
    main()
