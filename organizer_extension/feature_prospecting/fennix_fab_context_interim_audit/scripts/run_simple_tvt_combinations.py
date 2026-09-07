#!/usr/bin/env python3
"""Limited competition combinations after single-block Simple TVT freeze."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit"
SCRIPT = OUT / "scripts/run_simple_tvt_rescreen.py"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"

spec = importlib.util.spec_from_file_location("rescreen", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def concat_structs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    X = pd.concat(parts, axis=1, join="inner")
    return X.loc[:, ~X.columns.duplicated()]


def main():
    print("Loading bases + catalog…", flush=True)
    bases, base_names = mod.load_bases()
    specs = mod.build_family_catalog()
    by_key = {(s["target"], s["family"]): s for s in specs}
    all_df = pd.read_csv(OUT / "SIMPLE_TVT_ALL_BLOCKS.csv")

    # Shortlist: Primary∧Shadow free>0, prefer survives_fixed, non-redundant
    short = all_df[(all_df.free_primary_delta > 0) & (all_df.free_shadow_delta > 0)].copy()
    short["worst"] = short[["free_primary_delta", "free_shadow_delta"]].min(axis=1)
    short = short.sort_values(["target", "worst"], ascending=[True, False])

    # Competition shortlist table
    shortlist_rows = []
    for target in ["TmApp", "HIC"]:
        sub = short[short.target == target].head(8)
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            shortlist_rows.append(
                {
                    "rank": rank,
                    "target": target,
                    "family": r.family,
                    "free_primary_delta": r.free_primary_delta,
                    "free_shadow_delta": r.free_shadow_delta,
                    "free_worst_delta": r.worst,
                    "survives_fixed_alpha": r.survives_fixed_alpha,
                    "SCIENTIFIC_VERDICT": r.SCIENTIFIC_VERDICT,
                    "COMPETITION_VERDICT": r.COMPETITION_VERDICT,
                    "selected_for_combo": rank <= 5,
                    "notes": "non-redundant preferred: skip nested BioEmu combined with parts",
                }
            )
    pd.DataFrame(shortlist_rows).to_csv(OUT / "SIMPLE_TVT_COMPETITION_SHORTLIST.csv", index=False)

    # Prespecified combinations (names must exist in catalog)
    combos = [
        # TmApp
        ("TmApp", "BIOEMU_NEW_PAIRWISE+INTERIM_CONSTANT", ["BIOEMU_NEW_PAIRWISE", "INTERIM_CONSTANT"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_CONTACT+INTERIM_CONSTANT", ["BIOEMU_NEW_CONTACT", "INTERIM_CONSTANT"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_PAIRWISE+M1_PROTEINMPNN", ["BIOEMU_NEW_PAIRWISE", "M1_PROTEINMPNN"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_FLEX+INTERIM_CONSTANT", ["BIOEMU_NEW_FLEX", "INTERIM_CONSTANT"], "TmApp_BASE"),
        ("TmApp", "INTERIM_CONSTANT+M1_PROTEINMPNN", ["INTERIM_CONSTANT", "M1_PROTEINMPNN"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_PAIRWISE+S2_GENERATOR_DISAGREEMENT", ["BIOEMU_NEW_PAIRWISE", "S2_GENERATOR_DISAGREEMENT"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_CONTACT+BIOEMU_NEW_PAIRWISE+INTERIM_CONSTANT", ["BIOEMU_NEW_CONTACT", "BIOEMU_NEW_PAIRWISE", "INTERIM_CONSTANT"], "TmApp_BASE"),
        ("TmApp", "BIOEMU_NEW_PAIRWISE+INTERIM_CONSTANT+M1_PROTEINMPNN", ["BIOEMU_NEW_PAIRWISE", "INTERIM_CONSTANT", "M1_PROTEINMPNN"], "TmApp_BASE"),
        # HIC
        ("HIC", "CONTINUOUS_SURFACE+HYDRO_FIELD", ["CONTINUOUS_SURFACE", "HYDRO_FIELD"], "HIC_BASE_ARO"),
        ("HIC", "CONTINUOUS_SURFACE+TITRATION_SHAPE", ["CONTINUOUS_SURFACE", "TITRATION_SHAPE"], "HIC_BASE_ARO"),
        ("HIC", "HYDRO_FIELD+TITRATION_SHAPE", ["HYDRO_FIELD", "TITRATION_SHAPE"], "HIC_BASE_ARO"),
        ("HIC", "CONTINUOUS_SURFACE+HYDRO_FIELD+TITRATION_SHAPE", ["CONTINUOUS_SURFACE", "HYDRO_FIELD", "TITRATION_SHAPE"], "HIC_BASE_ARO"),
        ("HIC", "HIC_SURFACE_ALL+HYDRO_FIELD", ["HIC_SURFACE_ALL", "HYDRO_FIELD"], "HIC_BASE"),
    ]

    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)
    y_tm, y_hic = mod.load_y("TmApp"), mod.load_y("HIC")
    usable = mod.interim_usable()

    rows = []
    for target, name, fams, base_key in combos:
        parts = []
        id_filter = None
        ok = True
        for f in fams:
            s = by_key.get((target, f))
            if s is None:
                print("MISSING", target, f, flush=True)
                ok = False
                break
            parts.append(s["X"])
            if s["id_filter"] is not None:
                id_filter = s["id_filter"] if id_filter is None else (id_filter & s["id_filter"])
        if not ok:
            continue
        # Interim blocks need usable filter even if other blocks are full Dev
        if any(f.startswith("INTERIM_") for f in fams):
            id_filter = usable
        X = concat_structs(parts)
        base = bases[base_key]
        y = y_tm if target == "TmApp" else y_hic
        ids = list(base.index)
        if id_filter is not None:
            ids = [i for i in ids if i in id_filter]
        print(f"combo {target} {name} dim={X.shape[1]} N_cand={len(ids)}", flush=True)
        by = {}
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            for mode in ("FREE_ALPHA", "BASE_FIXED_ALPHA"):
                out, _ = mod.run_protocol(base, X, y, folds, ids, mode)
                by[(tag, mode)] = out
        if any(by[k] is None for k in by):
            print("  SKIP underpowered", flush=True)
            continue
        pf, sf = by[("Primary", "FREE_ALPHA")], by[("Shadow", "FREE_ALPHA")]
        pfix, sfix = by[("Primary", "BASE_FIXED_ALPHA")], by[("Shadow", "BASE_FIXED_ALPHA")]
        rows.append(
            {
                "target": target,
                "combination": name,
                "blocks": "|".join(fams),
                "base_name": base_names[base_key],
                "N": pf["N"],
                "feature_dim": pf["struct_dim"],
                "free_primary_delta": pf["delta"],
                "free_shadow_delta": sf["delta"],
                "free_worst_delta": min(pf["delta"], sf["delta"]),
                "free_mean_delta": 0.5 * (pf["delta"] + sf["delta"]),
                "fixed_primary_delta": pfix["delta"],
                "fixed_shadow_delta": sfix["delta"],
                "fixed_worst_delta": min(pfix["delta"], sfix["delta"]),
                "survives_fixed_alpha": bool(pfix["delta"] > 0 and sfix["delta"] > 0),
                "free_primary_base_mae": pf["base_mae"],
                "free_primary_combined_mae": pf["plus_mae"],
                "free_shadow_base_mae": sf["base_mae"],
                "free_shadow_combined_mae": sf["plus_mae"],
                "SCIENTIFIC_VERDICT": mod.scientific_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
                "COMPETITION_VERDICT": mod.competition_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
            }
        )

    comb = pd.DataFrame(rows).sort_values(["target", "free_worst_delta"], ascending=[True, False])
    comb.to_csv(OUT / "SIMPLE_TVT_LIMITED_COMBINATIONS.csv", index=False)
    print(comb.to_string(index=False), flush=True)
    print("wrote combinations", len(comb), flush=True)


if __name__ == "__main__":
    main()
