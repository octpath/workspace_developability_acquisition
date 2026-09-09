#!/usr/bin/env python3
"""Classical feature refinement (CV-only until freeze; no prediction ensembles).

Preprocessing (canonical):
  per-block impute → PCA32 only on AbLingua / new PLM blocks → StandardScaler → estimator
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from classical_features.rasa_cache import build_or_load_rasa_cache  # noqa: E402
from classical_features.pooling import (  # noqa: E402
    aromatic_rasa_summaries,
    build_pooled_block,
    content_sha_df,
    load_annotations,
    load_plm_pack,
)
from classical_features.cv_eval import (  # noqa: E402
    evaluate_primary_shadow,
    fit_full_dev_predict,
    load_folds,
)
from _lib import (  # noqa: E402
    add_split_column,
    feature_content_sha256,
    file_sha256,
    load_block,
    load_dev_test_folds,
)
from experiment_codes import issue_code, load_codes  # noqa: E402

BUNDLE = REPO / "top_models_feature_bundle"
CACHE = ROOT / "experiments" / "classical_cache"
BLOCK_DIR = CACHE / "blocks"
CV_CACHE = CACHE / "cv_cache"
for d in (CACHE, BLOCK_DIR, CV_CACHE):
    d.mkdir(parents=True, exist_ok=True)

Group = tuple[list[str], bool]


def concat_parts(
    parts: list[tuple[pd.DataFrame, bool]], ids: list[str]
) -> tuple[pd.DataFrame, list[Group]]:
    """parts: (frame_with_id, do_pca)."""
    base = pd.DataFrame({"id": ids})
    seen: set[str] = set()
    mats = []
    groups: list[Group] = []
    for fr, do_pca in parts:
        fr = fr.copy()
        fr["id"] = fr["id"].astype(str)
        sub = fr.set_index("id").loc[ids]
        rename = {}
        for c in sub.columns:
            name = c
            n = 0
            while name in seen:
                n += 1
                name = f"{c}__d{n}"
            seen.add(name)
            rename[c] = name
        cols = list(rename.values())
        mats.append(sub.rename(columns=rename)[cols].reset_index(drop=True))
        groups.append((cols, bool(do_pca)))
    return pd.concat([base] + mats, axis=1), groups


def is_plm_block(bid: str) -> bool:
    if bid == "FB_ARO_RASA_SUM":
        return False
    return bid.startswith(("FB_AL", "FB_ESM2", "FB_AL2"))


def load_guided_ablingua(family: str, ids: list[str]) -> pd.DataFrame:
    path = (
        REPO
        / "organizer_extension/feature_prospecting/ablingua600m/embeddings_guided"
        / f"ablingua600m_{family}.parquet"
    )
    df = pd.read_parquet(path)
    df["id"] = df["id"].astype(str)
    return df.set_index("id").loc[ids].reset_index()


def recipe_parts(recipe_id: str, ids: list[str]) -> list[tuple[pd.DataFrame, bool]]:
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    blocks = str(recipes[recipes["recipe_id"] == recipe_id].iloc[0]["feature_blocks"]).split("|")
    out = []
    for b in blocks:
        out.append((load_block(b, ids), b.startswith("AbLingua")))
    return out


def matrix_key(name: str, target: str, estimator: str, groups: list[Group], n_features: int) -> str:
    gsig = "|".join(f"{len(c)}:{int(p)}" for c, p in groups)
    return hashlib.sha256(f"{name}|{target}|{estimator}|{n_features}|{gsig}".encode()).hexdigest()[:24]


def main() -> int:
    t0 = time.time()
    dev, test, _ = load_dev_test_folds()
    primary, shadow = load_folds(ROOT / "data" / "folds.csv")
    all_ids = dev["id"].astype(str).tolist() + test["id"].astype(str).tolist()
    ymaps = {
        "TmApp": {str(r.id): float(r.TmApp) for _, r in dev.iterrows()},
        "HIC": {str(r.id): float(r.HIC) for _, r in dev.iterrows()},
    }

    audit_rows = []
    for name, path, notes in [
        ("ablang2_residue", BUNDLE / "residue_level/ablang2", "H/L 480"),
        ("ablingua_residue", BUNDLE / "residue_level/ablingua600m", "H/L 1280"),
        ("esm2_residue", BUNDLE / "residue_level/esm2", "H 1280"),
        ("annotations", BUNDLE / "residue_level/annotations.parquet", "IMGT FR/CDR"),
        (
            "ablingua_guided",
            REPO / "organizer_extension/feature_prospecting/ablingua600m/embeddings_guided",
            "historical guided pools",
        ),
        ("aromatic_topo", BUNDLE / "data/aromatic_topo.parquet", "existing aromatic"),
        ("continuous_surface", BUNDLE / "data/continuous_surface.parquet", "existing surface"),
        ("esmfold_fv_pdb", REPO / "feature_extension/data/esmfold_fv", "RASA source"),
        ("bioemu", BUNDLE / "data/bioemu_new_pairwise.parquet", "BioEmu"),
        ("proteinmpnn", BUNDLE / "data/proteinmpnn.parquet", "ProteinMPNN"),
    ]:
        audit_rows.append(
            {"asset": name, "path": str(path), "exists": path.exists(), "notes": notes}
        )
    pd.DataFrame(audit_rows).to_csv(ROOT / "results/CLASSICAL_FEATURE_ASSET_AUDIT.csv", index=False)

    print("Building/loading RASA cache...", flush=True)
    rasa = build_or_load_rasa_cache(dev, test)
    ann = load_annotations()

    block_registry: list[dict] = []
    blocks: dict[str, pd.DataFrame] = {}

    def register_block(bid: str, df: pd.DataFrame, **meta):
        path = BLOCK_DIR / f"{bid}.parquet"
        if not path.exists():
            df.to_parquet(path, index=False)
        else:
            df = pd.read_parquet(path)
            df["id"] = df["id"].astype(str)
        sha = content_sha_df(df)
        nfeat = len([c for c in df.columns if c != "id"])
        blocks[bid] = df
        block_registry.append(
            {
                "feature_block_id": bid,
                "n_features": nfeat,
                "content_sha256": sha,
                "path": str(path.relative_to(ROOT)),
                **meta,
            }
        )
        print(f"  block {bid} dim={nfeat}", flush=True)

    for fam in ["CDR_ALL", "CDR3", "RASA_WEIGHTED", "EXPOSED_CDR", "CDR_FR_SPLIT", "RESIDUE_GLOBAL"]:
        register_block(
            f"FB_AL_{fam}",
            load_guided_ablingua(fam, all_ids),
            target_scope="TmApp",
            source_asset="ablingua600m+guided",
            derivation_type="historical_guided_pool",
            chain_scope="HL",
            region_scope=fam,
            weighting="as_frozen",
            license_status="REVIEW",
            notes="reused organizer guided pooling",
        )

    need_build = any(not (BLOCK_DIR / f"{b}.parquet").exists() for b in [
        "FB_AL2_GLOBAL", "FB_ESM2_GLOBAL", "FB_ARO_RASA_SUM"
    ])
    if need_build:
        print("Loading PLM packs for new pools...", flush=True)
        al2 = load_plm_pack("ablang2")
        al = load_plm_pack("ablingua600m")
        esm = load_plm_pack("esm2")
    else:
        al2 = al = esm = None
        print("Loading derived blocks from cache...", flush=True)

    def maybe_pool(bid, pack, chains, region, g, g3, rp, src, scope):
        path = BLOCK_DIR / f"{bid}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["id"] = df["id"].astype(str)
        else:
            df = build_pooled_block(
                pack, ann, all_ids, chains=chains, region=region,
                cdr_gamma=g, cdr3_gamma=g3,
                rasa=rasa if rp else None, rasa_power=rp, prefix=bid,
            )
        register_block(
            bid, df, target_scope=scope, source_asset=src, derivation_type="residue_pool",
            chain_scope="".join(chains), region_scope=region,
            weighting=f"cdr_g={g};cdr3_g={g3};rasa_p={rp}",
            license_status="REVIEW", notes="derived classical pool",
        )

    specs_tm = [
        ("FB_AL2_GLOBAL", "ablang2", ["H", "L"], "ALL", 1.0, None, None),
        ("FB_AL2_CDR_ALL", "ablang2", ["H", "L"], "CDR", 1.0, None, None),
        ("FB_AL2_CDR3", "ablang2", ["H", "L"], "CDR3", 1.0, None, None),
        ("FB_AL2_FR_CDR_SPLIT", "ablang2", ["H", "L"], "SPLIT_FR_CDR", 1.0, None, None),
        ("FB_AL2_CDRW2", "ablang2", ["H", "L"], "ALL", 2.0, None, None),
        ("FB_AL2_CDRW4", "ablang2", ["H", "L"], "ALL", 4.0, None, None),
        ("FB_AL2_CDR3W4", "ablang2", ["H", "L"], "ALL", 1.0, 4.0, None),
        ("FB_AL2_RASA_P1", "ablang2", ["H", "L"], "ALL", 1.0, None, 1.0),
        ("FB_AL2_RASA_CDR", "ablang2", ["H", "L"], "CDR", 1.0, None, 1.0),
        ("FB_AL_CDRW2", "ablingua", ["H", "L"], "ALL", 2.0, None, None),
        ("FB_AL_CDR3W4", "ablingua", ["H", "L"], "ALL", 1.0, 4.0, None),
        ("FB_AL_RASA_P2", "ablingua", ["H", "L"], "ALL", 1.0, None, 2.0),
    ]
    pack_map = {"ablang2": al2, "ablingua": al, "esm2": esm}
    for bid, src, chains, region, g, g3, rp in specs_tm:
        if pack_map[src] is None and not (BLOCK_DIR / f"{bid}.parquet").exists():
            al2 = load_plm_pack("ablang2"); al = load_plm_pack("ablingua600m"); esm = load_plm_pack("esm2")
            pack_map.update({"ablang2": al2, "ablingua": al, "esm2": esm})
        maybe_pool(bid, pack_map[src], chains, region, g, g3, rp, src, "TmApp")

    # fusion from cached globals
    fusion, _ = concat_parts(
        [(blocks["FB_AL2_GLOBAL"], True), (blocks["FB_AL_RESIDUE_GLOBAL"], True)], all_ids
    )
    register_block(
        "FB_AL2_AL_CONCAT", fusion, target_scope="TmApp", source_asset="ablang2+ablingua",
        derivation_type="fixed_concat", chain_scope="HL", region_scope="GLOBAL",
        weighting="none", license_status="REVIEW", notes="AbLang2+AbLingua fixed concat",
    )

    specs_hic = [
        ("FB_ESM2_GLOBAL", ["H"], "ALL", 1.0, None, None),
        ("FB_ESM2_CDR_ALL", ["H"], "CDR", 1.0, None, None),
        ("FB_ESM2_CDR3", ["H"], "CDR3", 1.0, None, None),
        ("FB_ESM2_FR_CDR_SPLIT", ["H"], "SPLIT_FR_CDR", 1.0, None, None),
        ("FB_ESM2_CDRW2", ["H"], "ALL", 2.0, None, None),
        ("FB_ESM2_CDRW4", ["H"], "ALL", 4.0, None, None),
        ("FB_ESM2_CDR3W4", ["H"], "ALL", 1.0, 4.0, None),
        ("FB_ESM2_RASA_P1", ["H"], "ALL", 1.0, None, 1.0),
        ("FB_ESM2_RASA_P2", ["H"], "ALL", 1.0, None, 2.0),
        ("FB_ESM2_RASA_CDR", ["H"], "CDR", 1.0, None, 1.0),
        ("FB_ESM2_RASA_CDR3", ["H"], "CDR3", 1.0, None, 1.0),
    ]
    for bid, chains, region, g, g3, rp in specs_hic:
        if pack_map.get("esm2") is None and not (BLOCK_DIR / f"{bid}.parquet").exists():
            pack_map["esm2"] = load_plm_pack("esm2")
        maybe_pool(bid, pack_map.get("esm2"), chains, region, g, g3, rp, "esm2", "HIC")

    if (BLOCK_DIR / "FB_ARO_RASA_SUM.parquet").exists():
        aro = pd.read_parquet(BLOCK_DIR / "FB_ARO_RASA_SUM.parquet")
        aro["id"] = aro["id"].astype(str)
    else:
        aro = aromatic_rasa_summaries(dev, test, ann, rasa)
    register_block(
        "FB_ARO_RASA_SUM", aro, target_scope="BOTH", source_asset="esmfold_fv+annotations",
        derivation_type="aromatic_rasa_summary", chain_scope="HL", region_scope="ALL/CDR/CDR3",
        weighting="rasa_clip01", license_status="OK", notes="Y/F/W RASA summaries",
    )
    pd.DataFrame(block_registry).to_csv(ROOT / "results/FEATURE_BLOCKS.csv", index=False)

    bases = {
        "TmApp": [
            ("FS_TM_ABLINGUA_CDR3", "TM_PARENT_ABLINGUA_CDR3__RIDGE"),
            ("FS_TM_ABLINGUA_GLOBAL", "TM_PARENT_ABLINGUA_GLOBAL__RIDGE"),
            ("FS_TM_BIOEMU_MPNN", "TM_BASE_BIOEMU_MPNN__RIDGE"),
        ],
        "HIC": [
            ("FS_HIC_CONTINUOUS_SURFACE", "HIC_ARO_CONTINUOUS_SURFACE__LASSO"),
            ("FS_HIC_HYDRO_TITRATION", "HIC_HYDRO_TITRATION__LASSO"),
            ("FS_HIC_ESM2_SEQ_AROMATIC", "HIC_ESM2_SEQ_AROMATIC__LASSO"),
        ],
    }
    base_parts: dict[str, list[tuple[pd.DataFrame, bool]]] = {}
    for tgt, lst in bases.items():
        for fsid, rid in lst:
            base_parts[fsid] = recipe_parts(rid, all_ids)
            X, g = concat_parts(base_parts[fsid], all_ids)
            print(f"base {fsid} dim={X.shape[1]-1} pca={sum(1 for _,p in g if p)}", flush=True)

    screen_rows: list[dict] = []
    eval_memo: dict[str, dict] = {}

    def build_set(part_list: list[tuple[pd.DataFrame, bool]]) -> tuple[pd.DataFrame, list[Group]]:
        return concat_parts(part_list, all_ids)

    def eval_set(
        name: str,
        X: pd.DataFrame,
        groups: list[Group],
        target: str,
        estimator: str = "ridge",
        *,
        keep_oof: bool = False,
    ) -> dict:
        key = matrix_key(name, target, estimator, groups, int(X.shape[1] - 1))
        cache_path = CV_CACHE / f"{key}.json"
        need_oof = keep_oof
        res = None
        if key in eval_memo and (not need_oof or "oof_primary" in eval_memo[key]):
            res = eval_memo[key]
        elif cache_path.exists():
            res = json.loads(cache_path.read_text())
            if need_oof:
                op, os_ = CV_CACHE / f"{key}_op.csv", CV_CACHE / f"{key}_os.csv"
                if op.exists() and os_.exists():
                    res["oof_primary"] = pd.read_csv(op).set_index("id")["pred"]
                    res["oof_shadow"] = pd.read_csv(os_).set_index("id")["pred"]
                else:
                    res = None  # fall through to recompute
            if res is not None:
                eval_memo[key] = res
        if res is None or (need_oof and "oof_primary" not in res):
            Xdev = X[X["id"].isin(dev["id"].astype(str))].copy()
            full = evaluate_primary_shadow(
                Xdev, ymaps[target], primary, shadow, estimator=estimator, groups=groups
            )
            slim = {k: float(full[k]) for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")}
            slim["params_primary"] = full["params_primary"]
            slim["params_shadow"] = full["params_shadow"]
            cache_path.write_text(json.dumps(slim, default=str))
            full["oof_primary"].rename("pred").to_csv(CV_CACHE / f"{key}_op.csv", index_label="id")
            full["oof_shadow"].rename("pred").to_csv(CV_CACHE / f"{key}_os.csv", index_label="id")
            res = {**slim, "oof_primary": full["oof_primary"], "oof_shadow": full["oof_shadow"]}
            if not need_oof:
                # drop heavy oof from memo for screening
                eval_memo[key] = slim
            else:
                eval_memo[key] = res
        row = {
            "feature_set_name": name, "target": target, "estimator": estimator,
            "n_features": int(X.shape[1] - 1),
            "cv_primary_mae": res["cv_primary_mae"], "cv_shadow_mae": res["cv_shadow_mae"],
            "cv_mean_mae": res["cv_mean_mae"], "cv_worst_mae": res["cv_worst_mae"],
        }
        screen_rows.append(row)
        print(
            f"  {estimator} {name}: P={res['cv_primary_mae']:.4f} "
            f"S={res['cv_shadow_mae']:.4f} W={res['cv_worst_mae']:.4f}",
            flush=True,
        )
        out = {**{k: res[k] for k in row if k in res}, **row, "groups": groups}
        for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae"):
            out[k] = res[k]
        if keep_oof:
            out["X"] = X
            out["oof_primary"] = res["oof_primary"]
            out["oof_shadow"] = res["oof_shadow"]
        return out

    tm_new = [b for b in [
        "FB_AL2_AL_CONCAT", "FB_AL2_CDR_ALL", "FB_AL2_CDR3", "FB_AL2_FR_CDR_SPLIT",
        "FB_AL2_CDRW2", "FB_AL2_CDRW4", "FB_AL2_CDR3W4", "FB_AL2_RASA_P1", "FB_AL2_RASA_CDR",
        "FB_AL_CDRW2", "FB_AL_CDR3W4", "FB_AL_RASA_P2", "FB_AL_CDR_ALL", "FB_AL_CDR3",
        "FB_AL_RASA_WEIGHTED", "FB_ARO_RASA_SUM",
    ] if b in blocks][:16]
    hic_new = [b for b in [
        "FB_ESM2_CDR_ALL", "FB_ESM2_CDR3", "FB_ESM2_FR_CDR_SPLIT", "FB_ESM2_CDRW2",
        "FB_ESM2_CDRW4", "FB_ESM2_CDR3W4", "FB_ESM2_RASA_P1", "FB_ESM2_RASA_P2",
        "FB_ESM2_RASA_CDR", "FB_ESM2_RASA_CDR3", "FB_ARO_RASA_SUM", "FB_ESM2_GLOBAL",
    ] if b in blocks][:14]

    _RK = ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae",
           "feature_set_name", "target", "estimator", "n_features")

    print("\n=== STAGE A: single-block screen ===", flush=True)
    stage_a: list[dict] = []
    for tgt, new_ids, base_list in [
        ("TmApp", tm_new, bases["TmApp"]),
        ("HIC", hic_new, bases["HIC"]),
    ]:
        for bid in new_ids:
            parts = [(blocks[bid], is_plm_block(bid))]
            X, g = build_set(parts)
            r = eval_set(f"ALONE__{bid}", X, g, tgt, "ridge")
            stage_a.append({**{k: r[k] for k in _RK}, "stage": "A_alone", "block": bid})
            for fsid, _ in base_list:
                parts = list(base_parts[fsid]) + [(blocks[bid], is_plm_block(bid))]
                X, g = build_set(parts)
                r = eval_set(f"{fsid}+{bid}", X, g, tgt, "ridge")
                stage_a.append({**{k: r[k] for k in _RK}, "stage": "A_add", "block": bid, "base": fsid})

    def block_score(tgt, bid):
        rows = [r for r in stage_a if r["target"] == tgt and r.get("block") == bid and r["stage"] == "A_add"]
        return min((r["cv_worst_mae"] for r in rows), default=1e9)

    print("\n=== STAGE B: forward selection ===", flush=True)
    forward_rows = []
    selected_sets: dict[str, list] = {}
    for tgt, new_ids, base_list in [
        ("TmApp", tm_new, bases["TmApp"]),
        ("HIC", hic_new, bases["HIC"]),
    ]:
        cand_blocks = sorted(new_ids, key=lambda b: block_score(tgt, b))[:8]
        paths = []
        for fsid, _ in base_list:
            cur_ids: list[str] = []
            cur_parts = list(base_parts[fsid])
            X, g = build_set(cur_parts)
            cur_name = fsid
            cur_res = eval_set(f"B_start__{fsid}", X, g, tgt, "ridge")
            forward_rows.append({
                "target": tgt, "start": fsid, "step": 0, "accepted": "", "feature_set": cur_name,
                "cv_primary_mae": cur_res["cv_primary_mae"], "cv_shadow_mae": cur_res["cv_shadow_mae"],
                "cv_worst_mae": cur_res["cv_worst_mae"], "delta_worst": 0.0,
            })
            remaining = list(cand_blocks)
            for step in range(1, 4):
                best = None
                for bid in remaining:
                    parts = cur_parts + [(blocks[bid], is_plm_block(bid))]
                    X2, g2 = build_set(parts)
                    r = eval_set(f"B_{fsid}__{'+'.join(cur_ids+[bid])}", X2, g2, tgt, "ridge")
                    score = r["cv_worst_mae"]
                    if best is None or score < best[0] - 1e-12 or (
                        abs(score - best[0]) < 1e-12 and r["cv_mean_mae"] < best[1]["cv_mean_mae"]
                    ):
                        best = (score, r, bid, parts, g2, X2)
                if best is None or best[0] >= cur_res["cv_worst_mae"] - 1e-12:
                    forward_rows.append({
                        "target": tgt, "start": fsid, "step": step, "accepted": "STOP",
                        "feature_set": cur_name, "cv_primary_mae": cur_res["cv_primary_mae"],
                        "cv_shadow_mae": cur_res["cv_shadow_mae"], "cv_worst_mae": cur_res["cv_worst_mae"],
                        "delta_worst": 0.0,
                    })
                    break
                bid = best[2]
                delta = cur_res["cv_worst_mae"] - best[0]
                cur_ids.append(bid)
                remaining.remove(bid)
                cur_parts = best[3]
                cur_name = f"{fsid}+{'+'.join(cur_ids)}"
                cur_res = best[1]
                forward_rows.append({
                    "target": tgt, "start": fsid, "step": step, "accepted": bid,
                    "feature_set": cur_name, "cv_primary_mae": cur_res["cv_primary_mae"],
                    "cv_shadow_mae": cur_res["cv_shadow_mae"], "cv_worst_mae": cur_res["cv_worst_mae"],
                    "delta_worst": delta,
                })
            paths.append((cur_name, *build_set(cur_parts), [fsid] + cur_ids, cur_parts, cur_res))
        selected_sets[tgt] = paths
    pd.DataFrame(forward_rows).to_csv(ROOT / "results/FEATURE_FORWARD_SELECTION.csv", index=False)

    print("\n=== STAGE C: ablation ===", flush=True)
    abl_rows = []
    for tgt, paths in selected_sets.items():
        for name, X, g, blist, parts, res in paths:
            if len(blist) <= 1:
                continue
            base_id = blist[0]
            added = blist[1:]
            for drop in added:
                keep = [b for b in added if b != drop]
                parts2 = list(base_parts[base_id]) + [(blocks[b], is_plm_block(b)) for b in keep]
                X2, g2 = build_set(parts2)
                r = eval_set(f"ABL_{name}_drop_{drop}", X2, g2, tgt, "ridge")
                abl_rows.append({
                    "target": tgt, "feature_set": name, "dropped_block": drop,
                    "cv_primary_mae": r["cv_primary_mae"], "cv_shadow_mae": r["cv_shadow_mae"],
                    "cv_worst_mae": r["cv_worst_mae"],
                    "delta_primary": r["cv_primary_mae"] - res["cv_primary_mae"],
                    "delta_shadow": r["cv_shadow_mae"] - res["cv_shadow_mae"],
                    "delta_worst": r["cv_worst_mae"] - res["cv_worst_mae"],
                })
    pd.DataFrame(abl_rows).to_csv(ROOT / "results/FEATURE_BLOCK_ABLATION.csv", index=False)

    print("\n=== STAGE D: block scaling ===", flush=True)
    scale_rows = []
    scaled_sets: dict[str, list] = {t: [] for t in selected_sets}
    for tgt, paths in selected_sets.items():
        paths_sorted = sorted(paths, key=lambda x: x[5]["cv_worst_mae"])[:2]
        for name, X, g, blist, parts, res in paths_sorted:
            if len(blist) <= 1:
                scaled_sets[tgt].append((name, X, g, blist, res))
                continue
            base_id = blist[0]
            added = blist[1:]
            best_local = (res["cv_worst_mae"], name, X, g, blist, res)
            for scale in (0.5, 2.0):
                scaled_frames = []
                for bid in added:
                    df = blocks[bid].copy()
                    cols = [c for c in df.columns if c != "id"]
                    df[cols] = df[cols].astype(float) * scale
                    scaled_frames.append((df, is_plm_block(bid)))
                parts2 = list(base_parts[base_id]) + scaled_frames
                X2, g2 = build_set(parts2)
                r = eval_set(f"SCALE_{name}_x{scale}", X2, g2, tgt, "ridge")
                scale_rows.append({
                    "target": tgt, "feature_set": name, "secondary_scale": scale,
                    "cv_worst_mae": r["cv_worst_mae"], "cv_primary_mae": r["cv_primary_mae"],
                    "cv_shadow_mae": r["cv_shadow_mae"],
                })
                if r["cv_worst_mae"] < best_local[0]:
                    best_local = (r["cv_worst_mae"], f"{name}__scale{scale}", X2, g2, blist, r)
            scaled_sets[tgt].append(
                (best_local[1], best_local[2], best_local[3], best_local[4], best_local[5])
            )
    pd.DataFrame(scale_rows).to_csv(ROOT / "results/FEATURE_BLOCK_SCALING.csv", index=False)

    print("\n=== STAGE E: estimator panel ===", flush=True)
    final_pool: dict[str, list] = {}
    for tgt in ("TmApp", "HIC"):
        pool = list(scaled_sets[tgt])
        alone = sorted(
            [r for r in stage_a if r["target"] == tgt and r["stage"] == "A_alone"],
            key=lambda r: r["cv_worst_mae"],
        )[:2]
        for r in alone:
            bid = r["block"]
            X, g = build_set([(blocks[bid], is_plm_block(bid))])
            # re-eval to attach X/groups (cached)
            rr = eval_set(f"ALONE__{bid}", X, g, tgt, "ridge")
            pool.append((f"ALONE__{bid}", X, g, [bid], rr))
        uniq = {}
        for item in pool:
            uniq[item[0]] = item
        top = sorted(uniq.values(), key=lambda x: x[4]["cv_worst_mae"])[:6]
        final_pool[tgt] = top

    candidates = []
    cand_order = 0
    estimators = ["ridge", "lasso", "enet", "svr", "xgb"]
    for tgt in ("TmApp", "HIC"):
        for name, X, g, blist, _ in final_pool[tgt]:
            for est in estimators:
                cand_order += 1
                candidates.append({
                    "candidate_order": cand_order, "target": tgt, "feature_set_name": name,
                    "blocks": "|".join(blist), "estimator": est, "stage": "E",
                    "reason": "top feature sets × estimator panel; CV-only",
                })
    cand_df = pd.DataFrame(candidates)
    cand_df.to_csv(ROOT / "results/CLASSICAL_REFINEMENT_CANDIDATES.csv", index=False)

    results_store = []
    for _, c in cand_df.iterrows():
        tgt, name, est = c["target"], c["feature_set_name"], c["estimator"]
        X = g = blist = None
        for n, XX, gg, bb, rr in final_pool[tgt]:
            if n == name:
                X, g, blist = XX, gg, bb
                break
        try:
            res = eval_set(f"E__{name}__{est}", X, g, tgt, est, keep_oof=True)
        except Exception as e:
            print(f"FAILED {name} {est}: {e}", flush=True)
            continue
        results_store.append({
            "candidate_order": int(c["candidate_order"]), "target": tgt,
            "feature_set_name": name, "blocks": blist, "estimator": est,
            "X": X, "groups": g, "res": res,
        })

    pd.DataFrame(screen_rows).to_csv(ROOT / "results/CLASSICAL_REFINEMENT_CV_SCREEN.csv", index=False)
    ranked = []
    for tgt in ("TmApp", "HIC"):
        items = sorted(
            [r for r in results_store if r["target"] == tgt],
            key=lambda r: (r["res"]["cv_worst_mae"], r["res"]["cv_mean_mae"]),
        )
        for i, it in enumerate(items):
            ranked.append({
                "candidate_order": it["candidate_order"], "target": tgt,
                "feature_set_name": it["feature_set_name"], "estimator": it["estimator"],
                "rank_cv_worst": i + 1,
                "cv_primary_mae": it["res"]["cv_primary_mae"],
                "cv_shadow_mae": it["res"]["cv_shadow_mae"],
                "cv_worst_mae": it["res"]["cv_worst_mae"],
                "blocks": "|".join(it["blocks"]),
            })
    pd.DataFrame(ranked).to_csv(ROOT / "results/CLASSICAL_REFINEMENT_CV_RANKING.csv", index=False)

    freeze = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_metric": "cv_worst=max(primary,shadow); tie_break=cv_mean",
        "public_private_used_in_selection": False,
        "preprocessing_order": "per-block impute -> PCA32(if AbLingua/new PLM block) -> StandardScaler -> estimator",
        "candidates_path": "results/CLASSICAL_REFINEMENT_CANDIDATES.csv",
        "ranking_path": "results/CLASSICAL_REFINEMENT_CV_RANKING.csv",
        "n_results": len(results_store),
        "top_by_target": {
            tgt: [
                {
                    "feature_set_name": r["feature_set_name"], "estimator": r["estimator"],
                    "cv_worst_mae": float(r["res"]["cv_worst_mae"]),
                    "cv_primary_mae": float(r["res"]["cv_primary_mae"]),
                    "cv_shadow_mae": float(r["res"]["cv_shadow_mae"]),
                }
                for r in sorted(
                    [x for x in results_store if x["target"] == tgt],
                    key=lambda z: z["res"]["cv_worst_mae"],
                )[:5]
            ]
            for tgt in ("TmApp", "HIC")
        },
        "git_head_at_start": "37e47225",
    }
    (ROOT / "results/CLASSICAL_REFINEMENT_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )
    print("\nFROZE candidates", flush=True)

    print("\n=== POST-FREEZE Test ===", flush=True)
    sol = pd.read_csv(BUNDLE / "solution.csv")
    sol["id"] = sol["id"].astype(str)
    pub_ids = set(sol.loc[sol["is_public"].astype(bool), "id"])
    priv_ids = set(sol.loc[sol["is_private"].astype(bool), "id"])
    test_ids = test["id"].astype(str).tolist()

    exp_path = ROOT / "results/experiments.csv"
    exp_df = pd.read_csv(exp_path)
    snap77 = pd.read_csv(ROOT / "results/_preservation_snapshot_77_classical.csv")
    assert len(exp_df) == 77
    fs_df = pd.read_csv(ROOT / "results/FEATURE_SETS.csv")
    new_rows = []
    rasa_cdr_rows = []

    def make_eid(tgt, est, name: str) -> str:
        pref = {"ridge": "LIN", "lasso": "LIN", "enet": "ENET", "svr": "SVR", "xgb": "XGB"}[est]
        tag = name.replace("FS_", "").replace("FB_", "").replace("+", "_").replace("__", "_")
        tag = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in tag)
        tag = "_".join(t for t in tag.split("_") if t)[:70]
        mid = "TM" if tgt == "TmApp" else "HIC"
        est2 = {"ridge": "RIDGE", "lasso": "LASSO", "enet": "ENET", "svr": "SVR", "xgb": "XGB"}[est]
        return f"{pref}_{mid}_{tag}_{est2}"[:120]

    for it in results_store:
        tgt, est, name = it["target"], it["estimator"], it["feature_set_name"]
        X, g, res = it["X"], it["groups"], it["res"]
        eid = make_eid(tgt, est, name)
        existing = set(load_codes()["experiment_id"]) | {r["experiment_id"] for r in new_rows}
        base_eid, n = eid, 2
        while eid in existing:
            eid = f"{base_eid}_{n}"
            n += 1
        code = issue_code(
            eid, tgt, source_model_id=f"CLASSICAL::{name}::{est}",
            phase="CLASSICAL_REFINEMENT", notes="classical feature refinement; no ensemble",
        )
        feat = add_split_column(X.copy(), dev["id"].astype(str), test["id"].astype(str))
        fpath = ROOT / "experiments/features" / f"{code}.parquet"
        feat.to_parquet(fpath, index=False)
        fsha, csha = file_sha256(fpath), feature_content_sha256(feat)
        pred_dir = ROOT / "experiments/predictions" / code
        pred_dir.mkdir(parents=True, exist_ok=True)
        op = res["oof_primary"].reset_index(); op.columns = ["id", tgt]
        os_ = res["oof_shadow"].reset_index(); os_.columns = ["id", tgt]
        op.to_csv(pred_dir / "oof_primary.csv", index=False)
        os_.to_csv(pred_dir / "oof_shadow.csv", index=False)
        te_pred = fit_full_dev_predict(X, ymaps[tgt], primary, test_ids, estimator=est, groups=g)
        te_df = pd.DataFrame({"id": test_ids, tgt: te_pred})
        te_df.to_csv(pred_dir / "test.csv", index=False)
        te_s = te_df.set_index("id")[tgt]
        pub = [i for i in test_ids if i in pub_ids]
        priv = [i for i in test_ids if i in priv_ids]
        pub_mae = float(np.mean(np.abs(sol.set_index("id").loc[pub, tgt] - te_s.loc[pub])))
        priv_mae = float(np.mean(np.abs(sol.set_index("id").loc[priv, tgt] - te_s.loc[priv])))
        overall = float(np.mean(np.abs(sol.set_index("id").loc[test_ids, tgt].to_numpy(float) - te_pred)))

        cfg = {
            "experiment_code": code, "experiment_id": eid, "target": tgt,
            "family": "XGBOOST" if est == "xgb" else "LINEAR", "model_type": est.upper(),
            "feature_set_name": name, "blocks": it["blocks"],
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "ensemble": False,
            "pca_groups": [{"n": len(c), "pca": p} for c, p in g],
        }
        (ROOT / "experiments/configs" / f"{code}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        fsid = f"FS_{code}"
        fs_df = pd.concat([fs_df, pd.DataFrame([{
            "feature_set_id": fsid, "target": tgt, "n_features": feat.shape[1] - 2,
            "feature_content_sha256": csha, "feature_recipe_hash": "",
            "source_recipe_ids": name, "feature_blocks": "|".join(it["blocks"]),
            "notes": "classical refinement",
        }])], ignore_index=True)

        row = {c: "" for c in exp_df.columns}
        row.update({
            "experiment_code": code, "legacy_experiment_code": "", "experiment_id": eid,
            "target": tgt, "family": cfg["family"], "model_type": est.upper(),
            "source_model_id": f"CLASSICAL::{name}::{est}", "feature_set_id": fsid,
            "source_recipe_id": name,
            "cv_primary_mae": res["cv_primary_mae"], "cv_shadow_mae": res["cv_shadow_mae"],
            "cv_mean_mae": res["cv_mean_mae"], "cv_worst_mae": res["cv_worst_mae"],
            "public_mae": pub_mae, "private_mae": priv_mae, "test_overall_mae": overall,
            "public_private_delta": pub_mae - priv_mae, "public_private_gap": abs(pub_mae - priv_mae),
            "cv_protocol": "canonical_simple_tvt_primary_shadow",
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "artifact_status": "FULL", "source_reproducible": "YES",
            "drilldown_reproducible": "YES", "reproduction_status": "PASS",
            "config_path": f"experiments/configs/{code}.yaml",
            "feature_path": f"experiments/features/{code}.parquet",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "n_features": feat.shape[1] - 2, "feature_space": "RAW_PREPROCESS",
            "feature_sha256": fsha, "feature_content_sha256": csha,
            "score_source": "classical_refinement_cv+solution_postfreeze",
            "prediction_source": "classical_refinement", "feature_source": name,
            "license_status": "REVIEW",
            "license_reference": "PLM residue / structure-derived classical features",
            "ensemble_type": "", "member_experiment_codes": "",
            "notes": "CLASSICAL_FEATURE_REFINEMENT; no ensemble; CV-selected then postcomp evaluated",
        })
        new_rows.append(row)
        if any(x in name for x in ("RASA", "CDRW", "CDR3W", "CDR_ALL", "CDR3", "ARO_RASA")):
            rasa_cdr_rows.append({
                "target": tgt, "source_embedding": name, "feature_set_id": fsid,
                "estimator": est, "experiment_code": code,
                "cv_primary": res["cv_primary_mae"], "cv_shadow": res["cv_shadow_mae"],
                "cv_worst": res["cv_worst_mae"], "public": pub_mae, "private": priv_mae,
                "overall": overall,
            })
        print(f"REGISTERED {code} {eid} W={res['cv_worst_mae']:.4f} Pub={pub_mae:.4f} Priv={priv_mae:.4f}", flush=True)

    new_df = pd.DataFrame(new_rows)
    for c in exp_df.columns:
        if c not in new_df.columns:
            new_df[c] = ""
    out_exp = pd.concat([exp_df, new_df[exp_df.columns]], ignore_index=True)
    out_exp.to_csv(exp_path, index=False)
    fs_df.to_csv(ROOT / "results/FEATURE_SETS.csv", index=False)
    pd.DataFrame(rasa_cdr_rows).to_csv(ROOT / "results/RASA_CDR_WEIGHTING_RESULTS.csv", index=False)

    cur77 = out_exp[out_exp["experiment_code"].isin(snap77["experiment_code"])]
    m = snap77.merge(cur77, on="experiment_code", suffixes=("_old", "_new"))
    assert len(m) == 77
    for c in ("cv_primary_mae", "private_mae", "public_mae", "cv_shadow_mae", "cv_worst_mae"):
        d = (pd.to_numeric(m[f"{c}_old"], errors="coerce") - pd.to_numeric(m[f"{c}_new"], errors="coerce")).abs().max()
        assert float(d) == 0.0 or pd.isna(d), d

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; added {len(new_rows)}; total={len(out_exp)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
