#!/usr/bin/env python3
"""Reproduce frozen organizer Top-3 feature-level Simple TVT recipes from this bundle only.

Examples:
  python reproduce_top_recipes.py --dev dev.csv --test test.csv
  python reproduce_top_recipes.py --dev dev.csv --test test.csv --solution solution.csv
  python reproduce_top_recipes.py --dev dev.csv --test test.csv \\
      --tm-recipe TM_PARENT_ABLINGUA_CDR3__RIDGE \\
      --hic-recipe HIC_HYDRO_TITRATION__LASSO \\
      --submission outputs/submission.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bundle_simple_tvt import (
    FeatureAlignmentError,
    align_feature_block,
    concat_blocks,
    fit_predict_lasso,
    fit_predict_ridge_abl_blocks,
    fit_predict_ridge_base_struct,
    fit_predict_ridge_raw,
    mae,
    run_base_plus_struct_ridge,
    run_recipe_plus_abl_blocks_ridge,
    run_standalone_lasso,
    run_standalone_ridge,
    sha_arr,
)

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "feature_manifest.csv"
RECIPES = ROOT / "recipes.csv"
FOLDS = ROOT / "folds.csv"
FREEZE_CANDIDATES = [
    ROOT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json",
]
# Bundle-local copy of final alphas (participant release includes this file)
ALPHA_POLICY = ROOT / "FULL_DEV_ALPHA_POLICY_BUNDLE.json"
EXPECTED = ROOT / "EXPECTED_SCORES.json"

# Tight tolerances for deterministic sklearn reproduction
TOL_EXACT = 1e-10
TOL_EQUIV = 1e-8

BLOCK_FILE = {
    "AROMATIC_TOPO": "data/aromatic_topo.parquet",
    "AbLang2_HL_paired": "data/ablang2.parquet",
    "AbLingua_CDR3": "data/ablingua_cdr3.parquet",
    "AbLingua_HL_mean": "data/ablingua_global.parquet",
    "BIOEMU_NEW_PAIRWISE": "data/bioemu_new_pairwise.parquet",
    "CONTINUOUS_SURFACE": "data/continuous_surface.parquet",
    "ESM2_H": "data/esm2_heavy.parquet",
    "HYDRO_FIELD": "data/hydro_field.parquet",
    "M1_PROTEINMPNN": "data/proteinmpnn.parquet",
    "SEQ_ALL": "data/seq_all.parquet",
    "SEQ_BASIC": "data/seq_basic.parquet",
    "TITRATION_SHAPE": "data/titration_shape.parquet",
}


class SolutionLeakError(RuntimeError):
    """Raised if solution labels are accessed during CV / training."""


class GuardedSolution:
    def __init__(self, df: pd.DataFrame | None):
        self._df = df
        self._allow_score = False
        self.accessed_during_train = False

    def enable_scoring(self):
        self._allow_score = True

    def dataframe_for_scoring(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("no solution")
        if not self._allow_score:
            self.accessed_during_train = True
            raise SolutionLeakError("solution.csv accessed before scoring phase")
        return self._df


def load_block(name: str, ids: list[str], root: Path) -> pd.DataFrame:
    rel = BLOCK_FILE[name]
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"missing feature block file: {path}")
    raw = pd.read_parquet(path)
    return align_feature_block(raw, ids, name)


def recipe_base_id(recipe_id: str) -> str:
    # TM_PARENT_ABLINGUA_CDR3__RIDGE -> TM_PARENT_ABLINGUA_CDR3
    if recipe_id.endswith("__RIDGE") or recipe_id.endswith("__LASSO"):
        return recipe_id.rsplit("__", 1)[0]
    return recipe_id


def verify_freeze(recipes: pd.DataFrame) -> None:
    freeze_path = next((p for p in FREEZE_CANDIDATES if p.exists()), None)
    if freeze_path is None:
        print("WARN: ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json not found; skip freeze check")
        return
    freeze = json.loads(freeze_path.read_text())
    for target in ("TmApp", "HIC"):
        fids = [x["model_id"] for x in freeze["targets"][target]]
        rids = (
            recipes[recipes.target == target]
            .sort_values("cv_rank")
            .recipe_id.tolist()
        )
        if fids != rids:
            raise SystemExit(f"recipes.csv != freeze for {target}: {rids} vs {fids}")
        for x, (_, row) in zip(freeze["targets"][target], recipes[recipes.target == target].sort_values("cv_rank").iterrows()):
            if x["recipe_tokens"] != row.feature_blocks:
                raise SystemExit(f"block mismatch {row.recipe_id}")


def load_alpha(recipe_id: str, regressor: str) -> float:
    pol = json.loads(ALPHA_POLICY.read_text())
    key = f"{recipe_base_id(recipe_id)}::{regressor}"
    if key not in pol["final_alpha_by_recipe_regressor"]:
        raise KeyError(f"missing alpha policy for {key}")
    a = pol["final_alpha_by_recipe_regressor"][key]
    if a is None:
        raise ValueError(f"null alpha for {key}")
    return float(a)


def status_for(diff: float) -> str:
    ad = abs(diff)
    if ad <= TOL_EXACT:
        return "EXACT_REPRODUCTION"
    if ad <= TOL_EQUIV:
        return "NUMERICALLY_EQUIVALENT"
    return "MISMATCH"


def build_parts(blocks: list[str], ids: list[str], root: Path, recipe_id: str):
    """Return mode + aligned matrices for a recipe."""
    rid = recipe_base_id(recipe_id)
    if rid == "TM_PARENT_ABLINGUA_GLOBAL":
        core = [b for b in blocks if b != "AbLingua_HL_mean"]
        base = concat_blocks([load_block(b, ids, root) for b in core], ids, core)
        struct = load_block("AbLingua_HL_mean", ids, root)
        return {"mode": "base_struct", "base": base, "struct": struct, "ids": ids}
    if rid == "TM_PARENT_ABLINGUA_CDR3":
        core = [b for b in blocks if b not in ("AbLingua_HL_mean", "AbLingua_CDR3")]
        recipe = concat_blocks([load_block(b, ids, root) for b in core], ids, core)
        always = load_block("AbLingua_HL_mean", ids, root)
        extra = load_block("AbLingua_CDR3", ids, root)
        return {
            "mode": "abl_blocks",
            "recipe": recipe,
            "always": always,
            "extra": extra,
            "ids": ids,
        }
    X = concat_blocks([load_block(b, ids, root) for b in blocks], ids, blocks)
    return {"mode": "standalone", "X": X, "ids": ids}


def run_cv(parts, y: pd.Series, folds: pd.DataFrame, regressor: str) -> dict:
    if regressor == "RIDGE":
        if parts["mode"] == "standalone":
            return run_standalone_ridge(parts["X"], y, folds, parts["ids"])
        if parts["mode"] == "base_struct":
            return run_base_plus_struct_ridge(
                parts["base"], parts["struct"], y, folds, parts["ids"]
            )
        return run_recipe_plus_abl_blocks_ridge(
            parts["recipe"],
            [parts["always"]],
            [parts["extra"]],
            y,
            folds,
            parts["ids"],
        )
    if regressor == "LASSO":
        if parts["mode"] == "standalone":
            X = parts["X"]
        elif parts["mode"] == "base_struct":
            X = pd.concat([parts["base"], parts["struct"]], axis=1)
        else:
            X = pd.concat([parts["recipe"], parts["always"], parts["extra"]], axis=1)
        return run_standalone_lasso(X, y, folds, parts["ids"])
    raise ValueError(regressor)


def predict_test(parts_dev, parts_te, y_dev, regressor: str, alpha: float) -> pd.Series:
    if regressor == "RIDGE":
        if parts_dev["mode"] == "standalone":
            return fit_predict_ridge_raw(parts_dev["X"], y_dev, parts_te["X"], alpha)
        if parts_dev["mode"] == "base_struct":
            return fit_predict_ridge_base_struct(
                parts_dev["base"],
                parts_dev["struct"],
                y_dev,
                parts_te["base"],
                parts_te["struct"],
                alpha,
            )
        return fit_predict_ridge_abl_blocks(
            parts_dev["recipe"],
            parts_dev["always"],
            parts_dev["extra"],
            y_dev,
            parts_te["recipe"],
            parts_te["always"],
            parts_te["extra"],
            alpha,
        )
    if parts_dev["mode"] == "standalone":
        Xd, Xt = parts_dev["X"], parts_te["X"]
    elif parts_dev["mode"] == "base_struct":
        Xd = pd.concat([parts_dev["base"], parts_dev["struct"]], axis=1)
        Xt = pd.concat([parts_te["base"], parts_te["struct"]], axis=1)
    else:
        Xd = pd.concat([parts_dev["recipe"], parts_dev["always"], parts_dev["extra"]], axis=1)
        Xt = pd.concat([parts_te["recipe"], parts_te["always"], parts_te["extra"]], axis=1)
    return fit_predict_lasso(Xd, y_dev, Xt, alpha)


def validate_tables(dev: pd.DataFrame, test: pd.DataFrame, solution: pd.DataFrame | None):
    for name, df, need in [
        ("dev", dev, {"id", "heavy", "light", "TmApp", "HIC"}),
        ("test", test, {"id", "heavy", "light"}),
    ]:
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"{name}.csv missing columns {miss}")
    if len(dev) != 162 or len(test) != 162:
        raise SystemExit(f"expected N=162; got dev={len(dev)} test={len(test)}")
    if set(dev.id.astype(str)) & set(test.id.astype(str)):
        raise SystemExit("dev/test IDs not disjoint")
    if solution is not None:
        need = {"id", "heavy", "light", "TmApp", "HIC", "is_public", "is_private"}
        miss = need - set(solution.columns)
        if miss:
            raise SystemExit(f"solution.csv missing {miss}")
        if len(solution) != 162:
            raise SystemExit(f"solution N={len(solution)}")
        if set(solution.id.astype(str)) != set(test.id.astype(str)):
            raise SystemExit("solution IDs != test IDs")
        pub = solution.is_public.astype(bool)
        priv = solution.is_private.astype(bool)
        if int(pub.sum()) != 81 or int(priv.sum()) != 81:
            raise SystemExit(f"public/private N={pub.sum()}/{priv.sum()}")
        if not (pub ^ priv).all():
            raise SystemExit("is_public XOR is_private failed")


def score_pp(pred: pd.Series, solution: pd.DataFrame, target: str) -> dict:
    sol = solution.set_index("id")
    common = [i for i in pred.index if i in sol.index]
    y = sol.loc[common, target].astype(float)
    p = pred.loc[common]
    pub = [i for i in common if bool(sol.loc[i, "is_public"])]
    priv = [i for i in common if bool(sol.loc[i, "is_private"])]
    return {
        "public_mae": mae(y.loc[pub], p.loc[pub]),
        "private_mae": mae(y.loc[priv], p.loc[priv]),
        "overall_test_mae": mae(y, p),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default="dev.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--solution", default=None, help="OPTIONAL internal solution.csv")
    ap.add_argument("--target", choices=["TmApp", "HIC"], action="append", default=None)
    ap.add_argument("--recipe", action="append", default=None)
    ap.add_argument("--all", action="store_true", default=True)
    ap.add_argument("--tm-recipe", default=None)
    ap.add_argument("--hic-recipe", default=None)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--bundle-root", default=str(ROOT))
    args = ap.parse_args(argv)

    root = Path(args.bundle_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Prefer local paths relative to CWD, then bundle root
    def resolve_csv(p: str) -> Path:
        cand = Path(p)
        if cand.exists():
            return cand.resolve()
        alt = root / p
        if alt.exists():
            return alt.resolve()
        raise SystemExit(f"missing {p}")

    dev = pd.read_csv(resolve_csv(args.dev))
    test = pd.read_csv(resolve_csv(args.test))
    sol_df = pd.read_csv(resolve_csv(args.solution)) if args.solution else None
    validate_tables(dev, test, sol_df)
    guarded = GuardedSolution(sol_df)

    recipes = pd.read_csv(root / "recipes.csv")
    verify_freeze(recipes)
    expected = json.loads((root / "EXPECTED_SCORES.json").read_text())

    # Select recipes (--tm-recipe / --hic-recipe only pick submission pair; CV still all unless --recipe)
    sel = recipes.copy()
    if args.target:
        sel = sel[sel.target.isin(args.target)]
    if args.recipe:
        sel = sel[sel.recipe_id.isin(args.recipe)]
    sel = sel.drop_duplicates("recipe_id").sort_values(["target", "cv_rank"])

    folds_all = pd.read_csv(root / "folds.csv")
    folds_all["id"] = folds_all["id"].astype(str)
    dev["id"] = dev["id"].astype(str)
    test["id"] = test["id"].astype(str)
    dev_ids = dev["id"].tolist()
    test_ids = test["id"].tolist()

    # Hard assert: SEQ_BASIC alignment sanity on DEV (catches historical bug class)
    seqb = load_block("SEQ_BASIC", dev_ids, root)
    if seqb.isna().all().all():
        raise FeatureAlignmentError("SEQ_BASIC all-NaN on DEV — abort")

    rows = []
    pred_cache: dict[str, pd.Series] = {}

    for _, row in sel.iterrows():
        rid = row.recipe_id
        target = row.target
        regressor = row.regressor
        blocks = [b for b in str(row.feature_blocks).split("|") if b]
        print(f"=== {rid} ({target}/{regressor}) ===", flush=True)

        y = dev.set_index("id")[target].astype(float)
        folds_p = folds_all[folds_all.scheme == "primary"][["id", "fold"]].copy()
        folds_s = folds_all[folds_all.scheme == "shadow"][["id", "fold"]].copy()

        # CV — must not touch solution
        parts_dev = build_parts(blocks, dev_ids, root, rid)
        rp = run_cv(parts_dev, y, folds_p, regressor)
        # rebuild for shadow (fresh parts ok)
        parts_dev_s = build_parts(blocks, dev_ids, root, rid)
        rs = run_cv(parts_dev_s, y, folds_s, regressor)

        exp = expected[rid]
        d_p = rp["mae"] - exp["expected_primary_mae"]
        d_s = rs["mae"] - exp["expected_shadow_mae"]
        st = status_for(max(abs(d_p), abs(d_s)))
        print(
            f"  Primary {rp['mae']:.10f} (exp {exp['expected_primary_mae']:.10f}) "
            f"Shadow {rs['mae']:.10f} (exp {exp['expected_shadow_mae']:.10f}) [{st}]",
            flush=True,
        )

        # Save CV OOF
        pd.DataFrame({"id": rp["oof"].index, "prediction": rp["oof"].values}).to_csv(
            outdir / f"{rid}__cv_predictions_primary.csv", index=False
        )
        pd.DataFrame({"id": rs["oof"].index, "prediction": rs["oof"].values}).to_csv(
            outdir / f"{rid}__cv_predictions_shadow.csv", index=False
        )

        # Full-DEV refit + Test predict
        alpha = load_alpha(rid, regressor)
        parts_te = build_parts(blocks, test_ids, root, rid)
        # Align part keys for modes
        if parts_dev["mode"] == "standalone":
            # ensure same columns
            pass
        pred = predict_test(parts_dev, parts_te, y, regressor, alpha)
        pred = pred.reindex(test_ids)
        if pred.isna().any():
            raise RuntimeError(f"NaN Test predictions for {rid}")
        pd.DataFrame({"id": pred.index, "prediction": pred.values}).to_csv(
            outdir / f"{rid}__test_predictions.csv", index=False
        )
        pred_cache[rid] = pred
        thash = sha_arr(pred.to_numpy(dtype=float))

        pub = priv = overall = np.nan
        if sol_df is not None:
            guarded.enable_scoring()
            sc = score_pp(pred, guarded.dataframe_for_scoring(), target)
            pub, priv, overall = sc["public_mae"], sc["private_mae"], sc["overall_test_mae"]
            print(
                f"  Public {pub:.6f} Private {priv:.6f} Overall {overall:.6f}",
                flush=True,
            )

        rows.append(
            {
                "target": target,
                "recipe_id": rid,
                "regressor": regressor,
                "n_dev": rp["N"],
                "primary_mae": rp["mae"],
                "shadow_mae": rs["mae"],
                "expected_primary_mae": exp["expected_primary_mae"],
                "expected_shadow_mae": exp["expected_shadow_mae"],
                "primary_abs_difference": abs(d_p),
                "shadow_abs_difference": abs(d_s),
                "cv_reproduction_status": st,
                "public_mae": pub,
                "private_mae": priv,
                "overall_test_mae": overall,
                "expected_public_mae": exp.get("expected_public_mae", np.nan),
                "expected_private_mae": exp.get("expected_private_mae", np.nan),
                "final_alpha": alpha,
                "test_prediction_hash": thash,
            }
        )

    scores = pd.DataFrame(rows)
    scores.to_csv(outdir / "reproduction_scores.csv", index=False)

    # Submission
    sub_ok = True
    if args.submission:
        tm = args.tm_recipe or (
            scores[scores.target == "TmApp"].sort_values("recipe_id").iloc[0].recipe_id
            if (scores.target == "TmApp").any()
            else None
        )
        hic = args.hic_recipe or (
            scores[scores.target == "HIC"].sort_values("recipe_id").iloc[0].recipe_id
            if (scores.target == "HIC").any()
            else None
        )
        if not tm or not hic:
            print("WARN: need both TmApp and HIC recipes for submission", flush=True)
            sub_ok = False
        else:
            sub = pd.DataFrame(
                {
                    "id": test_ids,
                    "TmApp": pred_cache[tm].reindex(test_ids).values,
                    "HIC": pred_cache[hic].reindex(test_ids).values,
                }
            )
            sub_path = Path(args.submission)
            if not sub_path.is_absolute():
                sub_path = Path.cwd() / sub_path
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            sub.to_csv(sub_path, index=False)
            if list(sub.columns) != ["id", "TmApp", "HIC"]:
                sub_ok = False
            print(f"Wrote submission {sub_path}", flush=True)

    # Audit markdown
    lines = ["# Bundle Reproduction Audit\n\n"]
    all_ok = (scores.cv_reproduction_status != "MISMATCH").all() and len(scores) >= 1
    if args.recipe is None and args.target is None:
        all_ok = all_ok and len(scores) == 6
    gate = "PASS" if all_ok and not guarded.accessed_during_train else "FAIL"
    lines.append(f"**BUNDLE_SELF_REPRODUCTION = {gate}**\n\n")
    lines.append("| target | recipe | expected P/S | reproduced P/S | status |\n")
    lines.append("|--------|--------|--------------|----------------|--------|\n")
    for _, r in scores.iterrows():
        lines.append(
            f"| {r.target} | `{r.recipe_id}` | "
            f"{r.expected_primary_mae:.6f}/{r.expected_shadow_mae:.6f} | "
            f"{r.primary_mae:.6f}/{r.shadow_mae:.6f} | {r.cv_reproduction_status} |\n"
        )
    if sol_df is not None:
        lines.append("\n## Public / Private (solution supplied)\n\n")
        lines.append("| recipe | Public | Private | expected Pub/Priv |\n")
        lines.append("|--------|--------|---------|-------------------|\n")
        for _, r in scores.iterrows():
            lines.append(
                f"| `{r.recipe_id}` | {r.public_mae:.6f} | {r.private_mae:.6f} | "
                f"{r.expected_public_mae}/{r.expected_private_mae} |\n"
            )
    lines.append(f"\nSubmission generation: {'PASS' if sub_ok else 'N/A or FAIL'}\n")
    lines.append(
        f"\nsolution accessed during CV/train: {guarded.accessed_during_train}\n"
    )
    audit_path = outdir / "BUNDLE_REPRODUCTION_AUDIT.md"
    # also write at bundle root when running from bundle
    audit_path.write_text("".join(lines))
    (root / "BUNDLE_REPRODUCTION_AUDIT.md").write_text("".join(lines))
    print(f"\nBUNDLE_SELF_REPRODUCTION = {gate}", flush=True)
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
