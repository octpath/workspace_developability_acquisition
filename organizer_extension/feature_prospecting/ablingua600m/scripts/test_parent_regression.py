#!/usr/bin/env python3
"""Regression: SEQ_BASIC must align to competition ids (not RangeIndex)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
sys.path.insert(0, str(OUT / "scripts"))

import runpy

ev = runpy.run_path(str(OUT / "scripts/02_simple_tvt_eval.py"))
ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
seq = ev["load_seq_basic"](ids)
assert list(seq.index) == ids, "SEQ_BASIC index must equal competition ids"
assert seq.shape[0] == 162
assert seq.isna().mean().mean() < 0.05, f"SEQ_BASIC too many NaNs: {seq.isna().mean().mean()}"
# authoritative naming from load_bases
assert all(str(c).startswith("seqB_") for c in seq.columns)
print("assert_same_parent_seq_basic_alignment: OK", seq.shape)


def assert_same_parent_predictions():
    """PARENT MAE must match frozen authoritative CURRENT_RECIPE+AbLingua GLOBAL."""
    import importlib.util

    INTERIM = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit"
    spec = importlib.util.spec_from_file_location(
        "tvt", INTERIM / "scripts/run_simple_tvt_rescreen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evB = runpy.run_path(str(OUT / "scripts/04_guided_pooling_tvt.py"))
    recipe, HL = evB["load_recipe_and_global"](ids)
    y = mod.load_y("TmApp")
    primary = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    shadow = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_shadow.csv")
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)
    # Use fixed 02 path after SEQ fix
    out_p = ev["run_incr"](recipe, HL, y, primary, ids, "FREE_ALPHA", True)
    out_s = ev["run_incr"](recipe, HL, y, shadow, ids, "FREE_ALPHA", True)
    # Authoritative from reconciliation / guided sprint B
    exp_p, exp_s = 2.7466001170280285, 2.822725206703513
    assert abs(out_p["plus_mae"] - exp_p) < 1e-6, (out_p["plus_mae"], exp_p)
    assert abs(out_s["plus_mae"] - exp_s) < 1e-6, (out_s["plus_mae"], exp_s)
    print("assert_same_parent_predictions: OK", out_p["plus_mae"], out_s["plus_mae"])


if __name__ == "__main__":
    assert_same_parent_predictions()
