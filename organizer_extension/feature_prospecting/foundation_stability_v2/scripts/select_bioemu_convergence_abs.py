#!/usr/bin/env python3
"""Select 12 target-blind Abs for BioEmu sample-count convergence (no TmApp)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
V2 = ROOT / "organizer_extension/feature_prospecting/foundation_stability_v2"
OUT = V2 / "pilots/BIOEMU_CONVERGENCE_ANTIBODIES.csv"


def main():
    # All 324 ids from annotations (Dev+test) — sequences without using TmApp for selection
    ann = pd.read_csv(ROOT / "competition/organizer/frozen/sequence_derived_annotations_324.csv")
    seq = pd.concat(
        [
            pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]],
            # test features may lack targets in another file — use annotations join
        ]
    )
    # test sequences from test_features or secret? Use frozen annotations + reconstruct lengths from cdr file
    # Prefer: structure crosswalk ids + sequences from a combined source
    tf = ROOT / "competition/data/distribution/test_features.csv"
    if tf.exists():
        tdf = pd.read_csv(tf)
        cols = [c for c in tdf.columns if c.lower() in ("heavy", "light", "vh", "vl", "h_sequence", "l_sequence")]
        print("test_features cols sample", tdf.columns[:20].tolist())
    # Build from CDR index unique chains lengths
    cdr = pd.read_csv(ROOT / "organizer_extension/feature_prospecting/cdr_sequence_index_imgt.csv")
    lens = cdr.groupby(["id", "chain"]).size().unstack(fill_value=0)
    lens = lens.rename(columns={"H": "len_H", "L": "len_L"})
    df = ann.merge(lens, left_on="id", right_index=True, how="left")
    df["len_total"] = df.get("len_H", 0) + df.get("len_L", 0)
    # diversity bins without TmApp
    df = df.dropna(subset=["len_total"]).copy()
    df["len_q"] = pd.qcut(df["len_total"], 4, labels=False, duplicates="drop")
    germ_h = df.get("heavy_v_family", pd.Series(["NA"] * len(df)))
    # pick evenly: 3 per length quartile spanning germline
    picks = []
    rng = np.random.default_rng(20260904)
    for q in sorted(df["len_q"].dropna().unique()):
        sub = df[df["len_q"] == q]
        # stratified by heavy_v_family
        fams = sub["heavy_v_family"].fillna("NA").unique()
        rng.shuffle(fams)
        chosen = []
        for f in fams:
            cand = sub[sub["heavy_v_family"].fillna("NA") == f]
            chosen.append(cand.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0])
            if len(chosen) >= 3:
                break
        while len(chosen) < 3 and len(sub) > len(chosen):
            rest = sub[~sub.id.isin([c.id for c in chosen])]
            chosen.append(rest.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0])
        picks.extend(chosen[:3])
    out = pd.DataFrame(picks)[["id", "len_total", "len_H", "len_L", "heavy_v_family", "light_v_family", "light_chain_type"]]
    out["selection_rule"] = "target_blind_length_quartile_x_germline_seed20260904"
    assert len(out) == 12, len(out)
    assert "TmApp" not in out.columns
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
