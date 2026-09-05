#!/usr/bin/env python3
"""Select 12 target-blind Abs for BioEmu context convergence (no TmApp/HIC)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/bioemu_constant_context"
MAN = CTX / "BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv"
OUT = CTX / "pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv"
SEED = 20260905


def main():
    m = pd.read_csv(MAN)
    # germline families from gene prefixes
    def vfam(g):
        g = str(g)
        if g.startswith("IGHV") or g.startswith("IGKV") or g.startswith("IGLV"):
            parts = g.replace("*", "-").split("-")
            return "-".join(parts[:2]) if len(parts) >= 2 else g[:6]
        return g[:8] if g and g != "nan" else "UNK"

    m["heavy_v_family"] = m["heavy_V_gene"].map(vfam)
    m["light_v_family"] = m["light_V_gene"].map(vfam)
    m["len_total"] = m["heavy_length"] + m["light_length"]

    rng = np.random.default_rng(SEED)
    selected = []
    # Stratify: kappa/lambda × length quartiles; prefer germline diversity
    for locus in ("kappa", "lambda"):
        sub = m[m.light_locus == locus].copy()
        # take 8 kappa, 4 lambda (~cohort ratio 238:86)
        n_take = 8 if locus == "kappa" else 4
        qs = pd.qcut(sub.len_total, q=min(4, len(sub)), duplicates="drop")
        sub = sub.assign(len_q=qs)
        picks = []
        for q, g in sub.groupby("len_q", observed=True):
            # one from each quartile when possible
            # diversity by light V family
            fams = list(g.light_v_family.unique())
            rng.shuffle(fams)
            for fam in fams:
                cand = g[g.light_v_family == fam]
                picks.append(cand.sample(1, random_state=int(rng.integers(1e9))).iloc[0])
                break
        # fill remaining
        leftover = sub[~sub.id.isin([p.id for p in picks])]
        while len(picks) < n_take and len(leftover):
            # maximize new light families
            used = {p.light_v_family for p in picks}
            prefer = leftover[~leftover.light_v_family.isin(used)]
            pool = prefer if len(prefer) else leftover
            row = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            picks.append(row)
            leftover = leftover[leftover.id != row.id]
        selected.extend(picks[:n_take])

    out = pd.DataFrame(selected)[
        [
            "id",
            "light_locus",
            "len_total",
            "heavy_length",
            "light_length",
            "heavy_v_family",
            "light_v_family",
            "VH_len",
            "VL_len",
            "CH1_len",
            "CL_len",
        ]
    ].copy()
    out["selection_rule"] = f"target_blind_kappa_lambda_length_germline_seed{SEED}"
    assert len(out) == 12
    assert out.id.is_unique
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
