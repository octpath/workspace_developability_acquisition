#!/usr/bin/env python3
"""PKA-SHIFT_v1 target evaluation via shared ridge_eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
sys.path.insert(0, str(ROOT / "organizer_extension/feature_prospecting"))
from common.ridge_eval import run_family_evaluation  # noqa: E402

FAMILY = ROOT / "organizer_extension/feature_prospecting/PKA-SHIFT"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
# emphasize pre-specified physical summaries
UNI = [
    "mean_abs_delta_pKa",
    "max_abs_delta_pKa",
    "rms_delta_pKa",
    "cdr_mean_abs_delta",
    "cdr_max_abs_delta",
    "interface_mean_abs_delta",
    "interface_max_abs_delta",
    "buried_mean_abs_delta",
    "buried_max_abs_delta",
    "acidic_mean_abs_delta",
    "basic_mean_abs_delta",
    "frac_abs_delta_ge_1",
    "frac_abs_delta_ge_2",
]


def main():
    tmapp, hic, ok = run_family_evaluation(
        FAMILY, "PKA-SHIFT", "v1", CANON, univariate_features=UNI
    )
    print("AUDIT", ok)
    print("generator_signal TmApp", tmapp["_generator_signal"])
    print("generator_signal HIC", hic["_generator_signal"])
    for g in ["esmfold", "abodybuilder2", "boltz2"]:
        m = tmapp[g]
        print(
            "TmApp",
            g,
            m["standalone_signal"],
            m["incremental_signal"],
            m["empirical_verdict"],
            {k: round(m[k], 4) for k in ["MAE_CV", "MAE_Public", "MAE_Private", "delta_MAE_CV", "delta_MAE_Public", "delta_MAE_Private"]},
        )
    for g in ["esmfold", "abodybuilder2", "boltz2"]:
        m = hic[g]
        print(
            "HIC",
            g,
            m["standalone_signal"],
            m["incremental_signal"],
            m["empirical_verdict"],
            {k: round(m[k], 4) for k in ["MAE_CV", "MAE_Public", "MAE_Private", "delta_MAE_CV", "delta_MAE_Public", "delta_MAE_Private"]},
        )


if __name__ == "__main__":
    main()
