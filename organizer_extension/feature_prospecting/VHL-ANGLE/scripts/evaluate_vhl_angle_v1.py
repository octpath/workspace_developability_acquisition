#!/usr/bin/env python3
"""VHL-ANGLE_v1 target evaluation using shared Gate2A ridge_eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
sys.path.insert(0, str(ROOT / "organizer_extension/feature_prospecting"))

from common.ridge_eval import run_family_evaluation  # noqa: E402

FAMILY = ROOT / "organizer_extension/feature_prospecting/VHL-ANGLE"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]


def main():
    tmapp, hic, ok = run_family_evaluation(FAMILY, "VHL-ANGLE", "v1", CANON, univariate_features=CANON)
    print("generator_signal TmApp", tmapp["_generator_signal"])
    print("generator_signal HIC", hic["_generator_signal"])
    for g in ["esmfold", "abodybuilder2", "boltz2"]:
        m = tmapp[g]
        print(
            g,
            m["standalone_signal"],
            m["incremental_signal"],
            m["empirical_verdict"],
            {k: round(m[k], 4) for k in ["MAE_CV", "MAE_Public", "MAE_Private", "delta_MAE_CV", "delta_MAE_Public", "delta_MAE_Private"]},
        )


if __name__ == "__main__":
    main()
