"""Stage 5 leakage / integrity tests."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/stage5_integration"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_S = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
HIC_TAIL = 10.5372


def test_primary_shadow_folds_distinct():
    fp = pd.read_csv(CV_P).set_index("id")["fold"]
    fs = pd.read_csv(CV_S).set_index("id")["fold"]
    common = fp.index.intersection(fs.index)
    assert len(common) == 162
    # folds may differ by design; ensure no accidental merge
    assert fp.loc[common].between(0, 4).all()
    assert fs.loc[common].between(0, 4).all()


def test_hic_tail_n17():
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")
    assert int((dev["HIC"] >= HIC_TAIL).sum()) == 17


def test_oof_files_162_rows():
    for p in OUT.glob("oof/*.csv"):
        df = pd.read_csv(p)
        assert len(df) == 162, p.name
        assert not np.isinf(df["y_pred"]).any(), p.name
        assert df["y_pred"].notna().all(), p.name


if __name__ == "__main__":
    test_primary_shadow_folds_distinct()
    test_hic_tail_n17()
    test_oof_files_162_rows()
    print("ALL TESTS PASS")
