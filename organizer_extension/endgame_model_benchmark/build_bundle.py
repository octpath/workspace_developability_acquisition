#!/usr/bin/env python3
"""Build lightweight top_models_feature_bundle from CV Top-3 union."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/endgame_model_benchmark"
BUNDLE = ROOT / "top_models_feature_bundle"
SI = ROOT / "organizer_extension/feature_prospecting/score_integrity"
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(SI))
from feature_store import FeatureStore  # noqa: E402

# Known redistribution statuses from feature_extension RELEASE notes / MANIFEST
LICENSE = {
    "SEQ_BASIC": ("OK_COMPETITION_DERIVED", "Built from official competition sequences"),
    "SEQ_ALL": ("OK_COMPETITION_DERIVED", "Built from official competition sequences"),
    "AbLang2_HL_paired": ("REVIEW_MODEL_OUTPUT", "AbLang2 embeddings; code/weights != output — review redistribution"),
    "ESM2_H": ("REVIEW_MODEL_OUTPUT", "ESM2 embeddings; review redistribution"),
    "BIOEMU_NEW_PAIRWISE": ("SEE_FEATURE_EXTENSION", "Already in feature_extension bioemu bundle"),
    "M1_PROTEINMPNN": ("SEE_FEATURE_EXTENSION", "Already in feature_extension proteinmpnn.parquet"),
    "AbLingua_HL_mean": ("REVIEW_MODEL_OUTPUT", "AbLingua embeddings; HF weights + BioTokenizer — review"),
    "AbLingua_CDR3": ("REVIEW_MODEL_OUTPUT", "AbLingua guided pooling; review"),
    "AROMATIC_TOPO": ("SEE_FEATURE_EXTENSION", "aromatic_topology.parquet"),
    "CONTINUOUS_SURFACE": ("SEE_FEATURE_EXTENSION", "continuous_surface.parquet"),
    "TITRATION_SHAPE": ("SEE_FEATURE_EXTENSION", "titration_shape.parquet"),
    "HYDRO_FIELD": ("SEE_FEATURE_EXTENSION", "hydro_field.parquet"),
    "FENNIX_COMBINED_PREDECLARED": ("PARTICIPANT_ONLY_REVIEW_RECOMMENDED", "FeNNol LGPL-3.0; review hosting"),
    "OPENMM_DOMAIN_FLEXIBILITY": ("NOT_INCLUDED_NO_TEST", "DEV-only MD; not in Top-3 expected"),
}

FILE_MAP = {
    "SEQ_BASIC": "seq_basic.parquet",
    "SEQ_ALL": "seq_all.parquet",
    "AbLang2_HL_paired": "ablang2.parquet",
    "ESM2_H": "esm2_heavy.parquet",
    "BIOEMU_NEW_PAIRWISE": "bioemu_new_pairwise.parquet",
    "M1_PROTEINMPNN": "proteinmpnn.parquet",
    "AbLingua_HL_mean": "ablingua_global.parquet",
    "AbLingua_CDR3": "ablingua_cdr3.parquet",
    "AROMATIC_TOPO": "aromatic_topo.parquet",
    "CONTINUOUS_SURFACE": "continuous_surface.parquet",
    "TITRATION_SHAPE": "titration_shape.parquet",
    "HYDRO_FIELD": "hydro_field.parquet",
    "FENNIX_COMBINED_PREDECLARED": "fennix_combined.parquet",
}


def main():
    top = json.loads((OUT / "CV_SELECTED_RECIPES_FREEZE.json").read_text())
    post = pd.read_csv(OUT / "ORGANIZER_MODEL_POSTMORTEM.csv")
    cv = pd.read_csv(OUT / "ORGANIZER_ENDGAME_CV_TABLE.csv")

    # Union of blocks from Top-3
    needed = set()
    recipe_rows = []
    for target, lst in top["targets"].items():
        for p in lst:
            blocks = [b for b in str(p["feature_blocks"]).split("|") if b and b != "none"]
            needed.update(blocks)
            pr = post[
                (post.target == target)
                & (post.recipe_id == p["recipe_id"])
                & (post.regressor == p["regressor"])
            ]
            recipe_rows.append(
                {
                    "target": target,
                    "cv_rank": p["cv_rank"],
                    "recipe_id": p["recipe_id"],
                    "feature_blocks": p["feature_blocks"],
                    "regressor": p["regressor"],
                    "preprocessing": p["preprocessing"],
                    "cv_primary_mae": p["cv_primary_mae"],
                    "cv_shadow_mae": p["cv_shadow_mae"],
                    "public_mae": float(pr.public_mae.iloc[0]) if len(pr) else np.nan,
                    "private_mae": float(pr.private_mae.iloc[0]) if len(pr) else np.nan,
                    "comments": "cv_rank frozen BEFORE Public/Private",
                }
            )

    BUNDLE.mkdir(parents=True, exist_ok=True)
    data = BUNDLE / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir()
    (BUNDLE / "examples").mkdir(exist_ok=True)

    store = FeatureStore()
    # sequences
    seq = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    base = pd.concat([seq, te], ignore_index=True)
    base["id"] = base["id"].astype(str)
    base.to_parquet(data / "base_sequences.parquet", index=False)

    # folds
    prim = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    shad = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_shadow.csv")
    prim["scheme"] = "primary"
    shad["scheme"] = "shadow"
    folds = pd.concat([prim, shad], ignore_index=True)
    folds["id"] = folds["id"].astype(str)
    folds.to_csv(BUNDLE / "folds.csv", index=False)

    manifest = []
    blocked = []
    for block in sorted(needed):
        status, note = LICENSE.get(block, ("REVIEW", "unlisted"))
        fname = FILE_MAP.get(block)
        if fname is None:
            blocked.append(block)
            manifest.append(
                {
                    "block_name": block,
                    "file": "",
                    "n_ids": 0,
                    "n_features": 0,
                    "molecular_scope": "",
                    "source_model": block,
                    "model/version": "",
                    "feature_definition": note,
                    "target_used_in_generation": "NO",
                    "missing_ids": "",
                    "license_status": "NOT_INCLUDED_LICENSE_REVIEW",
                    "notes": note,
                }
            )
            continue
        X = store.get(block).copy()
        X = X.reset_index()
        if X.columns[0] != "id":
            X = X.rename(columns={X.columns[0]: "id"})
        X["id"] = X["id"].astype(str)
        # drop non-finite? keep documented missingness
        path = data / fname
        X.to_parquet(path, index=False)
        manifest.append(
            {
                "block_name": block,
                "file": f"data/{fname}",
                "n_ids": int(X.shape[0]),
                "n_features": int(X.shape[1] - 1),
                "molecular_scope": block,
                "source_model": block,
                "model/version": "cached_organizer",
                "feature_definition": note,
                "target_used_in_generation": "NO",
                "missing_ids": "",
                "license_status": status,
                "notes": "RAW features (no CV-fitted PCA)",
            }
        )

    pd.DataFrame(recipe_rows).to_csv(BUNDLE / "recipes.csv", index=False)
    pd.DataFrame(manifest).to_csv(BUNDLE / "feature_manifest.csv", index=False)

    readme = f"""# Organizer Top-Model Feature Bundle

## What this is

A compact set of **target-blind** features used by the strongest organizer-side Simple TVT CV recipes (Ridge/Lasso endgame).

## Files

See `feature_manifest.csv` for block → file mapping.

| file | role |
|------|------|
| `folds.csv` | fold_primary / fold_shadow |
| `recipes.csv` | Top-3 per target (CV rank frozen **before** Public/Private) |
| `data/base_sequences.parquet` | id, heavy, light |
| `data/*.parquet` | feature blocks |

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes

See `recipes.csv` (cv_rank 1–3 per target).

## Quick usage

```python
import pandas as pd
train = pd.read_csv("official_dev.csv")  # competition train table
feat = pd.read_parquet("data/ablang2.parquet")
train = train.merge(feat, on="id", how="left")
```

## Ridge / Lasso examples

- `examples/train_ridge.py`
- `examples/train_lasso.py`

## Important

- Fit scaler / PCA / imputation **inside training folds**
- Do **not** use Public/Private metadata for model selection
- Feature files are target-blind
- Some model-derived blocks may have redistribution conditions — see `license_status` in `feature_manifest.csv`
- Organizer Ridge may use PCA32 on AbLingua blocks; this bundle distributes **raw** embeddings

## Blocks in this release

{chr(10).join('- ' + b for b in sorted(needed))}
"""
    (BUNDLE / "README.md").write_text(readme)

    # examples
    (BUNDLE / "examples/load_features.py").write_text(
        '''#!/usr/bin/env python3
"""Load and merge feature blocks by id."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_block(name: str) -> pd.DataFrame:
    man = pd.read_csv(ROOT / "feature_manifest.csv")
    row = man.loc[man.block_name == name].iloc[0]
    if not row.file:
        raise FileNotFoundError(f"{name} not included: {row.license_status}")
    return pd.read_parquet(ROOT / row.file)

def merge_blocks(base: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    out = base.copy()
    out["id"] = out["id"].astype(str)
    for b in blocks:
        feat = load_block(b)
        feat["id"] = feat["id"].astype(str)
        out = out.merge(feat, on="id", how="left")
    return out

if __name__ == "__main__":
    seq = pd.read_parquet(ROOT / "data/base_sequences.parquet")
    print(merge_blocks(seq[["id"]], ["SEQ_BASIC"]).shape)
'''
    )
    (BUNDLE / "examples/train_ridge.py").write_text(
        '''#!/usr/bin/env python3
"""Minimal Ridge Simple-TVT-style example (no secret Test mask)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.1, 1.0, 10.0, 100.0]

def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))

def run(train_csv: str, target: str, feature_files: list[str], scheme: str = "primary"):
    train = pd.read_csv(train_csv)
    train["id"] = train["id"].astype(str)
    X = train[["id"]].copy()
    for f in feature_files:
        feat = pd.read_parquet(f)
        feat["id"] = feat["id"].astype(str)
        X = X.merge(feat, on="id", how="left")
    folds = pd.read_csv(ROOT / "folds.csv")
    folds = folds[folds.scheme == scheme]
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in X["id"] if i in fmap and i in set(train["id"])]
    y = train.set_index("id")[target].astype(float)
    num = X.set_index("id").select_dtypes("number")
    oof = pd.Series(index=ids, dtype=float)
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        med = num.loc[tr].median().fillna(0)
        def prep(ix):
            return num.loc[ix].fillna(med).fillna(0)
        best_a, best = 100.0, 1e18
        for a in ALPHAS:
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(prep(tr)), y.loc[tr])
            s = mae(y.loc[va], m.predict(sc.transform(prep(va))))
            if s < best:
                best, best_a = s, a
        tv = tr + va
        med2 = num.loc[tv].median().fillna(0)
        def prep2(ix):
            return num.loc[ix].fillna(med2).fillna(0)
        sc = StandardScaler()
        m = Ridge(alpha=best_a, random_state=0)
        m.fit(sc.fit_transform(prep2(tv)), y.loc[tv])
        oof.loc[te] = m.predict(sc.transform(prep2(te)))
    print("MAE", mae(y.loc[ids], oof))

if __name__ == "__main__":
    print("Provide official train CSV path as argv; see README.")
'''
    )
    (BUNDLE / "examples/train_lasso.py").write_text(
        '''#!/usr/bin/env python3
"""Minimal Lasso example: StandardScaler -> Lasso, VAL alpha selection, no PCA."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))

def run(train_csv: str, target: str, feature_files: list[str], scheme: str = "primary"):
    train = pd.read_csv(train_csv)
    train["id"] = train["id"].astype(str)
    X = train[["id"]].copy()
    for f in feature_files:
        feat = pd.read_parquet(f)
        feat["id"] = feat["id"].astype(str)
        X = X.merge(feat, on="id", how="left")
    folds = pd.read_csv(ROOT / "folds.csv")
    folds = folds[folds.scheme == scheme]
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in X["id"] if i in fmap]
    y = train.set_index("id")[target].astype(float)
    num = X.set_index("id").select_dtypes("number")
    oof = pd.Series(index=ids, dtype=float)
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        med = num.loc[tr].median().fillna(0)
        def prep(ix):
            return num.loc[ix].fillna(med).fillna(0.0)
        best_a, best = 1.0, 1e18
        for a in ALPHAS:
            sc = StandardScaler()
            m = Lasso(alpha=a, max_iter=200000, tol=1e-3, random_state=0)
            try:
                m.fit(sc.fit_transform(prep(tr)), y.loc[tr])
            except Exception:
                continue
            s = mae(y.loc[va], m.predict(sc.transform(prep(va))))
            if s < best:
                best, best_a = s, a
        tv = tr + va
        med2 = num.loc[tv].median().fillna(0)
        def prep2(ix):
            return num.loc[ix].fillna(med2).fillna(0.0)
        sc = StandardScaler()
        m = Lasso(alpha=best_a, max_iter=200000, tol=1e-3, random_state=0)
        m.fit(sc.fit_transform(prep2(tv)), y.loc[tv])
        oof.loc[te] = m.predict(sc.transform(prep2(te)))
    print("MAE", mae(y.loc[ids], oof), "alpha_last", best_a)

if __name__ == "__main__":
    print("Provide official train CSV path as argv; see README.")
'''
    )

    print("Bundle blocks:", sorted(needed))
    print("Wrote", BUNDLE)


if __name__ == "__main__":
    main()
