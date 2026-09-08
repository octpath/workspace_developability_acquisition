#!/usr/bin/env python3
"""Validate README.md / README_JA.md against recipes.csv and ADVANCED freeze."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREEZE = (
    ROOT.parent
    / "organizer_extension/linear_model_closure/ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json"
)


def recipe_ids_from_text(text: str) -> list[str]:
    # recipe IDs appear as `TM_...` or `HIC_...`
    return re.findall(r"`((?:TM_|HIC_)[A-Z0-9_]+)`", text)


def main() -> int:
    en = (ROOT / "README.md").read_text()
    ja = (ROOT / "README_JA.md").read_text()
    recipes = __import__("pandas").read_csv(ROOT / "recipes.csv")
    freeze = json.loads(FREEZE.read_text())

    # duplicated section check
    n_folds = len(re.findall(r"^## Recommended folds\s*$", en, flags=re.M))
    n_top = len(
        re.findall(r"^## Top recipes \(advanced-model freeze; CV only\)\s*$", en, flags=re.M)
    )
    dup_ok = n_folds == 1 and n_top == 1

    freeze_ids = []
    for target in ("TmApp", "HIC"):
        freeze_ids.extend([x["model_id"] for x in freeze["targets"][target]])
    csv_ids = recipes.sort_values(["target", "cv_rank"]).recipe_id.tolist()
    # recipes.csv order may be TmApp then HIC
    csv_ids_ordered = (
        recipes[recipes.target == "TmApp"].sort_values("cv_rank").recipe_id.tolist()
        + recipes[recipes.target == "HIC"].sort_values("cv_rank").recipe_id.tolist()
    )

    en_ids = recipe_ids_from_text(en)
    ja_ids = recipe_ids_from_text(ja)
    # keep first occurrence of each Top-3 id in document order within Top recipes section
    def top3_ids(text: str) -> list[str]:
        # after the Top recipes heading until Quick usage / 簡易利用例
        m = re.search(
            r"## Top recipes.*?\n(.*?)(?:\n## Quick usage|\n## 簡易利用例)",
            text,
            flags=re.S,
        )
        body = m.group(1) if m else text
        ids = recipe_ids_from_text(body)
        # unique preserving order
        out = []
        for i in ids:
            if i not in out:
                out.append(i)
        return out

    en_top = top3_ids(en)
    ja_top = top3_ids(ja)

    en_ja = en_top == ja_top == freeze_ids == csv_ids_ordered
    freeze_ok = en_top == freeze_ids and csv_ids_ordered == freeze_ids

    # CV score strings from freeze must appear in both
    score_ok = True
    for target in ("TmApp", "HIC"):
        for x in freeze["targets"][target]:
            p = f"{x['cv_primary_mae']}"
            s = f"{x['cv_shadow_mae']}"
            if p not in en or p not in ja or s not in en or s not in ja:
                score_ok = False
                print("MISSING SCORE", x["model_id"], p, s)

    # blocks
    man = __import__("pandas").read_csv(ROOT / "feature_manifest.csv")
    blocks = sorted(man.block_name.tolist())
    blocks_ok = all(b in en and b in ja for b in blocks)

    # fold rotation
    fold_ok = ("TEST=k, VAL=(k+1)%5" in en) and ("TEST=k, VAL=(k+1)%5" in ja)

    en_ja_status = "PASS" if (en_ja and score_ok and blocks_ok and fold_ok and dup_ok) else "FAIL"
    freeze_status = "PASS" if freeze_ok else "FAIL"

    print(f"n_Recommended_folds_sections={n_folds}")
    print(f"n_Top_recipes_freeze_sections={n_top}")
    print(f"en_top={en_top}")
    print(f"ja_top={ja_top}")
    print(f"freeze_ids={freeze_ids}")
    print(f"csv_ids={csv_ids_ordered}")
    print(f"README_EN_JA_CONSISTENCY = {en_ja_status}")
    print(f"README_RECIPE_FREEZE_CONSISTENCY = {freeze_status}")

    audit = ROOT / "README_CONSISTENCY_AUDIT.md"
    audit.write_text(
        "# Bundle README consistency audit\n\n"
        f"- duplicated Recommended folds sections in README.md: {n_folds} (expect 1)\n"
        f"- duplicated Top recipes (freeze) sections in README.md: {n_top} (expect 1)\n"
        f"- EN Top-3 IDs: {en_top}\n"
        f"- JA Top-3 IDs: {ja_top}\n"
        f"- freeze IDs: {freeze_ids}\n"
        f"- recipes.csv IDs: {csv_ids_ordered}\n"
        f"\n**README_EN_JA_CONSISTENCY = {en_ja_status}**\n"
        f"**README_RECIPE_FREEZE_CONSISTENCY = {freeze_status}**\n"
    )

    return 0 if en_ja_status == "PASS" and freeze_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
