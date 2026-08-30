#!/usr/bin/env python3
"""Validate frozen split + participant staging; write tests report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import CONFIG, ORG, PART, REPORTS, ensure_dirs, read_json, sha256_lines  # noqa: E402


def main():
    ensure_dirs()
    errors = []
    man = read_json(CONFIG / "FINAL_SPLIT_MANIFEST.json")
    pop = pd.read_csv(ORG / "final_population.csv")
    train = pd.read_csv(ORG / "train_ids.csv")
    pub = pd.read_csv(ORG / "public_ids.csv")
    priv = pd.read_csv(ORG / "private_ids.csv")
    role_map = pd.read_csv(ORG / "role_map.csv")
    p_train = pd.read_csv(PART / "train.csv")
    p_test = pd.read_csv(PART / "test.csv")
    sample = pd.read_csv(PART / "sample_submission.csv")
    hidden = pd.read_csv(ORG / "test_labels_hidden.csv")

    # duplicates
    if pop["id"].duplicated().any():
        errors.append("duplicate ids in population")
    if pop.duplicated(subset=["heavy", "light"]).any():
        errors.append("duplicate VH/VL pairs")

    # roles partition
    all_ids = set(train.id) | set(pub.id) | set(priv.id)
    if all_ids != set(pop.id):
        errors.append("id partition incomplete")
    if len(set(train.id) & set(pub.id)) or len(set(train.id) & set(priv.id)) or len(set(pub.id) & set(priv.id)):
        errors.append("role id overlap")

    # group leakage
    gmap = pop.set_index("id")["sequence_group"]
    for role_ids, name in [(train.id, "Train"), (pub.id, "Public"), (priv.id, "Private")]:
        pass
    for g, sub in pop.groupby("sequence_group"):
        if sub["id"].map(role_map.set_index("id")["role"]).nunique() > 1:
            # need role in pop - use role_map
            roles = set(role_map.set_index("id").loc[sub["id"], "role"])
            if len(roles) > 1:
                errors.append(f"group leakage group={g} roles={roles}")

    # targets complete
    if pop[["HIC", "TmApp"]].isna().any().any():
        errors.append("missing targets")

    # same public/private for both tracks (by construction one split)
    if set(pub.id) != set(hidden.loc[hidden.role == "Public", "id"]):
        errors.append("public id mismatch hidden")
    if set(priv.id) != set(hidden.loc[hidden.role == "Private", "id"]):
        errors.append("private id mismatch hidden")

    # participant schema
    if list(p_train.columns) != ["id", "heavy", "light", "TmApp", "HIC"]:
        errors.append(f"train schema {list(p_train.columns)}")
    if list(p_test.columns) != ["id", "heavy", "light"]:
        errors.append(f"test schema {list(p_test.columns)}")
    if list(sample.columns) != ["id", "TmApp", "HIC"]:
        errors.append(f"sample schema {list(sample.columns)}")
    for bad in ["role", "b_cell_subset", "donor", "sequence_group"]:
        if bad in p_train.columns or bad in p_test.columns:
            errors.append(f"organizer column leaked: {bad}")

    # hash check
    if sha256_lines(train.id) != man["id_list_sha256"]["train"]:
        errors.append("train hash mismatch")
    if sha256_lines(pub.id) != man["id_list_sha256"]["public"]:
        errors.append("public hash mismatch")
    if sha256_lines(priv.id) != man["id_list_sha256"]["private"]:
        errors.append("private hash mismatch")

    n_tr, n_pu, n_pr = len(train), len(pub), len(priv)
    ok = len(errors) == 0
    lines = [
        "# Competition staging audit / split tests",
        "",
        f"- PASS: {ok}",
        f"- Counts Train/Public/Private: {n_tr}/{n_pu}/{n_pr}",
        f"- |Public-Private|: {abs(n_pu-n_pr)}",
        f"- Errors: {errors if errors else 'none'}",
        "",
        "Assertions covered: no duplicate ids/pairs, no group leakage, target completeness,",
        "identical Public/Private IDs for both tracks, participant schema, hidden-label isolation, hash match.",
        "",
    ]
    (REPORTS / "competition_staging_audit.md").write_text("\n".join(lines) + "\n")
    print("TESTS", "PASS" if ok else "FAIL", errors, f"{n_tr}/{n_pu}/{n_pr}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
