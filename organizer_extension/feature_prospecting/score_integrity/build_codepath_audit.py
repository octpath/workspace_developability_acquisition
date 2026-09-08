#!/usr/bin/env python3
"""Build curated SCORE_INTEGRITY_CODEPATH_AUDIT.csv (organizer eval paths only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent

ROWS = [
    # Known bug (historical commit behavior; current file fixed)
    {
        "evaluation": "AbLingua Sprint A Simple TVT",
        "script": "organizer_extension/feature_prospecting/ablingua600m/scripts/02_simple_tvt_eval.py",
        "commit_if_known": "c121f7bf",
        "uses_make_xy": "YES",
        "uses_load_seq_basic": "YES",
        "ID_index_explicit": "NO (historical)",
        "reindex_operation": "make_xy RangeIndex then index=str(RangeIndex) + reindex(ADI-*)",
        "potentially_affected": "YES",
        "reason": "CONFIRMED BUG: SEQ_BASIC entirely NaN; fixed later to seq_b.index=ids",
    },
    {
        "evaluation": "AbLingua Sprint A Simple TVT (fixed)",
        "script": "organizer_extension/feature_prospecting/ablingua600m/scripts/02_simple_tvt_eval.py",
        "commit_if_known": "7374433b",
        "uses_make_xy": "YES",
        "uses_load_seq_basic": "YES",
        "ID_index_explicit": "YES",
        "reindex_operation": "seq_b.index = list(ids); hard-fail if all-NaN",
        "potentially_affected": "NO",
        "reason": "FIXED load_seq_basic; regression test_parent_regression.py",
    },
    {
        "evaluation": "Organizer Simple TVT rescreen",
        "script": "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit/scripts/run_simple_tvt_rescreen.py",
        "commit_if_known": "",
        "uses_make_xy": "YES",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "seq_b.index = ids; aro.reindex(ids); emb index=ids",
        "potentially_affected": "NO",
        "reason": "Correct ID assignment after make_xy; same pattern as load_bases()",
    },
    {
        "evaluation": "Organizer Simple TVT CV",
        "script": "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit/scripts/run_simple_tvt_cv.py",
        "commit_if_known": "",
        "uses_make_xy": "YES",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "seq_b.index = ids; aro.reindex(ids)",
        "potentially_affected": "NO",
        "reason": "Correct ID assignment after make_xy",
    },
    {
        "evaluation": "Stage1 family screen (not Simple TVT)",
        "script": "virtual_participant/stage1_features/scripts/run_stage1.py",
        "commit_if_known": "",
        "uses_make_xy": "YES",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "positional with y",
        "reindex_operation": "make_xy RangeIndex; y aligned by row position in same tables",
        "potentially_affected": "NO",
        "reason": "Internal CV uses same-table row order for X and y; not organizer ID reindex path",
    },
    {
        "evaluation": "AbLingua guided pooling TVT",
        "script": "organizer_extension/feature_prospecting/ablingua600m/scripts/04_guided_pooling_tvt.py",
        "commit_if_known": "27ce3f36",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "parquet set_index(id); parent from corrected recipe",
        "potentially_affected": "NO",
        "reason": "Used authoritative PARENT (correct SEQ); CDR3 GUIDED_TRY unchanged",
    },
    {
        "evaluation": "BioEmu isolated / Simple TVT via rescreen",
        "script": "organizer_extension/feature_prospecting/bioemu_isolated_reassessment/scripts/eval_reassess.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "feature CSV id column; common ridge_eval",
        "potentially_affected": "NO",
        "reason": "ID-keyed features; no make_xy reindex bug pattern",
    },
    {
        "evaluation": "ProteinMPNN M1 family eval",
        "script": "organizer_extension/feature_prospecting/structure_marathon/scripts/eval_families.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "CSV id index alignment",
        "potentially_affected": "NO",
        "reason": "No SEQ_BASIC make_xy path",
    },
    {
        "evaluation": "Gap closure / aromatic / surface",
        "script": "organizer_extension/feature_prospecting/structure_gap_closure/scripts/eval_gap_closure.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "feature id alignment",
        "potentially_affected": "NO",
        "reason": "Structure features keyed by id; BASE built via rescreen load_bases",
    },
    {
        "evaluation": "OpenMM Fab MD Simple TVT",
        "script": "organizer_extension/feature_prospecting/openmm_fab_md/scripts",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "usable cohort id list + feature parquet id",
        "potentially_affected": "NO",
        "reason": "Intersection cohort explicit; BASE from load_bases pattern",
    },
    {
        "evaluation": "FeNNix interim audit",
        "script": "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit/scripts/run_interim_audit.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "series.reindex(score.index)",
        "potentially_affected": "REVIEW",
        "reason": "Uses reindex on score index; not make_xy bug; verify OOF/id alignment if reused",
    },
    {
        "evaluation": "Common ridge_eval helper",
        "script": "organizer_extension/feature_prospecting/common/ridge_eval.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "oof Series(index=ids); X.loc[ids]",
        "potentially_affected": "NO",
        "reason": "Requires caller to pass ID-indexed X; no silent RangeIndex reindex",
    },
    {
        "evaluation": "Canonical Simple TVT (new)",
        "script": "organizer_extension/feature_prospecting/score_integrity/canonical_simple_tvt.py",
        "commit_if_known": "",
        "uses_make_xy": "NO",
        "uses_load_seq_basic": "NO",
        "ID_index_explicit": "YES",
        "reindex_operation": "align_feature_block hard-fail; concat by id",
        "potentially_affected": "NO",
        "reason": "Authoritative evaluator; rejects all-NaN / missing IDs",
    },
]


def main():
    df = pd.DataFrame(ROWS)
    path = OUT / "SCORE_INTEGRITY_CODEPATH_AUDIT.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} n={len(df)}")
    print(df["potentially_affected"].value_counts().to_string())


if __name__ == "__main__":
    main()
