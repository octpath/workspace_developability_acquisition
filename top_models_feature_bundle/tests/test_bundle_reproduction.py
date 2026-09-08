#!/usr/bin/env python3
"""Unit tests for bundle self-reproduction (no organizer_extension imports)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bundle_simple_tvt import FeatureAlignmentError, align_feature_block  # noqa: E402
from reproduce_top_recipes import BLOCK_FILE, verify_freeze  # noqa: E402


class BundleTests(unittest.TestCase):
    def test_freeze_matches_recipes(self):
        recipes = pd.read_csv(ROOT / "recipes.csv")
        verify_freeze(recipes)

    def test_shuffled_dev_and_features(self):
        from reproduce_top_recipes import build_parts, load_block, run_cv

        dev = pd.read_csv(ROOT / "dev.csv").sample(frac=1.0, random_state=0).reset_index(drop=True)
        ids = dev["id"].astype(str).tolist()
        folds = pd.read_csv(ROOT / "folds.csv")
        folds = folds[folds.scheme == "primary"][["id", "fold"]]
        y = dev.set_index("id")["HIC"].astype(str)
        y = pd.to_numeric(dev.set_index("id")["HIC"])
        # shuffled feature rows
        raw = pd.read_parquet(ROOT / "data/esm2_heavy.parquet").sample(frac=1.0, random_state=1)
        aligned = align_feature_block(raw, ids, "ESM2_H")
        self.assertEqual(list(aligned.index), ids)
        parts = build_parts(
            ["ESM2_H", "SEQ_ALL", "AROMATIC_TOPO"],
            ids,
            ROOT,
            "HIC_ESM2_SEQ_AROMATIC__LASSO",
        )
        r = run_cv(parts, y, folds, "LASSO")
        self.assertAlmostEqual(r["mae"], 0.4927303900206625, places=10)

    def test_duplicate_ids_fail(self):
        ids = ["a", "a"]
        df = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
        with self.assertRaises(FeatureAlignmentError):
            align_feature_block(df, ids, "dup")

    def test_missing_id_fail(self):
        df = pd.DataFrame({"id": ["a"], "x": [1.0]})
        with self.assertRaises(FeatureAlignmentError):
            align_feature_block(df, ["a", "missing"], "miss")

    def test_all_nan_block_fail(self):
        df = pd.DataFrame({"id": ["a", "b"], "x": [np.nan, np.nan]})
        with self.assertRaises(FeatureAlignmentError):
            align_feature_block(df, ["a", "b"], "nan")

    def test_submission_columns(self):
        sub = pd.read_csv(ROOT / "outputs/submission.csv")
        self.assertEqual(list(sub.columns), ["id", "TmApp", "HIC"])
        self.assertEqual(len(sub), 162)

    def test_solution_masks(self):
        sol = ROOT / "solution.csv"
        if not sol.exists():
            self.skipTest("solution.csv internal — not present")
        s = pd.read_csv(sol)
        self.assertEqual(int(s.is_public.sum()), 81)
        self.assertEqual(int(s.is_private.sum()), 81)
        self.assertTrue((s.is_public.astype(bool) ^ s.is_private.astype(bool)).all())

    def test_readme_examples_mention_script(self):
        for name in ("README.md", "README_JA.md"):
            t = (ROOT / name).read_text()
            self.assertIn("reproduce_top_recipes.py", t)
            self.assertIn("dev.csv", t)
            self.assertIn("test.csv", t)


class CleanDirectoryTest(unittest.TestCase):
    def test_clean_dir_standalone(self):
        release = [
            "README.md",
            "README_JA.md",
            "dev.csv",
            "test.csv",
            "folds.csv",
            "recipes.csv",
            "feature_manifest.csv",
            "EXPECTED_SCORES.json",
            "FULL_DEV_ALPHA_POLICY_BUNDLE.json",
            "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json",
            "bundle_simple_tvt.py",
            "reproduce_top_recipes.py",
            "RELEASE_FILE_POLICY.md",
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for f in release:
                src = ROOT / f
                if src.is_file():
                    shutil.copy2(src, td / f)
            shutil.copytree(ROOT / "data", td / "data")
            # Ensure no accidental organizer imports by removing PYTHONPATH tricks
            env = dict(**{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"})
            # Run one light recipe for speed
            cmd = [
                sys.executable,
                "reproduce_top_recipes.py",
                "--dev",
                "dev.csv",
                "--test",
                "test.csv",
                "--recipe",
                "HIC_ESM2_SEQ_AROMATIC__LASSO",
                "--outdir",
                "outputs",
            ]
            r = subprocess.run(cmd, cwd=td, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("EXACT_REPRODUCTION", r.stdout)
            # Must not import repository-internal packages at runtime
            import re

            src = (td / "reproduce_top_recipes.py").read_text() + "\n" + (td / "bundle_simple_tvt.py").read_text()
            self.assertIsNone(re.search(r"(?m)^\s*(import|from)\s+organizer_extension\b", src))
            self.assertIsNone(re.search(r"(?m)^\s*(import|from)\s+virtual_participant\b", src))


if __name__ == "__main__":
    unittest.main()
