#!/usr/bin/env python3
"""Unit tests for advanced model bundle (no organizer_extension imports)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from advanced_models.config import CHAIN_TO_IDX
from advanced_models.data import (
    DataIntegrityError,
    load_annotations,
    load_dev_test,
    load_residue_bundle,
    load_solution,
)
from advanced_models.models.annotated_transformer import AnnotatedTransformer


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev, cls.test = load_dev_test(ROOT / "dev.csv", ROOT / "test.csv")
        cls.ann = load_annotations()

    def test_reconstruction(self):
        seqs = pd.concat(
            [self.dev[["id", "heavy", "light"]], self.test[["id", "heavy", "light"]]],
            ignore_index=True,
        )
        for r in seqs.itertuples(index=False):
            for chain, seq in (("H", r.heavy), ("L", r.light)):
                got = "".join(
                    self.ann[(self.ann.id == str(r.id)) & (self.ann.chain == chain)]
                    .sort_values("seq_index")["aa"]
                    .tolist()
                )
                self.assertEqual(got, seq)

    def test_imgt_insertion_survives(self):
        # roundtrip parquet
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.parquet"
            self.ann.to_parquet(p)
            b = pd.read_parquet(p)
            has_ins = b[b.imgt_insertion.astype(str).str.len() > 0]
            self.assertGreater(len(has_ins), 0)
            self.assertTrue((has_ins.imgt_position.astype(str).str.contains(r"[A-Z]$")).any())


class TransformerShapeTests(unittest.TestCase):
    def test_shared_encoder_and_shapes(self):
        m = AnnotatedTransformer(
            content_mode="scratch",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            n_imgt=20,
            max_seq_pos=40,
        )
        # shared encoder module
        self.assertTrue(hasattr(m, "encoder"))
        self.assertEqual(m.chain_emb.num_embeddings, 2)
        self.assertNotEqual(CHAIN_TO_IDX["H"], CHAIN_TO_IDX["L"])
        B, L = 2, 10
        batch = {
            "heavy_aa": torch.randint(1, 21, (B, L)),
            "light_aa": torch.randint(1, 21, (B, L)),
            "heavy_mask": torch.ones(B, L, dtype=torch.bool),
            "light_mask": torch.ones(B, L, dtype=torch.bool),
            "heavy_pos": torch.arange(1, L + 1).unsqueeze(0).expand(B, L),
            "light_pos": torch.arange(1, L + 1).unsqueeze(0).expand(B, L),
            "heavy_imgt": torch.randint(1, 10, (B, L)),
            "light_imgt": torch.randint(1, 10, (B, L)),
            "heavy_region": torch.randint(1, 8, (B, L)),
            "light_region": torch.randint(1, 8, (B, L)),
        }
        out = m(batch)
        self.assertEqual(tuple(out.shape), (B,))
        h = m.forward_repr(batch)
        self.assertEqual(h.shape[-1], 256)  # concat

        m2 = AnnotatedTransformer(
            content_mode="scratch",
            annotation_mode="minimal",
            merge_mode="mean",
            chain_mode="HL",
            n_imgt=20,
            max_seq_pos=40,
        )
        self.assertEqual(m2.forward_repr(batch).shape[-1], 128)

        m3 = AnnotatedTransformer(
            content_mode="scratch",
            annotation_mode="minimal",
            merge_mode="h_only",
            chain_mode="H_ONLY",
            n_imgt=20,
            max_seq_pos=40,
        )
        self.assertEqual(m3.forward_repr(batch).shape[-1], 128)

    def test_padding_and_reg_mask(self):
        m = AnnotatedTransformer(
            content_mode="scratch",
            annotation_mode="minimal",
            merge_mode="h_only",
            chain_mode="H_ONLY",
            max_seq_pos=40,
        )
        B, L = 1, 8
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[0, :5] = True
        batch = {
            "heavy_aa": torch.randint(1, 21, (B, L)),
            "heavy_mask": mask,
            "heavy_pos": torch.arange(1, L + 1).unsqueeze(0),
        }
        # should run without error; REG always present
        _ = m(batch)


class IntegrityTests(unittest.TestCase):
    def test_duplicate_id_fails(self):
        dev, test = load_dev_test(ROOT / "dev.csv", ROOT / "test.csv")
        bad = dev.copy()
        bad.loc[0, "id"] = bad.loc[1, "id"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dev.csv"
            bad.to_csv(p, index=False)
            with self.assertRaises(DataIntegrityError):
                load_dev_test(p, ROOT / "test.csv")

    def test_solution_masks(self):
        sol_path = ROOT / "solution.csv"
        if not sol_path.exists():
            self.skipTest("solution.csv not present")
        sol = load_solution(sol_path)
        self.assertEqual(len(sol), 162)
        self.assertTrue((sol.is_public.astype(bool) ^ sol.is_private.astype(bool)).all())

    def test_frozen_alignment(self):
        if not (ROOT / "residue_level/esm2/ids.npy").exists():
            self.skipTest("residue assets missing")
        dev, test = load_dev_test(ROOT / "dev.csv", ROOT / "test.csv")
        rb = load_residue_bundle(dev, test, need_esm2=True)
        self.assertEqual(rb.esm2_hidden, 1280)
        self.assertEqual(len(rb.ids), 324)


class SubmissionSchemaTests(unittest.TestCase):
    def test_schema(self):
        ids = [f"x{i}" for i in range(162)]
        df = pd.DataFrame({"id": ids, "TmApp": 1.0, "HIC": 0.5})
        self.assertEqual(list(df.columns), ["id", "TmApp", "HIC"])
        self.assertEqual(len(df), 162)


if __name__ == "__main__":
    unittest.main()
