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

    def test_region_gate_normalization(self):
        m = AnnotatedTransformer(
            content_mode="scratch",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            pooling_mode="region_gate",
            n_imgt=20,
            max_seq_pos=40,
        )
        w = m.region_gate_weights()
        self.assertAlmostEqual(float(w["H"].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(w["L"].sum()), 1.0, places=5)
        B, L = 2, 12
        # cover FR + CDRs
        from advanced_models.config import REGION_TO_IDX

        region = torch.zeros(B, L, dtype=torch.long)
        region[:, :3] = REGION_TO_IDX["FR1"]
        region[:, 3:5] = REGION_TO_IDX["CDR1"]
        region[:, 5:7] = REGION_TO_IDX["CDR2"]
        region[:, 7:9] = REGION_TO_IDX["CDR3"]
        region[:, 9:] = REGION_TO_IDX["FR4"]
        batch = {
            "heavy_aa": torch.randint(1, 21, (B, L)),
            "light_aa": torch.randint(1, 21, (B, L)),
            "heavy_mask": torch.ones(B, L, dtype=torch.bool),
            "light_mask": torch.ones(B, L, dtype=torch.bool),
            "heavy_pos": torch.arange(1, L + 1).unsqueeze(0).expand(B, L),
            "light_pos": torch.arange(1, L + 1).unsqueeze(0).expand(B, L),
            "heavy_imgt": torch.randint(1, 10, (B, L)),
            "light_imgt": torch.randint(1, 10, (B, L)),
            "heavy_region": region,
            "light_region": region.clone(),
        }
        out = m(batch)
        self.assertEqual(tuple(out.shape), (B,))

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


class AbLang2ResidueTests(unittest.TestCase):
    def test_ablang2_bundle_if_present(self):
        meta = ROOT / "residue_level/ablang2/metadata.json"
        if not meta.exists():
            self.skipTest("ablang2 residue assets missing")
        import json

        m = json.loads(meta.read_text())
        self.assertEqual(m["hidden_dim"], 480)
        self.assertEqual(m["mapping"], "EXACT_AA_1TO1_SPECIAL_STRIPPED")
        self.assertEqual(m["license_status"], "REVIEW_MODEL_OUTPUT")
        qc = json.loads((ROOT / "residue_level/ablang2/ablang2_residue_qc_summary.json").read_text())
        self.assertEqual(qc["n_unique_ids"], 324)
        self.assertTrue(qc["all_residue_eq_seq_len"])
        self.assertTrue(qc["finite_100pct"])
        self.assertEqual(qc["semantic_vs_pooled"]["aa_residue_mean_vs_seqcoding"], "NOT_SEMANTICALLY_COMPARABLE")
        # assemble path: load via ResidueBundle
        if not (ROOT / "residue_level/ablang2/heavy_embeddings.npy").exists() and not (
            ROOT / "residue_level/ablang2/heavy_embeddings.npy.part0"
        ).exists():
            self.skipTest("ablang2 parts missing")
        dev, test = load_dev_test(ROOT / "dev.csv", ROOT / "test.csv")
        rb = load_residue_bundle(dev, test, need_ablang2=True)
        self.assertEqual(rb.ablang2_hidden, 480)
        self.assertEqual(rb.ablang2_h.shape[0], 324)
        self.assertTrue(np.isfinite(rb.ablang2_h).all())

    def test_benchmark_row_uniqueness(self):
        path = ROOT / "results/MODEL_BENCHMARK_SUMMARY.csv"
        if not path.exists():
            self.skipTest("benchmark missing")
        df = pd.read_csv(path)
        self.assertFalse(df.duplicated(["target", "model_id"]).any())
        gi = (ROOT.parent / ".gitignore").read_text()
        self.assertIn("top_models_feature_bundle/solution.csv", gi)


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
