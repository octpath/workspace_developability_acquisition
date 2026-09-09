#!/usr/bin/env python3
"""Reusable antibody Transformer (behavior-preserving extract).

Historical learned representations are not archived; representation_status for
historical experiments is HISTORICAL_UNAVAILABLE. Future representation
aggregation across seeds is UNDECIDED (not seed-mean by default).
"""
from __future__ import annotations

from .fusion import FeatureFusionModel
from .model import AnnotatedTransformer
from .training import (
    build_transformer,
    full_dev_transformer_predict,
    run_transformer_cv,
    train_transformer_seed,
)

__all__ = [
    "AnnotatedTransformer",
    "FeatureFusionModel",
    "build_transformer",
    "train_transformer_seed",
    "run_transformer_cv",
    "full_dev_transformer_predict",
]
