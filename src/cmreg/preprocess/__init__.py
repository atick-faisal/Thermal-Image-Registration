"""Preprocessing variants (TASKS.md P3-2): inversion, CLAHE, percentile normalization,
gradient magnitude, and thermal upsampling. Phase congruency and the learned polarity
front-end (P5-5) are outstanding. PLAN.md §15D catalogues the implementations ported here."""

from __future__ import annotations

from cmreg.preprocess.variants import (
    GrayImage,
    Preprocessed,
    PreprocessError,
    apply_variant,
    clahe,
    gradient,
    invert,
    percentile,
    preprocess_moving,
    preprocess_reference,
    upsample,
)

__all__ = [
    "GrayImage",
    "PreprocessError",
    "Preprocessed",
    "apply_variant",
    "clahe",
    "gradient",
    "invert",
    "percentile",
    "preprocess_moving",
    "preprocess_reference",
    "upsample",
]
