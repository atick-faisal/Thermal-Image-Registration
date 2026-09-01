"""Preprocessing variants (TASKS.md P3-2): inversion, CLAHE, percentile normalization,
gradient magnitude, thermal upsampling (P3-9) and the symmetric resolution axis (P3-12a).
Phase congruency and the learned polarity front-end (P5-5) are outstanding. PLAN.md §15D
catalogues the implementations ported here."""

from __future__ import annotations

from cmreg.preprocess.variants import (
    GrayImage,
    Preprocessed,
    PreprocessError,
    apply_variant,
    clahe,
    gradient,
    invert,
    kernel_for,
    percentile,
    preprocess_moving,
    preprocess_reference,
    rescale,
)

__all__ = [
    "GrayImage",
    "PreprocessError",
    "Preprocessed",
    "apply_variant",
    "clahe",
    "gradient",
    "invert",
    "kernel_for",
    "percentile",
    "preprocess_moving",
    "preprocess_reference",
    "rescale",
]
