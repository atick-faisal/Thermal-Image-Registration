"""The data contract: one manifest, one pairing rule, one frozen split record."""

from __future__ import annotations

from cmreg.data.manifest import IMAGE_SUFFIXES, DatasetManifest, ManifestError
from cmreg.data.pairing import Pairing, PairingError
from cmreg.data.splits import (
    SplitDriftError,
    SplitManifest,
    freeze_split,
    load_split_manifest,
    select_pairs,
    verify_split,
    write_split_manifest,
)

__all__ = [
    "IMAGE_SUFFIXES",
    "DatasetManifest",
    "ManifestError",
    "Pairing",
    "PairingError",
    "SplitDriftError",
    "SplitManifest",
    "freeze_split",
    "load_split_manifest",
    "select_pairs",
    "verify_split",
    "write_split_manifest",
]
