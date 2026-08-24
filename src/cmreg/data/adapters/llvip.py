"""LLVIP: 15,488 aligned visible-infrared surveillance pairs (Jia et al., ICCVW 2021).

Driving/street-level domain (PLAN.md §1), fixed surveillance cameras rather than a vehicle,
and almost entirely night-time -- which is the reason to carry it alongside MSRS and
FLIR-aligned rather than instead of them: it is the low-light end of the same domain family,
where the optical side carries least information and the modality gap is widest.

Verified raw layout
-------------------
Fetched as a single ``LLVIP.zip`` (4.0 GB) from the Google Drive link in
``bupt-ai-cz/LLVIP``'s ``download_dataset.md`` -- the *registered* set, not the separate "raw"
download of unregistered pairs and video. Verified 2026-08-24::

    LLVIP/visible/train/  12,025 jpg   LLVIP/infrared/train/  12,025 jpg
    LLVIP/visible/test/    3,463 jpg   LLVIP/infrared/test/    3,463 jpg
    LLVIP/Annotations/    15,488 xml   (VOC boxes, visible frame)

All images are 1280x1024 JPEG stored as three channels; the infrared side has all three
channels equal (mean ``|R-G| + |G-B|`` of 0.00), the visible side does not (14.8). Stems match
across the two modality directories, so pairing is exact rather than positional.

Upstream's ``test`` becomes our ``val``: this project's ``DatasetManifest`` declares exactly
two splits, and a benchmark evaluates on the held-out one. The ``Annotations`` are not
ingested -- no classes until the Tier-2 box transfer of P2-10, which is when they become that
task's input.

Pixels are untouched. Entries are written **byte-for-byte out of the archive**, never
re-encoded, so the benchmark measures LLVIP's own JPEG and not ours on top of it.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from cmreg.data.adapters import Inventory, register
from cmreg.data.adapters.common import (
    INFRARED,
    VISIBLE,
    AdapterError,
    images_dir,
    write_image_pair_bytes,
)

logger = logging.getLogger(__name__)

ARCHIVE = "LLVIP.zip"
# Upstream directory -> our split. LLVIP publishes train/test; we evaluate on `val`.
SPLIT_SOURCES = {"train": "train", "val": "test"}
# Upstream modality directory -> our contract's segment name.
MODALITY_DIRS = {VISIBLE: "visible", INFRARED: "infrared"}
SUFFIX = ".jpg"


def _members(bundle: zipfile.ZipFile, modality_dir: str, upstream: str) -> dict[str, str]:
    prefix = f"LLVIP/{modality_dir}/{upstream}/"
    return {
        Path(name).stem: name
        for name in bundle.namelist()
        if name.startswith(prefix) and name.lower().endswith(SUFFIX)
    }


def _adapt_split(bundle: zipfile.ZipFile, upstream: str, split: str, dest_root: Path) -> int:
    entries = {
        modality: _members(bundle, directory, upstream)
        for modality, directory in MODALITY_DIRS.items()
    }
    if not entries[VISIBLE]:
        raise AdapterError(
            f"llvip: no images under LLVIP/{MODALITY_DIRS[VISIBLE]}/{upstream}/ in {ARCHIVE}"
        )
    if entries[VISIBLE].keys() != entries[INFRARED].keys():
        # Positional pairing is the failure `data/pairing.py` exists to prevent; catching a
        # stem mismatch here means it can never reach a results row.
        odd = sorted(entries[VISIBLE].keys() ^ entries[INFRARED].keys())[:5]
        raise AdapterError(
            f"llvip: {upstream} visible and infrared stems differ "
            f"({len(entries[VISIBLE])} vs {len(entries[INFRARED])}); e.g. {odd}"
        )

    split_root = dest_root / split
    for modality in MODALITY_DIRS:
        images_dir(split_root, modality)

    stems = sorted(entries[VISIBLE])
    for index, stem in enumerate(stems):
        target = images_dir(split_root, VISIBLE) / f"{stem}{SUFFIX}"
        if target.exists():
            continue
        write_image_pair_bytes(
            bundle.read(entries[VISIBLE][stem]),
            bundle.read(entries[INFRARED][stem]),
            stem,
            split_root,
            SUFFIX,
        )
        if (index + 1) % 2000 == 0:
            logger.info("  %s: %d / %d pairs", split, index + 1, len(stems))
    return len(stems)


def adapt(raw_root: Path, dest_root: Path) -> Inventory:
    archive = raw_root / ARCHIVE
    if not archive.is_file():
        raise AdapterError(f"llvip: expected {archive}; run the fetch script first")

    counts: dict[str, int] = {}
    with zipfile.ZipFile(archive) as bundle:
        for split, upstream in SPLIT_SOURCES.items():
            counts[split] = _adapt_split(bundle, upstream, split, dest_root)

    return Inventory(
        name="llvip",
        domain="driving",
        platform="public",
        train_pairs=counts["train"],
        val_pairs=counts["val"],
        resolution="1280x1024",
        note="upstream 'test' split becomes our 'val'; images copied byte-for-byte, not re-encoded",
    )


register("llvip", adapt)
