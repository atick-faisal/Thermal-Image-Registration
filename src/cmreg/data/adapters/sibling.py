"""Datasets the sibling project already adapted: MSRS and FLIR-aligned.

``../Thermal-To-Optical-Translation`` converted both into the same
``{split}/{visible,infrared}/images`` contract this project reads, and they sit in the very
tree ``cmreg ingest`` writes into. Re-adapting them would duplicate gigabytes to produce
byte-identical output.

So this adapter converts nothing. It **verifies** the tree is shaped the way the contract
requires and reports the inventory row, which is the part that was actually missing: without
it those two datasets are the only ones absent from ``cmreg ingest --list``, and an inventory
table with holes in it is one nobody trusts.

Kept honest by the same rule as every other adapter -- it fails loudly, naming the directory
it looked for, rather than reporting zero pairs for a tree that is not there.
"""

from __future__ import annotations

from pathlib import Path

from cmreg.data.adapters import Inventory, register
from cmreg.data.adapters.common import (
    IMAGES_SEGMENT,
    INFRARED,
    SPLITS,
    VISIBLE,
    AdapterError,
)
from cmreg.data.manifest import IMAGE_SUFFIXES

# name -> (domain, platform, resolution, note). Recorded here rather than looked up per run so
# that a benchmark cell over these sets declares the same domain every other cell does.
KNOWN = {
    "msrs": ("driving", "public", "640x480", "adapted by ../Thermal-To-Optical-Translation"),
    "flir": ("driving", "public", "640x512", "FLIR-aligned; adapted by the sibling project"),
}


def _count(split_root: Path, modality: str, name: str) -> int:
    directory = split_root / modality / IMAGES_SEGMENT
    if not directory.is_dir():
        raise AdapterError(
            f"{name}: expected {directory}. This dataset is meant to be already adapted by "
            f"../Thermal-To-Optical-Translation; nothing here converts it."
        )
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _adapt(name: str, raw_root: Path, dest_root: Path) -> Inventory:
    del raw_root  # nothing is read from raw/: the adapted tree is the input and the output.
    counts: dict[str, int] = {}
    for split in SPLITS:
        visible = _count(dest_root / split, VISIBLE, name)
        infrared = _count(dest_root / split, INFRARED, name)
        if visible != infrared:
            raise AdapterError(
                f"{name}: {split} has {visible} visible and {infrared} infrared images; "
                f"the pair set is incomplete"
            )
        counts[split] = visible

    domain, platform, resolution, note = KNOWN[name]
    return Inventory(
        name=name,
        domain=domain,
        platform=platform,
        train_pairs=counts["train"],
        val_pairs=counts["val"],
        resolution=resolution,
        note=note,
    )


def _register_all() -> None:
    for name in KNOWN:
        # Default-argument binding, not a closure over the loop variable: the late-binding
        # version registers every name against whichever one the loop ended on.
        register(name, lambda raw, dest, _name=name: _adapt(_name, raw, dest))


_register_all()
