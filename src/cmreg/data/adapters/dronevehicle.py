"""DroneVehicle: 28,439 drone-captured RGB-infrared pairs (VisDrone, T-CSVT 2022).

The aerial domain family of PLAN.md §1 -- the hard case the whole paper leans on, and the
only non-driving public set in the benchmark until the private drone pairs arrive (P1-4).

Verified raw layout
-------------------
Fetched from the Hugging Face mirror ``McCheng/DroneVehicle`` as three zips (13.1 GB). Each
holds one split under a directory named after it, with four sibling directories::

    val.zip
      val/valimg/00001.jpg .. 01469.jpg      1,469 RGB
      val/valimgr/00001.jpg .. 01469.jpg     1,469 infrared
      val/vallabel/                          1,469 VOC-XML, RGB boxes
      val/vallabelr/                         1,469 VOC-XML, infrared boxes

Verified 2026-08-24 on ``val.zip``: the two image directories carry **identical stems**, so
pairing is exact rather than positional. The ``r`` suffix is the infrared side -- confirmed
by measurement, not by the README: every ``valimgr`` image has all three channels equal
(mean ``|R-G| + |G-B|`` of 0.00 over sampled images) while every ``valimg`` image does not
(2.4-4.1). The label directories are not ingested; this project has no classes until the
Tier-2 box transfer of P2-10, at which point they become that task's input.

``test.zip`` (8,980 pairs) is deliberately **not** ingested. It is the same distribution as
the other two, P1-8 will define this project's own leakage-free splits regardless, and
skipping it saves 6 GB for pairs nothing would read.

Corrupt upstream entries
------------------------
``train.zip`` contains **39 unreadable JPEG entries out of 35,980** (0.11%), each raising
``zipfile.BadZipFile: Bad CRC-32``. This is a defect in the published archive, not in the
download: the local file's sha256 matches the Hugging Face manifest byte for byte
(``d22eccae51872835...``, 8,880,004,598 bytes). The bad entries fall in contiguous runs --
``trainimg`` 02220-02223, 04895-04898, 05598-05599, 09012-09014, 12900-12902; ``trainimgr``
04991-04994, 05001-05003, 06463-06467, 14035-14038, 14048-14050, 16009-16012 -- which is the
signature of damaged storage blocks rather than scattered bit rot. ``val.zip`` is clean.

The adapter skips the affected **pairs** (both sides, since half a pair is unpairable), counts
them, logs them, and carries the count into the inventory note. It does not raise: one
unreadable JPEG must not cost this project its only aerial dataset. TASKS.md X-4 -- recorded,
never silently dropped.

The 100-pixel border
--------------------
Every image is 840x712 with a **100 px pure-white border on all four sides**; the content is
the middle 640x512, which is the native resolution of the drone's sensors. Verified: rows
0-99 have mean exactly 255.0 and row 100 is content (JPEG ringing puts the border *minimum*
at 253, which is why the check below is on the mean and not on equality with 255).

The border is cropped, and that is why this adapter re-encodes rather than hardlinking. It
writes **PNG**, not JPEG: the source is already JPEG, a second lossy pass would bake
compression artifacts into every pair, and JPEG quality is one of the degradation axes this
benchmark controls deliberately (P2-3). Costs ~14 GB for train+val against ~4 GB for JPEG-95.

The crop is *checked* before it is applied. A future release without the border would
otherwise lose 100 px of real content on every side, silently, and show up months later as an
inexplicable per-dataset offset.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from cmreg.data.adapters import Inventory, register
from cmreg.data.adapters.common import INFRARED, VISIBLE, AdapterError, images_dir

logger = logging.getLogger(__name__)

# Pixels of white margin on each side. Not a tunable: it is a property of how the upstream
# dataset was published, and the assertion below fails loudly if a release ever changes it.
BORDER_PX = 100
# A border row averaging below this is not a border. Generous, because the constant exists to
# catch "there is no border any more", not to grade JPEG ringing.
WHITE_MEAN = 250.0

# Upstream split -> ours. `test` is absent on purpose; see the module docstring.
SPLIT_SOURCES = {"train": "train", "val": "val"}
SUFFIX = ".png"


def _crop_border(image: Image.Image, stem: str, modality: str) -> Image.Image:
    """Remove the published white margin, after verifying it is actually there."""
    width, height = image.size
    if width <= 2 * BORDER_PX or height <= 2 * BORDER_PX:
        raise AdapterError(
            f"dronevehicle: {modality}/{stem} is {width}x{height}, too small to carry the "
            f"{BORDER_PX} px border this dataset publishes"
        )
    probe = np.asarray(image.convert("L"), dtype=np.float64)
    edges = {
        "top": probe[:BORDER_PX].mean(),
        "bottom": probe[-BORDER_PX:].mean(),
        "left": probe[:, :BORDER_PX].mean(),
        "right": probe[:, -BORDER_PX:].mean(),
    }
    faint = {side: value for side, value in edges.items() if value < WHITE_MEAN}
    if faint:
        raise AdapterError(
            f"dronevehicle: {modality}/{stem} has no white {BORDER_PX} px border "
            f"({', '.join(f'{s} mean {v:.1f}' for s, v in faint.items())}). The upstream layout "
            f"has changed; cropping here would discard real content on every image"
        )
    return image.crop((BORDER_PX, BORDER_PX, width - BORDER_PX, height - BORDER_PX))


def _adapt_split(archive: Path, upstream: str, split: str, dest_root: Path) -> tuple[int, str, int]:
    """Crop and write one split. Returns (pairs written, resolution, pairs skipped)."""
    if not archive.is_file():
        raise AdapterError(f"dronevehicle: expected {archive}; run the fetch script first")

    split_root = dest_root / split
    with zipfile.ZipFile(archive) as bundle:
        members = {
            modality: sorted(
                name
                for name in bundle.namelist()
                if name.startswith(f"{upstream}/{directory}/") and name.lower().endswith(".jpg")
            )
            for modality, directory in ((VISIBLE, f"{upstream}img"), (INFRARED, f"{upstream}imgr"))
        }
        stems = {
            modality: [Path(name).stem for name in names] for modality, names in members.items()
        }
        if not stems[VISIBLE]:
            found = sorted({name.split("/")[1] for name in bundle.namelist() if "/" in name})
            raise AdapterError(
                f"dronevehicle: no images under {upstream}/{upstream}img/ in {archive.name}; "
                f"found directories {found}"
            )
        if stems[VISIBLE] != stems[INFRARED]:
            # Positional pairing is the failure `data/pairing.py` exists to prevent; catching
            # the mismatch here means it can never reach a results row.
            missing = sorted(set(stems[VISIBLE]) ^ set(stems[INFRARED]))[:5]
            raise AdapterError(
                f"dronevehicle: {archive.name} RGB and infrared stems differ "
                f"({len(stems[VISIBLE])} vs {len(stems[INFRARED])}); e.g. {missing}"
            )

        for modality in (VISIBLE, INFRARED):
            images_dir(split_root, modality)

        # Stem-outer, both modalities together: a pair is the unit that must survive or be
        # dropped. Writing one side and losing the other would leave an unpairable file, which
        # `Pairing.validate_pairs` would then reject for the whole split.
        by_stem = {
            modality: dict(zip(stems[modality], names, strict=True))
            for modality, names in members.items()
        }
        written = 0
        corrupt: list[str] = []
        for index, stem in enumerate(stems[VISIBLE]):
            targets = {
                modality: images_dir(split_root, modality) / f"{stem}{SUFFIX}"
                for modality in (VISIBLE, INFRARED)
            }
            if all(target.exists() for target in targets.values()):
                written += 1
                continue
            try:
                cropped: dict[str, Image.Image] = {}
                for modality in (VISIBLE, INFRARED):
                    # Context-managed: `Image.open` holds the decoder open until closed, and
                    # leaking one per image over 18,000 of them exhausts file handles long
                    # before the split finishes. `crop` calls `load()` first, so the copy
                    # survives its source being closed here.
                    with Image.open(io.BytesIO(bundle.read(by_stem[modality][stem]))) as source:
                        cropped[modality] = _crop_border(source, stem, modality)
            except zipfile.BadZipFile:
                # An upstream defect, not a download one -- see the module docstring. Recorded
                # and counted rather than raised: one unreadable JPEG must not cost the only
                # aerial dataset in the benchmark. TASKS.md X-4 -- never silently dropped.
                corrupt.append(stem)
                continue
            for modality, image in cropped.items():
                image.save(targets[modality], "PNG", optimize=True)
                image.close()
            written += 1
            if (index + 1) % 2000 == 0:
                logger.info("  %s: %d / %d pairs", split, index + 1, len(stems[VISIBLE]))

    if corrupt:
        logger.warning(
            "dronevehicle %s: skipped %d pair(s) whose upstream archive entry has a bad CRC-32: %s",
            split,
            len(corrupt),
            ", ".join(corrupt[:10]) + ("..." if len(corrupt) > 10 else ""),
        )

    # Measured from what is on disk rather than from the last image written, so a re-run over
    # an already-adapted tree reports the same inventory row as the run that created it
    # instead of an empty one.
    surviving = sorted(p.stem for p in (split_root / VISIBLE / "images").iterdir())
    if not surviving:
        raise AdapterError(
            f"dronevehicle: {split} produced no usable pairs from {archive.name}; "
            f"{len(corrupt)} of {len(stems[VISIBLE])} entries were unreadable"
        )
    with Image.open(images_dir(split_root, VISIBLE) / f"{surviving[0]}{SUFFIX}") as probe:
        resolution = f"{probe.width}x{probe.height}"
    return written, resolution, len(corrupt)


def adapt(raw_root: Path, dest_root: Path) -> Inventory:
    counts: dict[str, int] = {}
    skipped = 0
    resolution = ""
    for split, upstream in SPLIT_SOURCES.items():
        counts[split], observed, dropped = _adapt_split(
            raw_root / f"{upstream}.zip", upstream, split, dest_root
        )
        resolution = observed or resolution
        skipped += dropped

    note = (
        f"{BORDER_PX}px white border cropped from 840x712; re-encoded to PNG; "
        "upstream test split (8,980 pairs) not ingested"
    )
    if skipped:
        note += f"; {skipped} pair(s) dropped -- corrupt upstream archive entries"
    return Inventory(
        name="dronevehicle",
        domain="aerial",
        platform="drone",
        train_pairs=counts["train"],
        val_pairs=counts["val"],
        resolution=resolution,
        note=note,
    )


register("dronevehicle", adapt)
