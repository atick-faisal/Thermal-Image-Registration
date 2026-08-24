"""The dataset adapters (TASKS.md P1-3).

Every test here builds its own raw tree, so the suite still touches neither the network nor
the 18 GB of real data. What each fixture reproduces is the *shape* of a layout verified
against the genuine archive -- the DroneVehicle border and its ``img``/``imgr`` directories,
LLVIP's ``visible``/``infrared`` split directories -- because the adapters' whole job is to
fail loudly when a layout is not what its docstring says it is.

The load-bearing assertions are the negative ones. An adapter that silently mis-pairs two
modalities does not crash; it produces a plausible-looking registration error and the method
gets blamed for the dataset's bookkeeping (``data/pairing.py``).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cmreg.data import DatasetManifest
from cmreg.data.adapters import AdapterError, adapt, available, register
from cmreg.data.adapters.common import write_pointer_manifest
from cmreg.data.adapters.dronevehicle import BORDER_PX

CORE = (40, 50)  # (height, width) of the real content, non-square so a transpose is visible
STEMS = ("00001", "00002", "00003")


def _image(rng: np.random.Generator, shape: tuple[int, int], border: int = 0) -> Image.Image:
    """Content of ``shape``, optionally inside a white margin of ``border`` px per side."""
    height, width = shape
    content = rng.integers(0, 200, size=(height, width, 3), dtype=np.uint8)
    if border == 0:
        return Image.fromarray(content, mode="RGB")
    canvas = np.full((height + 2 * border, width + 2 * border, 3), 255, dtype=np.uint8)
    canvas[border:-border, border:-border] = content
    return Image.fromarray(canvas, mode="RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


# --- DroneVehicle ------------------------------------------------------------------------


def _dronevehicle_raw(
    root: Path,
    border: int = BORDER_PX,
    drop_infrared_stem: bool = False,
    core: tuple[int, int] = CORE,
) -> Path:
    """The verified layout: ``<split>/<split>img`` (RGB) and ``<split>imgr`` (infrared)."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    for upstream in ("train", "val"):
        with zipfile.ZipFile(root / f"{upstream}.zip", "w") as bundle:
            for directory in (f"{upstream}img", f"{upstream}imgr"):
                for index, stem in enumerate(STEMS):
                    if drop_infrared_stem and directory.endswith("r") and index == 0:
                        continue
                    bundle.writestr(
                        f"{upstream}/{directory}/{stem}.jpg",
                        _jpeg_bytes(_image(rng, core, border=border)),
                    )
    return root


def test_dronevehicle_crops_the_published_border(tmp_path: Path) -> None:
    """The 100 px white margin is upstream packaging, not content. Getting this wrong in
    either direction -- not cropping, or cropping an image that has no border -- shifts every
    coordinate in the dataset by 100 px."""
    raw = _dronevehicle_raw(tmp_path / "raw")
    inventory = adapt("dronevehicle", raw, tmp_path / "out")

    assert inventory.train_pairs == len(STEMS)
    assert inventory.val_pairs == len(STEMS)
    assert inventory.resolution == f"{CORE[1]}x{CORE[0]}"
    assert inventory.domain == "aerial"
    with Image.open(tmp_path / "out" / "val" / "visible" / "images" / f"{STEMS[0]}.png") as img:
        assert img.size == (CORE[1], CORE[0])


def test_dronevehicle_refuses_to_crop_an_image_with_no_border(tmp_path: Path) -> None:
    """If a future release drops the margin, a blind crop would discard 100 px of real content
    on every side and surface months later as an inexplicable per-dataset offset.

    The content here is deliberately larger than twice the border, so the adapter *could*
    crop it -- otherwise this would only be testing the size check below it."""
    raw = _dronevehicle_raw(
        tmp_path / "raw", core=(2 * BORDER_PX + 20, 2 * BORDER_PX + 30), border=0
    )
    with pytest.raises(AdapterError, match=r"no white .* border"):
        adapt("dronevehicle", raw, tmp_path / "out")


def test_dronevehicle_refuses_an_image_too_small_to_hold_the_border(tmp_path: Path) -> None:
    raw = _dronevehicle_raw(tmp_path / "raw", core=(10, 12), border=0)
    with pytest.raises(AdapterError, match="too small"):
        adapt("dronevehicle", raw, tmp_path / "out")


def test_dronevehicle_refuses_a_stem_mismatch(tmp_path: Path) -> None:
    """Pairing by position is the failure `data/pairing.py` exists to prevent; an adapter that
    lets a half-populated archive through moves that failure into the results store."""
    raw = _dronevehicle_raw(tmp_path / "raw", drop_infrared_stem=True)
    with pytest.raises(AdapterError, match="stems differ"):
        adapt("dronevehicle", raw, tmp_path / "out")


def _corrupt_entry(archive: Path, name: str) -> None:
    """Flip a byte inside one stored entry's payload so reading it raises ``BadZipFile``.

    Reproduces the real defect: ``train.zip`` as published carries 39 entries out of 35,980
    whose CRC-32 does not match their data (see the adapter's module docstring). The local
    file header is parsed rather than guessed at, so the flip lands in the payload and not in
    the metadata -- corrupting the header would raise a different error and test nothing.
    """
    with zipfile.ZipFile(archive) as bundle:
        offset = bundle.getinfo(name).header_offset
    raw = bytearray(archive.read_bytes())
    name_len = int.from_bytes(raw[offset + 26 : offset + 28], "little")
    extra_len = int.from_bytes(raw[offset + 28 : offset + 30], "little")
    payload = offset + 30 + name_len + extra_len
    raw[payload + 40] ^= 0xFF
    archive.write_bytes(bytes(raw))


def test_dronevehicle_drops_and_counts_a_corrupt_pair(tmp_path: Path) -> None:
    """39 of DroneVehicle's 35,980 train entries are unreadable in the archive as published.
    Raising would cost this project its only aerial dataset over 0.11% of the images; dropping
    silently would violate X-4. So: drop the *pair*, count it, and say so in the note."""
    raw = _dronevehicle_raw(tmp_path / "raw")
    _corrupt_entry(raw / "val.zip", f"val/valimgr/{STEMS[0]}.jpg")

    inventory = adapt("dronevehicle", raw, tmp_path / "out")

    assert inventory.val_pairs == len(STEMS) - 1
    assert inventory.train_pairs == len(STEMS)
    assert "1 pair(s) dropped" in inventory.note
    # Neither side of the dropped pair may survive: a lone visible image is unpairable, and
    # `validate_pairs` would then reject the whole split.
    for modality in ("visible", "infrared"):
        images = tmp_path / "out" / "val" / modality / "images"
        assert not (images / f"{STEMS[0]}.png").exists()

    manifest = write_pointer_manifest(tmp_path / "pointer" / "data.yaml", tmp_path / "out")
    loaded = DatasetManifest.load(manifest)
    loaded.pairing.validate_pairs(loaded.images("val"))


# --- LLVIP -------------------------------------------------------------------------------


def _llvip_raw(root: Path) -> tuple[Path, dict[str, bytes]]:
    rng = np.random.default_rng(1)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, bytes] = {}
    with zipfile.ZipFile(root / "LLVIP.zip", "w") as bundle:
        for modality in ("visible", "infrared"):
            for upstream in ("train", "test"):
                for stem in STEMS:
                    name = f"LLVIP/{modality}/{upstream}/{stem}.jpg"
                    data = _jpeg_bytes(_image(rng, CORE))
                    bundle.writestr(name, data)
                    written[name] = data
    return root, written


def test_llvip_copies_bytes_without_re_encoding(tmp_path: Path) -> None:
    """LLVIP's own JPEG is what the benchmark should measure. Re-encoding would add a second
    lossy pass, and JPEG quality is a degradation axis this project controls deliberately
    (TASKS.md P2-3) rather than inherits by accident."""
    raw, written = _llvip_raw(tmp_path / "raw")
    inventory = adapt("llvip", raw, tmp_path / "out")

    assert (inventory.train_pairs, inventory.val_pairs) == (len(STEMS), len(STEMS))
    adapted = tmp_path / "out" / "val" / "visible" / "images" / f"{STEMS[0]}.jpg"
    assert adapted.read_bytes() == written[f"LLVIP/visible/test/{STEMS[0]}.jpg"]


def test_llvip_maps_the_upstream_test_split_to_val(tmp_path: Path) -> None:
    """Upstream publishes train/test; this project's manifest declares train/val and evaluates
    on the held-out one. A silent mapping to `train` would be test-set leakage (X-6)."""
    raw, written = _llvip_raw(tmp_path / "raw")
    adapt("llvip", raw, tmp_path / "out")
    val = tmp_path / "out" / "val" / "infrared" / "images" / f"{STEMS[1]}.jpg"
    assert val.read_bytes() == written[f"LLVIP/infrared/test/{STEMS[1]}.jpg"]


# --- the contract every adapter has to satisfy --------------------------------------------


@pytest.mark.parametrize("name", ["dronevehicle", "llvip"])
def test_an_adapted_tree_loads_through_the_manifest_and_pairs(name: str, tmp_path: Path) -> None:
    """The single assertion that matters: whatever an adapter writes must be readable by the
    same `DatasetManifest` every benchmark run goes through, with every pair resolvable."""
    raw = tmp_path / "raw"
    if name == "dronevehicle":
        _dronevehicle_raw(raw)
    else:
        _llvip_raw(raw)
    out = tmp_path / "out"
    adapt(name, raw, out)

    manifest = write_pointer_manifest(tmp_path / "pointer" / "data.yaml", out)
    loaded = DatasetManifest.load(manifest)
    assert loaded.root == out.resolve()
    for split in ("train", "val"):
        images = loaded.images(split)
        assert len(images) == len(STEMS)
        loaded.pairing.validate_pairs(images)


def test_adapting_twice_is_a_no_op(tmp_path: Path) -> None:
    """`cmreg ingest` is re-run whenever the inventory is refreshed, and DroneVehicle takes
    tens of minutes to re-encode. A second pass must not redo the work or change the output."""
    raw = _dronevehicle_raw(tmp_path / "raw")
    out = tmp_path / "out"
    first = adapt("dronevehicle", raw, out)
    target = out / "val" / "visible" / "images" / f"{STEMS[0]}.png"
    stamp = target.stat().st_mtime_ns
    second = adapt("dronevehicle", raw, out)
    assert first == second
    assert target.stat().st_mtime_ns == stamp


def test_an_unknown_dataset_names_the_known_ones(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(AdapterError, match="unknown dataset"):
        adapt("metu-vistir", tmp_path / "raw", tmp_path / "out")


def test_a_missing_raw_tree_says_how_to_fetch_it(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="fetch_datasets"):
        adapt("llvip", tmp_path / "absent", tmp_path / "out")


def test_registering_a_duplicate_name_is_refused() -> None:
    """Two adapters under one name would leave a tree half in each layout."""
    assert "llvip" in available()
    with pytest.raises(AdapterError, match="already registered"):
        register("llvip", lambda raw, dest: None)  # type: ignore[arg-type,return-value]


def test_the_sibling_adapter_says_what_it_looked_for(tmp_path: Path) -> None:
    """MSRS and FLIR-aligned are adapted by the sibling project; this one only verifies and
    counts. A missing tree must name the directory rather than report zero pairs."""
    (tmp_path / "raw").mkdir()
    with pytest.raises(AdapterError, match="already adapted"):
        adapt("msrs", tmp_path / "raw", tmp_path / "out")
