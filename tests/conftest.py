"""Synthetic fixtures. Nothing in the suite touches the network or the real dataset.

``dataset/`` is git-ignored wholesale (see .gitignore), so a fresh clone has no images and
every test has to build its own. The stem generation below is deliberately hostile in two
ways, both inherited from ``../Thermal-To-Optical-Translation/tests/conftest.py``:

* **Not alphabetical by index**, so a pairing bug that matches by sorted position instead of
  by path substitution produces mismatched modalities and fails loudly.
* **Disjoint between splits**, so train/val conflation shows up as a name collision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

N_TRAIN = 6
N_VAL = 4
IMAGE_SIZE = (64, 80)  # (height, width) -- non-square, so a transposed index is visible

TRAIN_STEMS = [f"{(7 * i) % 11:02d}_train_{i:03d}" for i in range(N_TRAIN)]
VAL_STEMS = [f"{(5 * i) % 11:02d}_val_{i:03d}" for i in range(N_VAL)]


def _write_pair(root: Path, split: str, stem: str, rng: np.random.Generator) -> None:
    height, width = IMAGE_SIZE
    optical = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    thermal = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    for modality, array, mode in (("optical", optical, "RGB"), ("thermal", thermal, "L")):
        directory = root / split / modality / "images"
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode=mode).save(directory / f"{stem}.png")


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-split paired optical/thermal dataset with its own ``data.yaml``."""
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    for split, stems in (("train", TRAIN_STEMS), ("val", VAL_STEMS)):
        for stem in stems:
            _write_pair(root, split, stem, rng)

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "optical", "thermal_token": "thermal"},
            }
        )
    )
    return root


@pytest.fixture(scope="session")
def manifest_path(dataset_root: Path) -> Path:
    return dataset_root / "data.yaml"


# --- textured fixtures -----------------------------------------------------------------
#
# The noise pairs above are built to break *pairing*; they are useless for matching, because a
# uniform-random image has no repeatable structure and SIFT finds nothing stable in it. The
# fixtures below exist for the opposite purpose: to give the evaluation cell something a
# classical detector can actually register, so a direction or scale error in the cell is
# visible as a large error rather than as a failed match.

TEXTURED_SIZE = (240, 320)  # (height, width)
# Six, not four: `train` takes the first two, leaving a four-pair `val` split. Anything
# smaller cannot express a *random* subsample of a split -- a two-pair val with limit 2 is
# the whole split however it was drawn, and the P3-1 selection test would pass vacuously.
N_TEXTURED = 6
TEXTURED_STEMS = [f"{(3 * i) % 7:02d}_tex_{i:03d}" for i in range(N_TEXTURED)]


def textured_image(rng: np.random.Generator, shape: tuple[int, int] = TEXTURED_SIZE):
    """A synthetic image with repeatable corner structure.

    Overlapping bright rectangles on a smooth gradient: plenty of corners at several scales,
    no periodicity (which would give a matcher ambiguous correspondences and make the test
    flaky for a reason unrelated to what it checks).
    """
    import cv2

    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    canvas = ((xs / width + ys / height) * 60.0).astype(np.float64)
    for _ in range(40):
        x0 = int(rng.integers(0, width - 20))
        y0 = int(rng.integers(0, height - 20))
        w = int(rng.integers(8, 40))
        h = int(rng.integers(8, 40))
        canvas[y0 : y0 + h, x0 : x0 + w] += float(rng.uniform(40, 160))
    canvas += rng.normal(0.0, 3.0, size=canvas.shape)
    blurred = cv2.GaussianBlur(np.clip(canvas, 0, 255).astype(np.uint8), (3, 3), 0.8)
    return np.asarray(blurred, dtype=np.uint8)


@pytest.fixture(scope="session")
def aligned_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A dataset whose two modalities are *identical* textured images.

    Deliberately degenerate: with both sides the same image, the only thing standing between
    the evaluation cell and a sub-pixel result is whether its warp directions, its inverse and
    its coordinate scaling are right. That is exactly what ``tests/test_runner.py`` asserts,
    and no cross-modal pair could isolate it -- a large error there is indistinguishable from
    the modality gap.
    """
    root = tmp_path_factory.mktemp("aligned")
    rng = np.random.default_rng(7)
    for split, stems in (("train", TEXTURED_STEMS[:2]), ("val", TEXTURED_STEMS[2:])):
        for stem in stems:
            image = textured_image(rng)
            for modality in ("optical", "thermal"):
                directory = root / split / modality / "images"
                directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image, mode="L").save(directory / f"{stem}.png")

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "optical", "thermal_token": "thermal"},
            }
        )
    )
    return root


# --- the offset fixture ----------------------------------------------------------------
#
# The two fixtures above cannot express the thing TASKS.md P1-1b measures. `aligned_dataset`
# has *identical* modalities, so a mono-modal control and a cross-modal cell read the same
# pixels and the flag under test has no observable effect; `dataset_root` is uniform noise and
# unmatchable by construction. This one stands in for a real capture rig: both modalities show
# the same scene, but the second camera is mounted a known number of pixels off.

OFFSET_PX = (5, 3)  # (dx, dy) -- the thermal camera's fixed displacement, in pixels
_OFFSET_MARGIN = 16  # crop inset, comfortably larger than the offset in both axes


@pytest.fixture(scope="session")
def offset_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A dataset whose modalities are two crops of one image, offset by :data:`OFFSET_PX`.

    Two overlapping crops rather than a shifted copy: a roll wraps content around the border
    and a pad invents it, and either would give the matcher edge structure that does not obey
    the transform the fixture claims to encode. Both crops here are wholly real content, and
    the exact relative transform between them is a translation by ``(dx, dy)``.

    Optical pixel ``(x, y)`` and thermal pixel ``(x - dx, y - dy)`` are the same scene point,
    so the map from thermal into optical -- the direction the evaluation cell estimates -- is a
    translation by ``(+dx, +dy)`` and the dataset's own residual misalignment is exactly
    ``hypot(dx, dy)`` px.
    """
    root = tmp_path_factory.mktemp("offset")
    rng = np.random.default_rng(11)
    height, width = TEXTURED_SIZE
    dx, dy = OFFSET_PX
    margin = _OFFSET_MARGIN

    for split, stems in (("train", TEXTURED_STEMS[:2]), ("val", TEXTURED_STEMS[2:])):
        for stem in stems:
            scene = textured_image(rng, (height + 2 * margin, width + 2 * margin))
            crops = {
                "optical": scene[margin : margin + height, margin : margin + width],
                "thermal": scene[
                    margin + dy : margin + dy + height, margin + dx : margin + dx + width
                ],
            }
            for modality, crop in crops.items():
                directory = root / split / modality / "images"
                directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop, mode="L").save(directory / f"{stem}.png")

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "optical", "thermal_token": "thermal"},
            }
        )
    )
    return root
