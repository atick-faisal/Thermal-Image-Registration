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
N_TEXTURED = 4
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
