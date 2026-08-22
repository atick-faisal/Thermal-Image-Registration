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
