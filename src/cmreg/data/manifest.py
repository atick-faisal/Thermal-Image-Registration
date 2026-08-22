"""The dataset's own ``data.yaml``, read as the single source of truth.

Ported from ``../Thermal-To-Optical-Translation/src/t2o/data/manifest.py``, with the
detection-specific ``nc``/``names`` keys dropped (this project has no classes until the
Tier-2 box-transfer GT of TASKS.md P2-10) and the ``rgbt`` block's tokens renamed
``visible``/``infrared`` -> ``optical``/``thermal``.

Everything about a dataset -- where it lives, its splits, and which path segment names each
modality -- is declared in this one file. Re-declaring any of it in the experiment config
would create a second source of truth that can silently disagree with what is on disk, so
nothing here is configurable: point ``data.manifest`` at a ``data.yaml`` and that file decides.

Only ``path``/``train``/``val``/``rgbt`` are read; every other key is ignored silently, so a
manifest carrying leftover detection keys still loads.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from cmreg.data.pairing import Pairing

logger = logging.getLogger(__name__)

RGBT_BLOCK = "rgbt"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


class ManifestError(ValueError):
    """Raised when a ``data.yaml`` is missing, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """A resolved ``data.yaml``. All paths are absolute."""

    path: Path  # the data.yaml itself
    root: Path
    train_images: Path
    val_images: Path
    pairing: Pairing

    @classmethod
    def load(cls, path: Path | str) -> DatasetManifest:
        path = Path(path)
        if path.is_dir():
            # A directory is very likely a dataset root; point at its manifest rather than
            # failing on something the caller obviously meant.
            path = path / "data.yaml"
        if not path.is_file():
            raise ManifestError(f"dataset manifest not found: {path}")

        try:
            declared = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ManifestError(f"{path}: could not be parsed: {exc}") from exc
        if not isinstance(declared, Mapping):
            raise ManifestError(f"{path}: expected a mapping at the top level")

        root = _resolve_root(path, declared)
        train = _split_dir(path, root, declared, "train")
        val = _split_dir(path, root, declared, "val")

        block = declared.get(RGBT_BLOCK) or {}
        if not isinstance(block, Mapping):
            raise ManifestError(f"{path}: '{RGBT_BLOCK}' must be a mapping")
        pairing = Pairing(
            optical_token=str(block.get("optical_token", Pairing.optical_token)),
            thermal_token=str(block.get("thermal_token", Pairing.thermal_token)),
        )
        if pairing.optical_token == pairing.thermal_token:
            raise ManifestError(f"{path}: optical_token and thermal_token must differ")

        logger.info(
            "%s: root %s, tokens %s/%s",
            path.name,
            root,
            pairing.optical_token,
            pairing.thermal_token,
        )
        return cls(
            path=path.resolve(), root=root, train_images=train, val_images=val, pairing=pairing
        )

    def images(self, split: str) -> tuple[Path, ...]:
        """The optical images of ``split``, sorted.

        Sorted for reproducibility only -- pairing never uses the ordering (see
        ``data/pairing.py``); it derives each thermal counterpart from its optical path.
        """
        if split not in {"train", "val"}:
            raise ManifestError(f"unknown split {split!r}; expected 'train' or 'val'")
        directory = self.train_images if split == "train" else self.val_images
        return tuple(sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))


def _resolve_root(manifest: Path, declared: Mapping) -> Path:
    """Find the dataset root a ``path:`` entry refers to.

    ``path`` is conventionally relative to wherever the project is run from, not to the
    manifest, so the same file resolves differently depending on the caller. Rather than
    guess one convention, try each and take the first that actually contains the declared
    training split.
    """
    train = declared.get("train")
    candidates: list[Path] = []

    raw = declared.get("path")
    if raw is not None:
        declared_path = Path(str(raw))
        if declared_path.is_absolute():
            candidates.append(declared_path)
        else:
            candidates.append(manifest.parent / declared_path)
            candidates.append(Path.cwd() / declared_path)
    candidates.append(manifest.parent)

    for candidate in candidates:
        if train is None or (candidate / str(train)).exists():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise ManifestError(
        f"{manifest}: could not locate the dataset root containing '{train}'. Tried:\n  {tried}"
    )


def _split_dir(manifest: Path, root: Path, declared: Mapping, split: str) -> Path:
    if split not in declared:
        raise ManifestError(f"{manifest}: missing required '{split}' split")
    directory = (root / str(declared[split])).resolve()
    if not directory.is_dir():
        raise ManifestError(f"{manifest}: '{split}' points at {directory}, which does not exist")
    return directory
