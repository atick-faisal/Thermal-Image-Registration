"""Freeze and hash a dataset's train/val split membership.

Ported from ``../Thermal-To-Optical-Translation/src/t2o/data/splits.py``.

``dataset/`` is never tracked in git -- a fresh clone has no images at all -- so the images
themselves can never be the thing that proves a split hasn't drifted. What *can* be committed
is small: which filename stems belong to ``train`` vs ``val``, and a hash of that membership.
A re-fetch, an adapter re-run, or an upstream dataset revision that reshuffles files then
shows up as a loud, diffable ``git diff`` on ``splits/<name>.json`` (or a raised
:class:`SplitDriftError`) instead of a silent change nobody notices.

This is the mechanism behind TASKS.md P1-8 ("no capture-session leakage across splits") and
X-6 ("no test-set leakage into any training or hyperparameter decision").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmreg.data.manifest import IMAGE_SUFFIXES, DatasetManifest

# Matches Config.config_hash()'s sha256-hex-truncated convention (config/schema.py).
_HASH_LENGTH = 16
_PREVIEW = 5


class SplitDriftError(ValueError):
    """Raised when a dataset's current split membership no longer matches its frozen record."""


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A frozen record of which image stems belong to ``train`` vs ``val``."""

    name: str
    train_stems: tuple[str, ...]  # sorted
    val_stems: tuple[str, ...]  # sorted
    train_hash: str
    val_hash: str

    @property
    def combined_hash(self) -> str:
        """One fingerprint for the whole split -- changes if either half does."""
        return _hash_text(self.train_hash + self.val_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": {
                "count": len(self.train_stems),
                "hash": self.train_hash,
                "stems": list(self.train_stems),
            },
            "val": {
                "count": len(self.val_stems),
                "hash": self.val_hash,
                "stems": list(self.val_stems),
            },
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitManifest:
        return cls(
            name=data["name"],
            train_stems=tuple(data["train"]["stems"]),
            val_stems=tuple(data["val"]["stems"]),
            train_hash=data["train"]["hash"],
            val_hash=data["val"]["hash"],
        )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_HASH_LENGTH]


def _stems(images_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))


def freeze_split(name: str, manifest: DatasetManifest) -> SplitManifest:
    """Compute the current train/val stem membership and its hash for ``manifest``."""
    train_stems = _stems(manifest.train_images)
    val_stems = _stems(manifest.val_images)
    return SplitManifest(
        name=name,
        train_stems=train_stems,
        val_stems=val_stems,
        train_hash=_hash_text("\n".join(train_stems)),
        val_hash=_hash_text("\n".join(val_stems)),
    )


def write_split_manifest(record: SplitManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return path


def load_split_manifest(path: Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text()))


def verify_split(record: SplitManifest, manifest: DatasetManifest) -> None:
    """Raise :class:`SplitDriftError` if ``manifest``'s current membership no longer matches
    ``record`` -- the enforcement half of the frozen-split contract."""
    current = freeze_split(record.name, manifest)
    if current.train_hash != record.train_hash or current.val_hash != record.val_hash:
        raise SplitDriftError(_drift_message(record, current))


def _drift_message(record: SplitManifest, current: SplitManifest) -> str:
    parts = [f"{record.name}: split membership no longer matches the frozen record"]
    for split, before, after in (
        ("train", record.train_stems, current.train_stems),
        ("val", record.val_stems, current.val_stems),
    ):
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        if added or removed:
            parts.append(f"{split}: +{len(added)} -{len(removed)}")
            if added:
                parts.append(f"  added: {_preview(added)}")
            if removed:
                parts.append(f"  removed: {_preview(removed)}")
    return "; ".join(parts)


def _preview(stems: list[str]) -> str:
    return f"{', '.join(stems[:_PREVIEW])}{'...' if len(stems) > _PREVIEW else ''}"
