"""Raw public dataset -> the internal contract (TASKS.md P1-3).

One adapter per dataset, each registered under the name ``cmreg ingest`` takes. The registry
imports lazily per adapter, mirroring ``matchers/base.py``: an adapter whose raw tree is
absent must fail with a message naming the directory it looked for, not at import time and
not by taking the other adapters down with it.

Every adapter is expected to record, in its own module docstring, the **verified** raw layout
it was written against with exact file counts -- the discipline
``../Thermal-To-Optical-Translation/src/t2o/data/adapters/flir.py`` established. A layout
guessed from a README is the failure this project spends the most time paying for: it does not
crash, it silently pairs the wrong images, and the method gets blamed for the bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cmreg.data.adapters.common import (
    AdapterError,
    already_populated,
    images_dir,
    place_image_pair,
    write_image_pair_bytes,
    write_pointer_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterError",
    "Inventory",
    "adapt",
    "already_populated",
    "available",
    "images_dir",
    "place_image_pair",
    "register",
    "write_image_pair_bytes",
    "write_pointer_manifest",
]


@dataclass(frozen=True, slots=True)
class Inventory:
    """One row of the TASKS.md P1-1 inventory table.

    Returned by every adapter rather than recomputed by the caller, because only the adapter
    knows what it skipped and why. ``domain`` and ``platform`` are the ``EvalConfig`` values a
    benchmark run over this dataset must declare; carrying them here keeps a dataset's family
    with the dataset instead of in whichever YAML happened to be written first.
    """

    name: str
    domain: str
    platform: str
    train_pairs: int
    val_pairs: int
    # "1024x768", or "mixed" when the dataset is not single-resolution. Recorded because a
    # matcher's error is reported in pixels and pixels are not comparable across resolutions.
    resolution: str
    # Anything a reader of the table needs in order not to be misled by the numbers above:
    # the split the upstream set calls something else, images dropped, a crop applied.
    note: str = ""

    @property
    def pairs(self) -> int:
        return self.train_pairs + self.val_pairs


# `(raw_root, dest_root)` -> what was written. Both are dataset-specific directories, already
# joined with the dataset name by the caller, so an adapter never has to know its own name
# twice.
AdapterFn = Callable[[Path, Path], Inventory]
_REGISTRY: dict[str, AdapterFn] = {}


def register(name: str, adapter: AdapterFn) -> None:
    """Add an adapter under ``name``. Re-registering is an error: two adapters writing one
    dataset name would leave a tree that is half one layout and half the other."""
    if name in _REGISTRY:
        raise AdapterError(f"adapter {name!r} is already registered")
    _REGISTRY[name] = adapter


def available() -> tuple[str, ...]:
    """Every registered dataset name, sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def adapt(name: str, raw_root: Path, dest_root: Path) -> Inventory:
    """Run one adapter. ``raw_root`` is where the fetch script put the download."""
    _load_builtins()
    adapter = _REGISTRY.get(name)
    if adapter is None:
        raise AdapterError(f"unknown dataset {name!r}; available: {', '.join(available())}")
    if not raw_root.is_dir():
        raise AdapterError(
            f"{name}: raw dataset not found at {raw_root}. "
            f"Fetch it first: `uv run python scripts/fetch_datasets.py --dataset {name}`"
        )
    logger.info("adapting %s: %s -> %s", name, raw_root, dest_root)
    return adapter(raw_root, dest_root)


_BUILTINS_LOADED = False


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    # M3FD has no adapter yet, deliberately: Google Drive's per-file quota blocked its
    # download on 2026-08-24, so its raw layout is unverified. An adapter written from a
    # README is exactly the failure this module's docstring warns about, so it lands in the
    # same commit as the verified layout rather than ahead of it (TASKS.md P1-3).
    from cmreg.data.adapters import dronevehicle, llvip, sibling  # noqa: F401  (registers)
