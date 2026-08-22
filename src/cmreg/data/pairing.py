"""Optical <-> thermal path derivation by path-segment substitution.

Ported from ``../Thermal-To-Optical-Translation/src/t2o/data/pairing.py``. Its docstring
names the failure this avoids better than a summary can:

    Pairing by *sorted index* into two separate directory listings is the failure mode this
    avoids: a single missing or extra file silently shifts every subsequent pair by one and
    trains on mismatched modalities.

For a registration benchmark the consequence is worse than for training: a shifted pair does
not fail loudly, it produces a plausible-looking large registration error, and the method
gets blamed for the dataset's bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


class PairingError(ValueError):
    """Raised when a counterpart path cannot be derived unambiguously."""


def _substitute_segment(path: Path, old: str, new: str) -> Path:
    matches = [i for i, part in enumerate(path.parts) if part == old]
    if len(matches) != 1:
        raise PairingError(
            f"expected exactly one '{old}' path segment in '{path}' (found {len(matches)}); "
            f"cannot derive the '{new}' counterpart"
        )
    parts = list(path.parts)
    parts[matches[0]] = new
    return Path(*parts)


@dataclass(frozen=True, slots=True)
class Pairing:
    """Maps an optical image path to its thermal counterpart."""

    optical_token: str = "optical"
    thermal_token: str = "thermal"

    def thermal_path(self, optical: Path) -> Path:
        return _substitute_segment(optical, self.optical_token, self.thermal_token)

    def optical_path(self, thermal: Path) -> Path:
        """The inverse. Both directions exist because which modality is the reference frame
        is a per-experiment choice (PLAN.md §4.1 warps optical into thermal geometry, but the
        public benchmarks are indexed the other way)."""
        return _substitute_segment(thermal, self.thermal_token, self.optical_token)

    def validate_pairs(self, optical_paths: Iterable[Path]) -> None:
        """Raise if any optical image lacks a thermal counterpart.

        Reports the full count and a preview, not just the first failure -- a systematic
        problem (wrong token, half-synced directory) should be diagnosable in one run.
        """
        paths: Sequence[Path] = list(optical_paths)
        missing = [p for p in paths if not self.thermal_path(p).exists()]
        if missing:
            preview = ", ".join(str(self.thermal_path(p)) for p in missing[:5])
            more = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
            raise FileNotFoundError(
                f"{len(missing)}/{len(paths)} thermal counterparts missing: {preview}{more}"
            )
