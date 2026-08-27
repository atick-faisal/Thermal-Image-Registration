"""A dataset's residual cross-modal misalignment, frozen as a published constant.

Tier-1 ground truth assumes a pair *starts* aligned: ``eval/runner.py`` warps the moving
modality by ``H_gt`` and scores the matcher against ``inv(H_gt)``. TASKS.md P1-1a measured
that assumption and it is false on every public benchmark -- MSRS 4.7 px, FLIR-aligned 5.9 px,
LLVIP 4.2 px, DroneVehicle 4.7 px median under a 28% gross-failure rate -- so the true
correspondence is ``R . inv(H_gt)`` and ``R`` is a systematic floor under every Tier-1 number.
P3-7's stage-A run 1 then measured what that costs: ten of twenty matchers scored
``success_rate_5px`` of exactly 0.0000 on ``flir``, and twelve structurally unrelated matchers
piled into a 4.5-5.8 px band that is the dataset's own residual rather than anything about
them. This module is the constant that removes it.

**Why a corner field and not "1.28 degrees of roll".** P1-1d estimated ``R`` under three
matchers spanning three decades of cost. The *total* is well determined -- the corner fields
agree to 1.23 px mean, 2.33 px worst case -- but its decomposition into rotation and scale is
not: the rotation disagrees by 38% and the first principal scale flips sign between matchers,
because over a limited field of view a small roll and a small scale anisotropy generate nearly
the same corner displacement. So the publishable object is the four corner displacements, and
the degrees are a reading of them that must never be the thing that is stored.

**Why dataset-level and never per pair.** P1-1b: ``R_i`` comes from a matcher, so folding it
into the ground truth scores that matcher against its own output. Only one matrix, estimated
once from several matchers and published with its across-matcher spread, is admissible --
which is what ``spread_px`` and ``worst_case_px`` exist to carry, and why they are required
fields rather than optional metadata. ``experiments/GRID.md`` §3 holds the per-dataset policy
for which sets may compose at all; this module does not decide that, it only represents it.

The file layout follows ``data/splits.py``: a small tracked JSON under a top-level directory,
a frozen dataclass with ``from_dict``, and a layer-specific error that names what drifted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from cmreg.warp import corners

FloatArray = NDArray[np.float64]

# Matches `Config.config_hash()` and `data/splits.py` (config/schema.py).
_HASH_LENGTH = 16
_CORNERS = 4
# Clockwise from the top-left, the order `warp.corners` produces. Named so a console block and
# a JSON file cannot disagree about which pair of numbers is which corner.
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


class CalibrationError(ValueError):
    """Raised for a malformed calibration, or one applied to data it does not describe."""


@dataclass(frozen=True, slots=True)
class ResidualCalibration:
    """The one homography a dataset's two cameras are offset by, as four corner shifts."""

    dataset: str
    height: int
    width: int
    # Four (dx, dy) displacements in pixels, clockwise from the top-left, carrying a reference
    # image corner to where the moving modality's content actually lands. This *is* the
    # constant; everything else on this class is provenance.
    corner_shift: tuple[tuple[float, float], ...]
    # The matchers the element-wise median was taken over. One leg is not a calibration -- it
    # is that matcher's bias, and P1-1d measured roma's at up to 3.44 px against a two-matcher
    # cluster. Recorded so a reader can see how many witnesses the constant has.
    matchers: tuple[str, ...]
    # Mean and worst-case distance of a leg from the published median, in pixels. This is the
    # constant's stated uncertainty and the number that decides whether composing it is a net
    # win: P1-1d's 1.23 px mean is 30% of the 4.09 px per-pair scatter it must be small
    # against, where one matcher's own `R_bar` was 84% of it.
    spread_px: float
    worst_case_px: float
    n_pairs: int
    split: str
    git_sha: str
    note: str

    def __post_init__(self) -> None:
        if len(self.corner_shift) != _CORNERS:
            raise CalibrationError(
                f"{self.dataset}: a corner field has {_CORNERS} corners, got "
                f"{len(self.corner_shift)}"
            )
        if not self.matchers:
            raise CalibrationError(
                f"{self.dataset}: a calibration records the matchers it was estimated from, "
                "and one leg is a matcher's bias rather than a constant; got none"
            )
        if self.height <= 0 or self.width <= 0:
            raise CalibrationError(
                f"{self.dataset}: shape must be positive, got {self.height}x{self.width}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``, the convention every array and metric in this repo uses."""
        return (self.height, self.width)

    def homography(self) -> FloatArray:
        """The constant as a matrix: refit from the four displaced corners.

        Reuses ``warp.corners`` rather than restating ``[[0,0],[w,0],[w,h],[0,h]]``, because
        that function's docstring records the one thing that would silently halve a scale
        term -- the corners are the outer edges of the sampling grid, not ``w-1``/``h-1``.
        """
        reference = corners(self.shape)
        shifted = reference + np.asarray(self.corner_shift, dtype=np.float64)
        return np.asarray(
            cv2.getPerspectiveTransform(reference.astype(np.float32), shifted.astype(np.float32)),
            dtype=np.float64,
        )

    def magnitude_px(self) -> float:
        """Mean corner displacement -- how much misalignment composing this removes."""
        return float(np.linalg.norm(np.asarray(self.corner_shift, dtype=np.float64), axis=1).mean())

    def digest(self) -> str:
        """A fingerprint of the *constant*, not of the file that carries it.

        Only the dataset, the shape and the four displacements go in: two files agreeing on
        those describe the same calibration whatever their notes say, and a results row
        stamped with this digest is traceable to the exact matrix that produced it. That is
        the half ``config_hash()`` cannot do -- it hashes the configured *path*, and two
        different constants can share a filename across two machines.
        """
        payload = json.dumps(
            {
                "dataset": self.dataset,
                "height": self.height,
                "width": self.width,
                "corner_shift": [[round(dx, 6), round(dy, 6)] for dx, dy in self.corner_shift],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]

    def validate_for(self, dataset: str, shape: tuple[int, int]) -> None:
        """Raise unless this constant describes ``dataset`` at ``shape``.

        Both checks are fatal rather than accommodating. A corner field measured at 640x512
        and applied to a 1280x1024 pair is silently *half* the misalignment it claims to be,
        and a constant composed for the wrong dataset removes an offset that was never there
        while adding one that is -- neither fails loudly on its own, and both would show up
        months later as an inexplicable table.
        """
        if dataset != self.dataset:
            raise CalibrationError(
                f"calibration is for '{self.dataset}' but the run is on '{dataset}'"
            )
        if shape != self.shape:
            raise CalibrationError(
                f"{self.dataset}: calibration was measured at {self.height}x{self.width} but "
                f"the pair is {shape[0]}x{shape[1]}; a corner field does not rescale"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "height": self.height,
            "width": self.width,
            "corner_shift": [[dx, dy] for dx, dy in self.corner_shift],
            "matchers": list(self.matchers),
            "spread_px": self.spread_px,
            "worst_case_px": self.worst_case_px,
            "n_pairs": self.n_pairs,
            "split": self.split,
            "git_sha": self.git_sha,
            "note": self.note,
            # Written for the reader and for `git diff`, never read back: `from_dict` ignores
            # it, so a hand-edited corner cannot be hidden behind a stale digest.
            "digest": self.digest(),
            "magnitude_px": self.magnitude_px(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualCalibration:
        try:
            return cls(
                dataset=data["dataset"],
                height=int(data["height"]),
                width=int(data["width"]),
                corner_shift=tuple((float(dx), float(dy)) for dx, dy in data["corner_shift"]),
                matchers=tuple(data["matchers"]),
                spread_px=float(data["spread_px"]),
                worst_case_px=float(data["worst_case_px"]),
                n_pairs=int(data["n_pairs"]),
                split=data["split"],
                git_sha=data["git_sha"],
                note=data["note"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError(f"malformed calibration record: {error}") from error


def load_calibration(path: Path | str) -> ResidualCalibration:
    """Read a calibration JSON. Raises :class:`CalibrationError` rather than ``OSError``."""
    location = Path(path)
    try:
        data = json.loads(location.read_text())
    except OSError as error:
        raise CalibrationError(f"cannot read calibration {location}: {error}") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{location} is not valid JSON: {error}") from error
    return ResidualCalibration.from_dict(data)


def write_calibration(record: ResidualCalibration, path: Path | str) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return location
