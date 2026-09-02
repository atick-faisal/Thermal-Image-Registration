"""The per-pair results store (TASKS.md P0-8).

TASKS.md §0 fixes the division of labour: **the source of truth for per-pair results is local
Parquet, not W&B.** W&B Tables degrade past a few tens of thousands of rows and the staged
factorial of P3-1 produces millions, so W&B receives summary metrics and the full rows stay on
disk, to be read back by the Phase-8 aggregator.

Because those rows never leave the machine that produced them, the *text* channel matters:
``results/report.py`` renders a copy-pasteable block, and ``cmreg report`` re-renders it from a
Parquet file without re-running anything.

**The Arrow schema is declared, never inferred.** A run in which every pair failed has
all-null measurement columns, and pyarrow would infer those as ``null`` dtype -- the file then
refuses to concatenate with a successful run's, and the failure surfaces at aggregation time,
months later, as a type error nobody can trace back to the run that caused it.

One row is one ``(pair, matcher, preprocessing, estimator, warp, seed)`` cell. Every axis is a
column rather than being folded into a run name, so the aggregator groups by them directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FILENAME = "pairs.parquet"

# A 3x3 homography, row-major -- the convention `cmreg gt` already writes into `gt_*.json`.
_HOMOGRAPHY_ENTRIES = 9


class ResultsError(ValueError):
    """Raised when a results file is missing, empty, or does not match the declared schema."""


@dataclass(frozen=True, slots=True)
class PairRow:
    """One evaluated pair under one configuration.

    Field order is the column order in the file. Identity columns first, then measurements --
    the two halves are read for different reasons and keeping them adjacent makes a ``head``
    of the file legible.
    """

    # --- identity: what was run -------------------------------------------------------
    stem: str
    dataset: str
    split: str
    domain: str
    platform: str
    # The reference image's size, in pixels. Carried because nothing spatial is recomputable
    # from this file without it -- `corner_error` and `diagonal` both need the shape, and the
    # aggregator has no image to ask (TASKS.md P1-1b). Null only for a pair that could not be
    # decoded at all, which is the one case where there is no shape to record.
    height: int | None
    width: int | None
    matcher: str
    preprocess_ref: str
    preprocess_mov: str
    upsample: int
    interpolation: str
    # Stage F's resolution level (TASKS.md P3-12a): the factor **both** sides were resized by
    # before matching. Nullable, and the null means "written before this axis existed" -- every
    # stage A-E file on the server predates it, `read_rows` backfills a missing nullable column
    # and a missing non-nullable one is fatal (see below). Writing 1.0 into those rows instead
    # would be inventing a value the file never recorded, even though it is the one they ran at.
    #
    # Note this is *not* the resolution the row's metrics are in: `height`/`width` above are
    # native and so is every error column, because keypoints are mapped back before estimation
    # (`eval/runner.py::_evaluate`). It is the resolution the *matcher* saw.
    input_scale: float | None
    estimator: str
    threshold_px: float
    warp: str
    # Stage F's match-count axis (TASKS.md P3-12c): the cap on how many correspondences reached
    # the solver, and which ones. `0` is "no cap"; null means "written before this axis existed",
    # which is every stage A-F file on the server and is why both are nullable (the P2-12 rule).
    # Per row rather than per run, unlike `input_scale` above: this axis is *downstream* of the
    # matcher, so a cap costs one `cv2.findHomography` call off a shared `MatchResult` and a
    # single run directory holds every cap.
    max_matches: int | None
    match_selection: str | None
    # How many correspondences the solver actually saw -- `min(max_matches, n_matches)`, or
    # `n_matches` when uncapped. Carried separately because it is the denominator of
    # `inlier_ratio` below and `n_matches` is not: without it a capped row's inlier ratio cannot
    # be reproduced from the file.
    n_selected: int | None
    moving: str
    # The reference modality. Without it a mono-modal control row (TASKS.md P1-1b, where both
    # sides come from one modality) is indistinguishable from a benchmark row in this file.
    reference: str
    seed: int
    config_hash: str
    # Digest of the dataset residual `R` composed into this row's ground truth, or null for a
    # run that composed none (TASKS.md P2-12). `config_hash` covers the configured *path*, and
    # two different constants can share a filename across two machines; this is the column that
    # ties a number to the exact matrix. It also makes a composed row impossible to confuse
    # with a pre-composition one when the two sit in the same aggregate.
    residual_calibration: str | None
    git_sha: str
    run_name: str

    # --- measurements: what happened --------------------------------------------------
    # False when no homography could be fitted. Every nullable column below is null exactly
    # when this is False, which is what makes `failure_rate` recoverable from the file alone.
    success: bool
    failure_reason: str | None
    # Fraction of source pixels landing inside the target canvas under the GT warp. A property
    # of the pair, not of the method, so it is present even on failures.
    overlap: float
    corner_err: float | None
    epe_mean: float | None
    epe_median: float | None
    # The estimated homography, row-major 9 floats -- the same convention `cmreg gt` writes
    # into `gt_*.json` (`cli.py::_run_gt`). Null exactly when `success` is False.
    #
    # Stored because a corner error is lossy: it says how far the fit was, never in which
    # direction. TASKS.md P1-1b needs the direction to tell a fixed rig miscalibration (every
    # pair sharing one offset) from a cross-modal localisation limit (per-pair scatter), and
    # P3-1's "estimate R per pair and compose it into the GT" option cannot be implemented at
    # all without it. ~72 B/row, against having to re-run every matcher to recover it.
    h: list[float] | None
    # Null when the matcher reports no detection count: a dense matcher has no detection
    # stage, and `vismatch`'s interface cannot tell that apart from a detector that fired on
    # nothing (`matchers/vismatch_backend.py::_detected`). `0` therefore keeps meaning
    # "looked and found nothing", which is the distinction the column exists for.
    n_detected_ref: int | None
    n_detected_mov: int | None
    # The matcher's own yield, **not** what was fitted. Unchanged by P3-12c's cap on purpose:
    # `match/total` is a property of the matcher and stage A's column has to keep meaning what it
    # meant. `n_selected` above is what reached the solver.
    n_matches: int
    n_inliers: int
    # Inliers over `n_selected`, not over `n_matches`. The two are equal on every uncapped row --
    # which is everything measured before P3-12c -- and diverge exactly where the cap bites.
    inlier_ratio: float
    reproj_err: float | None
    extract_ms: float
    match_ms: float
    estimate_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        # The width guarantee the Arrow type cannot carry; see `_ARROW_TYPES` for why.
        if self.h is not None and len(self.h) != _HOMOGRAPHY_ENTRIES:
            raise ResultsError(
                f"h must be {_HOMOGRAPHY_ENTRIES} row-major floats, got {len(self.h)}"
            )


_ARROW_TYPES: dict[str, pa.DataType] = {
    "str": pa.string(),
    "str | None": pa.string(),
    "int": pa.int64(),
    "int | None": pa.int64(),
    "float": pa.float64(),
    "float | None": pa.float64(),
    "bool": pa.bool_(),
    # Variable-size, though a homography is always 9 floats. `pa.list_(pa.float64(), 9)` is
    # the honest type and it *cannot be used*: Parquet has no fixed-size-list logical type, so
    # a null is written as a zero-length list and `pq.read_table` then rejects its own file
    # with "Expected all lists to be of size=9 but index N had size=0". Nulls are not the
    # exception here -- every failed pair has one. The width is enforced in
    # `PairRow.__post_init__` instead, which catches a malformed row earlier than the writer
    # would have anyway.
    "list[float] | None": pa.list_(pa.float64()),
}


def schema() -> pa.Schema:
    """The declared Arrow schema, derived from :class:`PairRow`'s annotations.

    Derived rather than hand-written so the dataclass and the file cannot drift apart; an
    annotation the mapping does not cover fails here, at import, rather than producing a
    column of the wrong type.
    """
    columns = []
    for field in fields(PairRow):
        arrow = _ARROW_TYPES.get(str(field.type))
        if arrow is None:
            raise ResultsError(
                f"PairRow.{field.name} is annotated {field.type!r}, which has no declared "
                f"Arrow type; add one to _ARROW_TYPES"
            )
        columns.append(pa.field(field.name, arrow, nullable="None" in str(field.type)))
    return pa.schema(columns)


def to_table(rows: Sequence[PairRow]) -> pa.Table:
    """Materialise rows against the declared schema."""
    if not rows:
        raise ResultsError("cannot build a results table from zero rows")
    records = [asdict(row) for row in rows]
    columns = {name: [record[name] for record in records] for name in records[0]}
    return pa.Table.from_pydict(columns, schema=schema())


def write_rows(rows: Sequence[PairRow], path: Path | str) -> Path:
    """Write rows to ``path`` (a file, or a directory in which to place ``pairs.parquet``)."""
    target = Path(path)
    if target.suffix != ".parquet":
        target = target / FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), target)
    logger.info("wrote %d result rows to %s", len(rows), target)
    return target


def read_rows(path: Path | str) -> tuple[PairRow, ...]:
    """Read rows back. The inverse of :func:`write_rows`, and the aggregator's entry point."""
    source = Path(path)
    if source.is_dir():
        source = source / FILENAME
    if not source.is_file():
        raise ResultsError(f"results file not found: {source}")

    table = pq.read_table(source)
    declared = {field.name: str(field.type) for field in fields(PairRow)}
    absent = [name for name in declared if name not in table.column_names]
    # A **nullable** column absent from an older file is read as null rather than refused. The
    # project adds columns as it learns what to record -- `h` and the shape at P1-1b,
    # `residual_calibration` at P2-12 -- and every one of those was declared `X | None`
    # precisely because the value can be unknown. A file written before the column existed is
    # the purest case of unknown, so refusing it would make each addition retroactively destroy
    # every run on disk, which is exactly what P1-1b promised additive columns would not do.
    # A missing *non*-nullable column is still fatal: there is no honest value to invent.
    fillable = [name for name in absent if "None" in declared[name]]
    fatal = [name for name in absent if name not in fillable]
    if fatal:
        raise ResultsError(f"{source}: missing columns {fatal}; written by an older schema?")
    if fillable:
        logger.info(
            "%s predates columns %s; reading them as null", source, ", ".join(sorted(fillable))
        )
    # Typed `dict[str, Any]`, not the `dict[str, None]` `fromkeys` infers, so that merging it
    # into a record does not widen every field to `X | None` for the type checker.
    blank: dict[str, Any] = dict.fromkeys(fillable)
    present = [name for name in declared if name in table.column_names]
    records = table.select(present).to_pylist()
    return tuple(PairRow(**{**blank, **record}) for record in records)


def concat(paths: Iterable[Path | str]) -> tuple[PairRow, ...]:
    """Read and concatenate several results files, for cross-run aggregation."""
    return tuple(row for path in paths for row in read_rows(path))
