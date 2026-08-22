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

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FILENAME = "pairs.parquet"


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
    matcher: str
    preprocess_ref: str
    preprocess_mov: str
    upsample: int
    interpolation: str
    estimator: str
    threshold_px: float
    warp: str
    moving: str
    seed: int
    config_hash: str
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
    # Null when the matcher reports no detection count: a dense matcher has no detection
    # stage, and `vismatch`'s interface cannot tell that apart from a detector that fired on
    # nothing (`matchers/vismatch_backend.py::_detected`). `0` therefore keeps meaning
    # "looked and found nothing", which is the distinction the column exists for.
    n_detected_ref: int | None
    n_detected_mov: int | None
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reproj_err: float | None
    extract_ms: float
    match_ms: float
    estimate_ms: float
    total_ms: float


_ARROW_TYPES: dict[str, pa.DataType] = {
    "str": pa.string(),
    "str | None": pa.string(),
    "int": pa.int64(),
    "int | None": pa.int64(),
    "float": pa.float64(),
    "float | None": pa.float64(),
    "bool": pa.bool_(),
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
    names = [field.name for field in fields(PairRow)]
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ResultsError(f"{source}: missing columns {missing}; written by an older schema?")
    records = table.select(names).to_pylist()
    return tuple(PairRow(**record) for record in records)


def concat(paths: Iterable[Path | str]) -> tuple[PairRow, ...]:
    """Read and concatenate several results files, for cross-run aggregation."""
    return tuple(row for path in paths for row in read_rows(path))
