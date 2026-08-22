"""The per-pair results store and the text channel out of it (TASKS.md P0-8)."""

from __future__ import annotations

from cmreg.results.report import ReportError, Summary, render, render_comparison, summarize
from cmreg.results.store import (
    FILENAME,
    PairRow,
    ResultsError,
    concat,
    read_rows,
    schema,
    to_table,
    write_rows,
)

__all__ = [
    "FILENAME",
    "PairRow",
    "ReportError",
    "ResultsError",
    "Summary",
    "concat",
    "read_rows",
    "render",
    "render_comparison",
    "schema",
    "summarize",
    "to_table",
    "write_rows",
]
