"""The stage-A driver's resume guard (`scripts/p3a_grid.py`).

Only the guard is tested here. The rest of the driver is a `cmreg bench` invocation and a
table renderer, both covered where they live -- but the guard is the one piece whose failure
is *silent*, and it has already cost one server trip: stage-A run 2 asked for `flir` composed,
found run 1's pre-composition rows on disk, skipped the cell, and printed the old floors under
the new banner. Nothing in the pasted console said so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import p3a_grid  # on the path via `[tool.pytest.ini_options] pythonpath`
from cmreg.results import write_rows
from tests.test_results import make_row


def _run_dir(tmp_path: Path, digest: str | None) -> Path:
    write_rows([make_row("0", residual_calibration=digest)], tmp_path)
    return tmp_path


def _cell(composes: bool) -> p3a_grid.Cell:
    return p3a_grid.Cell("msrs", "driving", "public", "unit", composes=composes)


def test_resuming_onto_a_differently_composed_run_is_refused(tmp_path: Path) -> None:
    """The regression: rows carrying a constant, under a cell that no longer composes."""
    with pytest.raises(SystemExit, match="was scored with residual_calibration"):
        p3a_grid._refuse_a_stale_run(_cell(composes=False), _run_dir(tmp_path, "210142740cdba163"))


def test_resuming_onto_a_matching_run_is_permitted(tmp_path: Path) -> None:
    """The reason the skip exists at all: an interrupted run must still pick up where it
    stopped, or a crashed 20-matcher cell costs the whole grid."""
    p3a_grid._refuse_a_stale_run(_cell(composes=False), _run_dir(tmp_path, None))
