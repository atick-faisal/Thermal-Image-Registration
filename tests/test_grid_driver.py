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

import p3_stageb_polarity
import p3a_grid  # on the path via `[tool.pytest.ini_options] pythonpath`
from cmreg.results import write_rows
from tests.test_results import make_row


def _run_dir(tmp_path: Path, digest: str | None) -> Path:
    write_rows([make_row("0", residual_calibration=digest)], tmp_path)
    return tmp_path


# The hash `tests.test_results.make_row` stamps on every row it builds.
ROW_HASH = "0123456789abcdef"


def test_resuming_onto_a_differently_composed_run_is_refused(tmp_path: Path) -> None:
    """The regression: rows carrying a constant, under a cell that no longer composes.

    Reported separately from the general `config_hash` mismatch below even though the
    constant's path is inside that hash, because this is the one whose fix an operator has to
    be told -- produce the constant, or stop asking for it.
    """
    with pytest.raises(SystemExit, match="was scored with residual_calibration"):
        p3a_grid.refuse_a_stale_run(_run_dir(tmp_path, "210142740cdba163"), ROW_HASH, None)


def test_resuming_onto_a_matching_run_is_permitted(tmp_path: Path) -> None:
    """The reason the skip exists at all: an interrupted run must still pick up where it
    stopped, or a crashed 20-matcher cell costs the whole grid."""
    p3a_grid.refuse_a_stale_run(_run_dir(tmp_path, None), ROW_HASH, None)


def test_resuming_onto_a_differently_configured_run_is_refused(tmp_path: Path) -> None:
    """The general case the calibration check was a special case of.

    Stage B (P3-8) varies the preprocess pair, which lives inside `config_hash` and nowhere in
    a printed table: rows scored under `none/percentile` tabulate under an `invert/percentile`
    banner exactly as run 2's uncomposed rows tabulated under a composed one.
    """
    with pytest.raises(SystemExit, match=f"config_hash {ROW_HASH}"):
        p3a_grid.refuse_a_stale_run(_run_dir(tmp_path, None), "ffffffffffffffff", None)


def test_the_intended_hash_is_the_one_the_run_would_stamp(tmp_path: Path) -> None:
    """`intended_hash` resolves the config through `cmreg`'s own parser, so it must agree with
    `Config.load` on the same overrides -- and must move when a scientific field moves."""
    from cmreg.config import Config

    argv = ["bench", "-c", p3a_grid.CONFIG, "--preprocess-ref", "none"]
    assert (
        p3a_grid.intended_hash(argv)
        == Config.load(Path(p3a_grid.CONFIG), {"preprocess": {"reference": "none"}}).config_hash()
    )
    assert p3a_grid.intended_hash(argv) != p3a_grid.intended_hash(
        ["bench", "-c", p3a_grid.CONFIG, "--preprocess-ref", "invert"]
    )
    # `runtime` is excluded from the hash, so a relabelled cell is the same experiment.
    assert p3a_grid.intended_hash(argv) == p3a_grid.intended_hash(
        [*argv, "--group", "p3b_polarity_msrs", "--name", "stageb_msrs_neither"]
    )


class TestStageB:
    """`scripts/p3_stageb_polarity.py` (P3-8). Two things are tested here and nothing else: the
    cell enumeration, because a collision would resume-skip half the stage in silence, and the
    two summary blocks, because their arithmetic *is* the finding and lives nowhere else."""

    def test_the_grid_is_sixteen_cells_with_distinct_directories(self) -> None:
        dirs = [
            p3_stageb_polarity.run_dir_for(cell, polarity)
            for cell in p3a_grid.CELLS
            for polarity in p3_stageb_polarity.POLARITIES
        ]
        assert len(dirs) == 16
        assert len(set(dirs)) == 16

    def test_each_relative_polarity_has_two_recipes(self) -> None:
        """The design (GRID.md §6): the 2x2 is two recipes per relative polarity, differing
        only in which side carries the inversion. A polarity list that lost that symmetry
        would leave the `within` column of the summary block meaningless."""
        flipped = [p for p in p3_stageb_polarity.POLARITIES if p.flipped]
        same = [p for p in p3_stageb_polarity.POLARITIES if not p.flipped]
        assert len(flipped) == len(same) == 2
        assert {p.label for p in flipped} == {"optical", "thermal"}

    def test_a_purely_relative_effect_is_reported_as_one(self) -> None:
        """Rows built so that only the relative polarity matters: the two flipped recipes agree
        to the digit, the two unflipped ones agree, and the pairs differ. The block must report
        ~0 within and a large ratio -- the reading that says 'invert the optical grayscale' is a
        statement about the pair, not about the optical image."""
        table = _polarity_table({"optical": 5.0, "thermal": 5.0, "neither": 25.0, "both": 25.0})
        block = p3_stageb_polarity.relative_polarity_block({"msrs": table})
        row = _row_for("msrs", block)
        assert row[1:4] == ["0.00", "0.00", "20.00"]
        assert row[4] == "inf"

    def test_a_side_specific_effect_is_not_reported_as_relative(self) -> None:
        """The alternative hypothesis, which the 2x2 exists to separate: inverting the optical
        side helps and inverting the thermal side does not, so the two flipped recipes disagree
        as much as the pairs do and the ratio lands near 1."""
        table = _polarity_table({"optical": 5.0, "thermal": 25.0, "neither": 25.0, "both": 25.0})
        row = _row_for("msrs", p3_stageb_polarity.relative_polarity_block({"msrs": table}))
        assert row[1:4] == ["20.00", "0.00", "20.00"]
        assert row[4] == "1.00"

    def test_generality_counts_the_matchers_the_anchor_helps(self) -> None:
        """PLAN.md Figure 6 is a count across matchers, so a matcher the inversion *hurts* must
        not be averaged away by one it helps a great deal."""
        table = {
            "optical": {"roma": _metrics(5.0), "sift": _metrics(30.0)},
            "neither": {"roma": _metrics(25.0), "sift": _metrics(20.0)},
        }
        row = _row_for("msrs", p3_stageb_polarity.generality_block({"msrs": table}))
        assert row[1] == "1/2"  # roma improved by 20 px, sift got 10 px worse
        assert row[2] == "5.00"  # median of [+20, -10]


def _metrics(mace: float, success: float = 0.5) -> dict[str, float]:
    return {
        p3_stageb_polarity.HEADLINE: mace,
        p3_stageb_polarity.SECONDARY: success,
        "reg/n_pairs": 300.0,
    }


def _polarity_table(mace: dict[str, float]) -> p3_stageb_polarity.Table:
    return {label: {"roma": _metrics(value)} for label, value in mace.items()}


def _row_for(dataset: str, block: str) -> list[str]:
    (line,) = [row for row in block.splitlines() if row.startswith(dataset)]
    return line.split()
