"""The Parquet store and the summary/rendering rules built on it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cmreg.results import (
    PairRow,
    ResultsError,
    read_rows,
    render,
    render_comparison,
    schema,
    summarize,
    write_rows,
)

THRESHOLDS = (3.0, 5.0, 10.0)


def make_row(
    stem: str = "a", *, matcher: str = "sift", success: bool = True, **overrides
) -> PairRow:
    base = PairRow(
        stem=stem,
        dataset="msrs",
        split="val",
        domain="driving",
        platform="public",
        height=480,
        width=640,
        matcher=matcher,
        preprocess_ref="invert",
        preprocess_mov="percentile",
        upsample=1,
        interpolation="bicubic",
        estimator="magsac",
        threshold_px=3.0,
        warp="homography",
        moving="thermal",
        reference="optical",
        seed=0,
        config_hash="0123456789abcdef",
        git_sha="deadbeef",
        run_name="unit",
        success=success,
        failure_reason=None if success else "too_few_matches",
        overlap=0.9,
        corner_err=2.0 if success else None,
        epe_mean=1.5 if success else None,
        epe_median=1.0 if success else None,
        h=[1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0] if success else None,
        n_detected_ref=100,
        n_detected_mov=90,
        n_matches=50,
        n_inliers=40 if success else 0,
        inlier_ratio=0.8 if success else 0.0,
        reproj_err=0.5 if success else None,
        extract_ms=10.0,
        match_ms=5.0,
        estimate_ms=1.0,
        total_ms=16.0,
    )
    return replace(base, **overrides)


def test_schema_declares_every_field(tmp_path: Path) -> None:
    """Declared, never inferred: a run in which every pair failed has all-null measurement
    columns, and an inferred schema would type them `null` and refuse to concatenate."""
    assert schema().names == _field_names()


def _field_names() -> list[str]:
    from dataclasses import fields

    return [field.name for field in fields(PairRow)]


def test_round_trip_preserves_every_row(tmp_path: Path) -> None:
    rows = [make_row("a"), make_row("b", success=False)]
    path = write_rows(rows, tmp_path)
    assert path.name == "pairs.parquet"
    assert list(read_rows(tmp_path)) == rows


def test_an_all_failure_run_round_trips(tmp_path: Path) -> None:
    """The null-column case the declared schema exists for."""
    rows = [make_row(str(i), success=False) for i in range(3)]
    write_rows(rows, tmp_path)
    restored = read_rows(tmp_path)
    assert all(row.corner_err is None for row in restored)
    assert all(row.failure_reason == "too_few_matches" for row in restored)


def test_writing_zero_rows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="zero rows"):
        write_rows([], tmp_path)


def test_reading_a_missing_file_names_it(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="not found"):
        read_rows(tmp_path)


def test_means_run_over_successes_only() -> None:
    """A failed pair has no homography and so no EPE; including it as a zero would reward
    failing."""
    rows = [make_row("a", corner_err=2.0, epe_mean=2.0), make_row("b", success=False)]
    summary = summarize(rows, THRESHOLDS)
    assert summary.metrics["reg/mace"] == pytest.approx(2.0)
    assert summary.metrics["reg/epe_mean"] == pytest.approx(2.0)
    assert summary.metrics["reg/failure_rate"] == pytest.approx(0.5)
    assert summary.metrics["reg/n_pairs"] == 2


def test_failures_count_against_auc_and_success_rate() -> None:
    """The other half of the asymmetry: failures enter the threshold metrics as infinite
    error. Without this, a method that solves one pair perfectly and drops the rest would
    outrank one that solves them all well."""
    solved = [make_row(str(i), corner_err=0.0, epe_mean=0.0) for i in range(2)]
    assert summarize(solved, THRESHOLDS).metrics["reg/success_rate_3px"] == pytest.approx(1.0)
    assert summarize(solved, THRESHOLDS).metrics["reg/auc_3px"] == pytest.approx(1.0)

    mixed = [*solved, make_row("c", success=False), make_row("d", success=False)]
    assert summarize(mixed, THRESHOLDS).metrics["reg/success_rate_3px"] == pytest.approx(0.5)
    assert summarize(mixed, THRESHOLDS).metrics["reg/auc_3px"] == pytest.approx(0.5)


def test_epe_median_is_taken_over_pairs_not_over_pooled_pixels() -> None:
    """Pooling would weight a 4K pair 25x a 640x512 one, so a dataset-wise median would
    mostly describe whichever images happen to be largest."""
    rows = [
        make_row(str(i), corner_err=1.0, epe_mean=value) for i, value in enumerate([1.0, 2.0, 9.0])
    ]
    summary = summarize(rows, THRESHOLDS)
    assert summary.metrics["reg/epe_median"] == pytest.approx(2.0)
    assert summary.metrics["reg/epe_mean"] == pytest.approx(4.0)


def test_an_all_failure_summary_is_not_a_perfect_score() -> None:
    """`nan`, not `0.0`: a run with no successes has no mean, and zero error would read as
    flawless registration."""
    summary = summarize([make_row("a", success=False)], THRESHOLDS)
    assert np.isnan(summary.metrics["reg/mace"])
    assert summary.metrics["reg/failure_rate"] == pytest.approx(1.0)
    assert summary.metrics["reg/success_rate_3px"] == pytest.approx(0.0)


def test_match_and_time_means_include_failures() -> None:
    """A matcher that found nothing still spent the time and still found nothing."""
    rows = [make_row("a", n_matches=100), make_row("b", success=False, n_matches=0)]
    assert summarize(rows, THRESHOLDS).metrics["match/total"] == pytest.approx(50.0)
    assert summarize(rows, THRESHOLDS).metrics["time/total_ms"] == pytest.approx(16.0)


def test_summarizing_zero_rows_is_refused() -> None:
    from cmreg.results import ReportError

    with pytest.raises(ReportError, match="zero rows"):
        summarize([], THRESHOLDS)


def test_the_console_block_carries_every_metric_key() -> None:
    """The block is the primary channel out of the training server, so a key missing from it
    is a number nobody can read."""
    summary = summarize([make_row("a")], THRESHOLDS)
    block = render(summary)
    assert block.startswith("=== CMREG RESULT ===")
    assert block.endswith("=== END ===")
    for key in summary.metrics:
        assert key in block
    assert "0123456789abcdef" in block


def test_the_block_states_the_failure_asymmetry_when_there_are_failures() -> None:
    with_failure = render(summarize([make_row("a"), make_row("b", success=False)], THRESHOLDS))
    assert "excluded from epe/mace" in with_failure
    assert "excluded from epe/mace" not in render(summarize([make_row("a")], THRESHOLDS))


def test_the_comparison_table_has_one_row_per_matcher() -> None:
    summaries = [
        summarize([make_row("a", matcher="sift")], THRESHOLDS),
        summarize([make_row("a", matcher="orb")], THRESHOLDS),
    ]
    table = render_comparison(summaries, ("reg/mace", "reg/success_rate_3px"))
    assert "sift" in table
    assert "orb" in table
    # Header columns must not abut, or the table is unreadable in a pasted block.
    assert "  " in table.splitlines()[1]
