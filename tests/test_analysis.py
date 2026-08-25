"""The residual decomposition (TASKS.md P1-1b).

The whole point of this module is to tell a *systematic* residual from a *random* one, so the
suite has to exercise both directions. A statistic that always answered "systematic" would pass
a test written only against the fixture's fixed offset, and would then have confirmed P1-1a's
most consequential open question by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmreg.analysis.residual import AnalysisError, by_matcher, render, residual_structure
from tests.test_results import make_row

SHAPE = (240, 320)


def _translation(dx: float, dy: float) -> list[float]:
    return [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]


def _rows(offsets, matcher: str = "roma"):
    return [
        make_row(str(i), matcher=matcher, h=_translation(dx, dy), height=SHAPE[0], width=SHAPE[1])
        for i, (dx, dy) in enumerate(offsets)
    ]


def test_a_shared_offset_reads_as_systematic() -> None:
    """Every pair displaced identically: the consensus carries the whole magnitude and there
    is nothing left to scatter."""
    structure = residual_structure(_rows([(5.0, 3.0)] * 20))
    expected = float(np.hypot(5.0, 3.0))
    assert structure.magnitude == pytest.approx(expected)
    assert structure.consensus == pytest.approx(expected)
    assert structure.scatter == pytest.approx(0.0, abs=1e-6)
    assert structure.systematic_fraction == pytest.approx(1.0)
    for dx, dy in structure.corner_shift:
        assert (dx, dy) == pytest.approx((5.0, 3.0))


def test_zero_mean_noise_reads_as_random() -> None:
    """The negative case. Residuals scattered around identity leave essentially no consensus,
    so the magnitude lands almost wholly in the scatter term."""
    rng = np.random.default_rng(3)
    offsets = rng.normal(0.0, 4.0, size=(400, 2))
    structure = residual_structure(_rows(offsets))
    assert structure.consensus < 0.5
    assert structure.scatter == pytest.approx(structure.magnitude, rel=0.05)
    assert structure.systematic_fraction < 0.1


def test_an_offset_buried_in_noise_is_still_recovered() -> None:
    """The case the real datasets are expected to be: a fixed rig offset plus per-pair noise.
    Both parts have to come back separated, not pooled into one number."""
    rng = np.random.default_rng(5)
    offsets = rng.normal(0.0, 1.0, size=(400, 2)) + np.array([5.0, 3.0])
    structure = residual_structure(_rows(offsets))
    assert structure.consensus == pytest.approx(float(np.hypot(5.0, 3.0)), abs=0.2)
    assert 0.5 < structure.systematic_fraction < 0.95


def test_one_degenerate_fit_does_not_capture_the_consensus() -> None:
    """P1-1a measured a single near-degenerate fit driving a dataset mean to 5.6e8 px. The
    consensus is a median for exactly that reason, and this pins it."""
    rows = _rows([(5.0, 3.0)] * 19 + [(1e7, -1e7)])
    structure = residual_structure(rows)
    assert structure.consensus == pytest.approx(float(np.hypot(5.0, 3.0)))


def test_failures_are_excluded_and_counted() -> None:
    rows = [*_rows([(5.0, 3.0)] * 4), make_row("x", matcher="roma", success=False)]
    structure = residual_structure(rows)
    assert (structure.n_pairs, structure.n_failed) == (4, 1)


def test_a_group_with_no_successes_is_refused() -> None:
    with pytest.raises(AnalysisError, match="no successful rows"):
        residual_structure([make_row("x", success=False)])


def test_mixed_image_shapes_are_refused() -> None:
    """A corner error is only comparable at one image size, so pooling two is a silent
    category error rather than a wider average."""
    rows = [*_rows([(5.0, 3.0)]), make_row("z", h=_translation(5.0, 3.0), height=480, width=640)]
    with pytest.raises(AnalysisError, match="one image shape"):
        residual_structure(rows)


def test_by_matcher_keeps_first_appearance_order() -> None:
    rows = [*_rows([(5.0, 3.0)] * 2, matcher="roma"), *_rows([(1.0, 0.0)] * 2, matcher="eloftr")]
    assert [s.matcher for s in by_matcher(rows)] == ["roma", "eloftr"]


def test_the_block_carries_both_parts_and_a_reading() -> None:
    block = render(residual_structure(_rows([(5.0, 3.0)] * 8)))
    assert "consensus" in block and "scatter" in block
    assert "systematic" in block
    assert block.startswith("=== CMREG RESIDUAL STRUCTURE ===") and block.endswith("=== END ===")


def test_the_decomposition_runs_end_to_end_off_a_real_run(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """The fixture's rig offset, through the actual evaluation cell and the Parquet store,
    comes back out of the decomposition as a consensus with no scatter."""
    from cmreg.eval import run_benchmark
    from cmreg.results import read_rows
    from tests.conftest import OFFSET_PX
    from tests.test_runner import IDENTITY_WARP, base_config

    config = base_config(offset_dataset / "data.yaml", tmp_path / "e2e", gt=IDENTITY_WARP)
    run_benchmark(config)
    structure = residual_structure(read_rows(tmp_path / "e2e"))

    assert structure.consensus == pytest.approx(float(np.hypot(*OFFSET_PX)), abs=0.1)
    # Not zero: SIFT localises a corner to a few hundredths of a pixel and the fixture's val
    # split is two pairs. The claim under test is the ratio below, not an absolute px floor.
    assert structure.scatter < 0.3
    assert structure.systematic_fraction > 0.95


def test_a_heavy_tail_is_reported_as_such_rather_than_as_random() -> None:
    """The DroneVehicle shape: most pairs agree on a small offset, a minority are gross fits.

    Both `magnitude` and `scatter` are means, so the minority drags the ratio toward 0 and the
    coarse label would read "random -- the cross-modal localisation limit". That is the wrong
    conclusion: the typical pair is fine and a handful of fits failed. The median separates the
    two cases and the reading has to abstain.
    """
    rows = _rows([(5.0, 3.0)] * 36 + [(400.0, -400.0)] * 14)
    structure = residual_structure(rows)

    assert structure.magnitude_median == pytest.approx(float(np.hypot(5.0, 3.0)))
    assert structure.magnitude > 2.0 * structure.magnitude_median
    assert structure.consensus == pytest.approx(float(np.hypot(5.0, 3.0)))
    assert "tail-dominated" in render(structure)


def test_a_clean_dataset_reports_a_median_matching_its_mean() -> None:
    """The guard must not fire on the case it is not for: without a tail, mean == median and
    the systematic/random label stands."""
    rows = _rows([(5.0, 3.0)] * 20)
    structure = residual_structure(rows)

    assert structure.magnitude_median == pytest.approx(structure.magnitude)
    assert "tail-dominated" not in render(structure)
    assert "systematic" in render(structure)
