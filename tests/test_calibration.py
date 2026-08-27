"""The residual calibration constant (TASKS.md P2-12).

Two things are worth testing here and they are not the same thing. One is the record: a corner
field that round-trips, a digest that identifies the constant rather than the file, and two
mismatches that must be fatal because neither announces itself. The other is the construction
-- that an across-matcher median is a median and not an average, which is what stops one
degenerate leg becoming the published rig constant.

The *direction* of composition is pinned in ``test_runner.py`` instead, against the
``offset_dataset`` fixture's known rig displacement. It belongs there because it is a property
of the evaluation cell, and because only an end-to-end run can catch an inversion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmreg.analysis.residual import across_matchers, residual_structure
from cmreg.gt import CalibrationError, ResidualCalibration, load_calibration, write_calibration
from cmreg.metrics import corner_error
from cmreg.warp import corners
from tests.test_results import make_row

SHAPE = (480, 640)


def make_calibration(**overrides) -> ResidualCalibration:
    declared = {
        "dataset": "flir",
        "height": SHAPE[0],
        "width": SHAPE[1],
        "corner_shift": ((0.5, 6.0), (-12.0, 0.9), (0.4, -5.7), (13.5, -0.1)),
        "matchers": ("roma", "superpoint-lightglue", "eloftr"),
        "spread_px": 1.23,
        "worst_case_px": 2.33,
        "n_pairs": 1013,
        "split": "val",
        "git_sha": "deadbeef",
        "note": "unit",
    }
    return ResidualCalibration(**{**declared, **overrides})


def _translation(dx: float, dy: float) -> list[float]:
    return [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]


def _rows(matcher: str, shifts: list[tuple[float, float]]) -> list:
    return [
        make_row(str(i), matcher=matcher, h=_translation(dx, dy), height=SHAPE[0], width=SHAPE[1])
        for i, (dx, dy) in enumerate(shifts)
    ]


# --- the record ------------------------------------------------------------------------


def test_the_corner_field_round_trips_through_its_homography() -> None:
    """The matrix and the four displacements are the same object, stated two ways."""
    record = make_calibration()
    reference = corners(record.shape)
    from cmreg.warp import warp_points

    recovered = warp_points(reference, record.homography()) - reference
    # 1e-4 rather than machine epsilon: `cv2.getPerspectiveTransform` takes float32 points, so
    # the refit costs ~3e-6 px on a 640 px corner. Three orders of magnitude below the 1.23 px
    # uncertainty the constant is published with, and worth stating rather than hiding in a
    # loose tolerance.
    assert recovered == pytest.approx(np.asarray(record.corner_shift), abs=1e-4)


def test_the_magnitude_is_the_misalignment_composition_removes() -> None:
    """`magnitude_px` must agree with the corner error the constant carries, since that is the
    quantity every P1-1 table is quoted in."""
    record = make_calibration()
    against_identity = corner_error(record.homography(), np.eye(3), record.shape)
    assert record.magnitude_px() == pytest.approx(against_identity, abs=1e-4)


def test_a_json_round_trip_preserves_the_record(tmp_path: Path) -> None:
    record = make_calibration()
    path = write_calibration(record, tmp_path / "flir.json")
    assert load_calibration(path) == record


def test_the_digest_identifies_the_constant_not_the_file() -> None:
    """Provenance changing must not change the digest, and a corner moving must.

    This is the whole reason the row carries a digest instead of the configured path: two
    machines can hold one filename over two different matrices, and one machine can re-note a
    file without changing what it says.
    """
    record = make_calibration()
    assert make_calibration(note="rewritten", git_sha="cafe").digest() == record.digest()
    moved = list(record.corner_shift)
    moved[2] = (moved[2][0] + 0.01, moved[2][1])
    assert make_calibration(corner_shift=tuple(moved)).digest() != record.digest()


def test_the_written_digest_is_not_read_back(tmp_path: Path) -> None:
    """A hand-edited corner must not be able to hide behind the digest written beside it."""
    path = write_calibration(make_calibration(), tmp_path / "flir.json")
    data = json.loads(path.read_text())
    data["corner_shift"][0][0] += 5.0
    path.write_text(json.dumps(data))
    assert load_calibration(path).digest() != data["digest"]


def test_a_wrong_dataset_is_refused() -> None:
    """Composing `flir`'s rig into `msrs` removes an offset that was never there and adds one
    that is -- and neither half fails loudly on its own."""
    with pytest.raises(CalibrationError, match=r"flir.*msrs"):
        make_calibration().validate_for("msrs", SHAPE)


def test_a_wrong_shape_is_refused() -> None:
    """A corner field does not rescale: applied at twice the resolution it is silently half the
    misalignment it claims to be."""
    with pytest.raises(CalibrationError, match="does not rescale"):
        make_calibration().validate_for("flir", (960, 1280))


def test_the_matching_dataset_and_shape_pass() -> None:
    make_calibration().validate_for("flir", SHAPE)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"corner_shift": ((0.0, 0.0), (1.0, 1.0))}, "4 corners"),
        ({"matchers": ()}, "estimated from"),
        ({"height": 0}, "positive"),
    ],
)
def test_a_malformed_record_is_refused(overrides: dict, match: str) -> None:
    with pytest.raises(CalibrationError, match=match):
        make_calibration(**overrides)


def test_a_malformed_file_names_what_was_wrong(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"dataset": "flir"}')
    with pytest.raises(CalibrationError, match="malformed"):
        load_calibration(path)


def test_a_missing_file_raises_a_calibration_error(tmp_path: Path) -> None:
    """Not an `OSError`: every layer here raises its own class, so a caller can tell a bad
    constant from a bad path without reading the message."""
    with pytest.raises(CalibrationError, match="cannot read"):
        load_calibration(tmp_path / "absent.json")


# --- the construction ------------------------------------------------------------------


def test_the_across_matcher_field_is_a_median_not_a_mean() -> None:
    """One matcher failing badly must not move the published constant.

    The per-pair version of this is pinned in ``test_analysis.py``; this is the level above,
    where the samples are matchers rather than pairs and there are only ever two or three of
    them -- which is exactly where a mean is most easily captured.
    """
    legs = [
        residual_structure(_rows("roma", [(5.0, 3.0)] * 8)),
        residual_structure(_rows("eloftr", [(5.2, 3.1)] * 8)),
        residual_structure(_rows("splg", [(400.0, 300.0)] * 8)),
    ]
    consensus = across_matchers(legs)
    assert consensus.corner_shift[0] == pytest.approx((5.2, 3.1), abs=1e-3)
    # And the outlier is *reported*, not silently discarded (X-4).
    assert consensus.worst_case_px > 100.0


def test_each_leg_reports_its_distance_from_the_median() -> None:
    """P1-1d's finding was which matcher stood apart, not merely that one did."""
    legs = [
        residual_structure(_rows("roma", [(5.0, 3.0)] * 4)),
        residual_structure(_rows("eloftr", [(8.0, 3.0)] * 4)),
        residual_structure(_rows("splg", [(11.0, 3.0)] * 4)),
    ]
    consensus = across_matchers(legs)
    assert consensus.matchers == ("roma", "eloftr", "splg")
    assert consensus.leg_distance_px == pytest.approx((3.0, 0.0, 3.0), abs=1e-3)
    assert consensus.spread_px == pytest.approx(2.0, abs=1e-3)


def test_a_single_leg_is_permitted_and_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """Permitted because a one-matcher run has nothing else to offer; warned because shipping
    it as a calibration publishes that matcher's bias as the dataset's rig."""
    with caplog.at_level("WARNING"):
        consensus = across_matchers([residual_structure(_rows("roma", [(5.0, 3.0)] * 4))])
    assert consensus.leg_distance_px == pytest.approx((0.0,))
    assert "not a calibration" in caplog.text


def test_the_thinnest_leg_bounds_the_pair_count() -> None:
    """Summing would count the same pairs once per matcher and flatter the evidence."""
    legs = [
        residual_structure(_rows("roma", [(5.0, 3.0)] * 8)),
        residual_structure(_rows("eloftr", [(5.0, 3.0)] * 3)),
    ]
    assert across_matchers(legs).n_pairs == 3


def test_combining_across_datasets_is_refused() -> None:
    legs = [
        residual_structure(_rows("roma", [(5.0, 3.0)] * 4)),
        residual_structure(
            [
                make_row(
                    "x",
                    matcher="eloftr",
                    dataset="flir",
                    h=_translation(5.0, 3.0),
                    height=SHAPE[0],
                    width=SHAPE[1],
                )
            ]
        ),
    ]
    with pytest.raises(Exception, match="across datasets"):
        across_matchers(legs)


def test_combining_across_shapes_is_refused() -> None:
    """The same four displacements are a different warp at a different resolution."""
    legs = [
        residual_structure(_rows("roma", [(5.0, 3.0)] * 4)),
        residual_structure(
            [make_row("x", matcher="eloftr", h=_translation(5.0, 3.0), height=960, width=1280)]
        ),
    ]
    with pytest.raises(Exception, match="across shapes"):
        across_matchers(legs)
