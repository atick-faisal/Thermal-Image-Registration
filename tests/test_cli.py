"""The CLI surface: version, the override table, and the `gt` end-to-end path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmreg import __version__
from cmreg.cli import build_parser, main, overrides_from_args


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_omitted_flags_do_not_shadow_the_config_file() -> None:
    args = build_parser().parse_args(["gt", "--seed", "7"])
    assert overrides_from_args(args) == {"gt": {"seed": 7}}


def test_override_table_nests_by_dotted_path() -> None:
    args = build_parser().parse_args(["gt", "--device", "cpu", "--split", "train", "--limit", "2"])
    assert overrides_from_args(args) == {
        "runtime": {"device": "cpu"},
        "data": {"split": "train", "limit": 2},
    }


def test_the_upsampling_axis_is_expressible_from_the_command_line() -> None:
    """P3-9's stage C varies exactly these two fields against the anchor config, so a missing
    flag would force a second YAML -- which is how a stage quietly redefines its own defaults
    (GRID.md ss1)."""
    args = build_parser().parse_args(["bench", "--upsample", "3", "--interpolation", "lanczos"])
    assert overrides_from_args(args) == {
        "preprocess": {"moving_upsample": 3, "moving_interpolation": "lanczos"}
    }


def test_the_match_count_axis_is_expressible_from_the_command_line() -> None:
    """P3-12c's stage varies exactly these fields against the anchor config, for the same reason
    stage C's flags exist: a missing flag forces a second YAML, which is how a stage quietly
    redefines its own defaults (GRID.md ss1). The caps parse as ints -- a string list would reach
    pydantic and fail naming a nested config path the caller never wrote."""
    args = build_parser().parse_args(
        [
            "bench",
            "--sweep-max-matches",
            "0,256,64",
            "--sweep-selections",
            "confidence,random",
        ]
    )
    assert overrides_from_args(args) == {
        "estimate": {
            "sweep_max_matches": (0, 256, 64),
            "sweep_match_selections": ("confidence", "random"),
        }
    }


def test_a_bad_cap_names_the_flag_not_a_config_path() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bench", "--sweep-max-matches", "0,many"])


def test_gt_writes_records_and_a_config_snapshot(manifest_path: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "gt",
                "--data",
                str(manifest_path),
                "--split",
                "val",
                "--run-dir",
                str(run_dir),
                "--seed",
                "3",
            ]
        )
        == 0
    )

    payload = json.loads((run_dir / "gt_val.json").read_text())
    assert payload["split"] == "val"
    assert len(payload["pairs"]) == 4
    assert len(payload["pairs"][0]["homography"]) == 9
    assert (run_dir / "config.yaml").is_file()


def test_gt_is_reproducible_for_a_given_seed(manifest_path: Path, tmp_path: Path) -> None:
    def run(directory: Path, seed: str) -> list:
        main(["gt", "--data", str(manifest_path), "--run-dir", str(directory), "--seed", seed])
        return json.loads((directory / "gt_val.json").read_text())["pairs"]

    first = run(tmp_path / "a", "3")
    assert run(tmp_path / "b", "3") == first
    assert run(tmp_path / "c", "4") != first


def test_gt_respects_the_limit(manifest_path: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "limited"
    main(["gt", "--data", str(manifest_path), "--run-dir", str(run_dir), "--limit", "2"])
    assert len(json.loads((run_dir / "gt_val.json").read_text())["pairs"]) == 2


def test_calibrate_publishes_a_constant_from_stored_rows(tmp_path: Path) -> None:
    """`cmreg calibrate` reads Parquet and never a matcher (TASKS.md P2-12).

    P1-1b persists each pair's fitted `h` precisely so a third leg costs a re-read rather than
    a second inference run; this is the pin that the command actually takes that path.
    """
    from cmreg.gt import load_calibration
    from cmreg.results import write_rows
    from tests.test_calibration import SHAPE, _rows

    run_dir = tmp_path / "audit"
    write_rows([*_rows("roma", [(5.0, 3.0)] * 4), *_rows("eloftr", [(5.0, 3.0)] * 4)], run_dir)

    out = tmp_path / "flir.json"
    assert main(["calibrate", str(run_dir), "--out", str(out), "--note", "unit"]) == 0

    record = load_calibration(out)
    assert record.matchers == ("roma", "eloftr")
    assert record.height, record.width == SHAPE
    assert record.corner_shift[0] == pytest.approx((5.0, 3.0), abs=1e-3)
    assert record.spread_px == pytest.approx(0.0, abs=1e-3)
    assert record.n_pairs == 4


def test_calibrate_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The console block is the delivery mechanism -- the training server returns text, not
    files -- so it has to carry the whole record on its own."""
    from cmreg.results import write_rows
    from tests.test_calibration import _rows

    run_dir = tmp_path / "audit"
    write_rows(_rows("roma", [(5.0, 3.0)] * 4), run_dir)
    out = tmp_path / "flir.json"
    assert main(["calibrate", str(run_dir), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()

    printed = capsys.readouterr().out
    assert "CMREG RESIDUAL CALIBRATION" in printed
    assert "not a calibration" in printed  # the one-leg warning
    # The block must carry the file itself, not a summary of it: this is the only channel the
    # constant has back from the server.
    body = printed[printed.index("{") : printed.rindex("}") + 1]
    assert json.loads(body)["dataset"] == "msrs"


def test_calibrate_combines_several_run_directories(tmp_path: Path) -> None:
    """A leg per run directory is the shape `scripts/p3b_calibrate.py` produces on the server,
    where each matcher is its own resumable cell."""
    from cmreg.gt import load_calibration
    from cmreg.results import write_rows
    from tests.test_calibration import _rows

    for matcher, shift in (("roma", (5.0, 3.0)), ("eloftr", (7.0, 3.0))):
        write_rows(_rows(matcher, [shift] * 4), tmp_path / matcher)

    out = tmp_path / "msrs.json"
    assert (
        main(["calibrate", str(tmp_path / "roma"), str(tmp_path / "eloftr"), "--out", str(out)])
        == 0
    )
    record = load_calibration(out)
    assert record.matchers == ("roma", "eloftr")
    assert record.corner_shift[0] == pytest.approx((6.0, 3.0), abs=1e-3)
