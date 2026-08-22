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
