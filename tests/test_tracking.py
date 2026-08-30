"""Tracking is a no-op without W&B, and its tags are derived rather than authored."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cmreg.config import Config
from cmreg.tracking import RunTracker, git_sha, run_name, run_tags


def test_tracker_is_inert_when_wandb_is_off() -> None:
    with RunTracker(Config.load()) as tracker:
        assert not tracker.enabled
        tracker.log({"reg/mace": 1.0})  # must not raise


def test_run_name_matches_the_frozen_format() -> None:
    assert (
        run_name("p3", "roma", "metuvistir", "invgray-up3-magsac", 0)
        == "p3_roma_metuvistir_invgray-up3-magsac_s0"
    )


def test_tags_cover_the_required_vocabulary() -> None:
    config = Config.load(overrides={"gt": {"seed": 4}})
    tags = run_tags(
        config, matcher="roma", preprocess="invgray", estimator="magsac", warp="homography"
    )
    assert {tag.split(":", 1)[0] for tag in tags} == {
        "matcher",
        "preprocess",
        "dataset",
        "domain",
        "estimator",
        "warp",
        "seed",
        "git_sha",
        "platform",
    }
    assert "seed:4" in tags


def test_git_sha_never_raises() -> None:
    assert isinstance(git_sha(), str)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-commit repository, with `git_sha`'s subprocesses run inside it."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "unit@test")
    _git(tmp_path, "config", "user.name", "unit")
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "one")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_clean_tree_carries_no_suffix(repo: Path) -> None:
    assert "-" not in git_sha()


def test_an_untracked_file_is_not_reported_as_dirty(repo: Path) -> None:
    """The distinction P3-7 paid for twice: stage-A runs 2 and 3 read `e60e196-dirty` and
    `367810a-dirty` for a log file and a stray calibration constant, and closing that took two
    round trips to the server. An untracked file is still reported -- an untracked `.py` can
    change a result -- but it does not claim the commit is not what ran."""
    (repo / "stray.log").write_text("noise\n")
    assert git_sha().endswith("-untracked")


def test_a_modified_tracked_file_is_dirty(repo: Path) -> None:
    """The case the flag exists for, and it wins over `-untracked` when both hold: a re-pull
    reproduces neither, and the tracked change is the one that makes the run unreproducible."""
    (repo / "tracked.txt").write_text("two\n")
    (repo / "stray.log").write_text("noise\n")
    assert git_sha().endswith("-dirty")


def test_a_failing_wandb_init_does_not_take_down_the_run(monkeypatch, tmp_path) -> None:
    """The module's first principle, applied to `init` rather than only to `log`.

    A P3-10 cell publishes ~96 W&B runs and the stage ~960, all after the GPU work is done and
    the Parquet is on disk. One transient failure four hours in must not discard every console
    block the stage exists to print -- but it must be loud, because X-1 forbids a local-only
    result and the operator has to see which run is missing.
    """
    import sys
    import types

    stub = types.ModuleType("wandb")
    stub.init = lambda **_: (_ for _ in ()).throw(RuntimeError("network"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", stub)

    config = Config.load(overrides={"runtime": {"wandb": True, "path": str(tmp_path)}})
    with RunTracker(config) as tracker:
        assert not tracker.enabled
        tracker.log({"reg/mace": 1.0})  # still inert rather than raising
