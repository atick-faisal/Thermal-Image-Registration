"""Tracking is a no-op without W&B, and its tags are derived rather than authored."""

from __future__ import annotations

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
