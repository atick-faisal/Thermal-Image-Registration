"""Every tracked experiment config must load.

An experiment *is* a config file, so a config that no longer validates is a broken
experiment — and with `extra="forbid"`, renaming a schema field silently orphans every YAML
that still sets the old name. This test is what turns that into a failing check rather than
a failed run on the training box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cmreg.config import Config

EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "experiments").glob("*.yaml"))


def test_there_are_experiments_to_check() -> None:
    assert EXPERIMENTS, "experiments/ is empty; this test would otherwise pass vacuously"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_config_loads(path: Path) -> None:
    config = Config.load(path)
    assert len(config.config_hash()) == 16


def test_smoke_config_is_actually_a_smoke_config() -> None:
    """It exists to run on CPU in seconds; a stray edit that unsets either is a slow surprise."""
    config = Config.load(Path(__file__).resolve().parents[1] / "experiments" / "smoke.yaml")
    assert config.runtime.device == "cpu"
    assert 0 < config.data.limit <= 16
    assert not config.runtime.wandb
