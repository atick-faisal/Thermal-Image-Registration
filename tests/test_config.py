"""The config layer's two load-bearing behaviours: unknown-key rejection and hash stability."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmreg.config import Config, ConfigError, Domain, Platform, deep_merge


def _write(tmp_path: Path, data: dict) -> Path:
    target = tmp_path / "experiment.yaml"
    target.write_text(yaml.safe_dump(data))
    return target


def test_defaults_load_without_a_file() -> None:
    config = Config.load()
    assert config.data.split == "val"
    assert config.eval.domain is Domain.DRIVING
    assert config.eval.platform is Platform.PUBLIC


def test_unknown_keys_are_rejected_not_ignored(tmp_path: Path) -> None:
    """A silently-dropped typo costs a full sweep to discover."""
    path = _write(tmp_path, {"gt": {"rotaton_deg": 10.0}})
    with pytest.raises(ConfigError, match=r"gt\.rotaton_deg"):
        Config.load(path)


def test_every_error_is_reported_not_just_the_first(tmp_path: Path) -> None:
    path = _write(tmp_path, {"gt": {"typo_one": 1, "typo_two": 2}, "data": {"typo_three": 3}})
    with pytest.raises(ConfigError) as excinfo:
        Config.load(path)
    assert "3 errors" in str(excinfo.value)


def test_missing_config_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "absent.yaml")


def test_overrides_merge_nestedly_without_wiping_the_section(tmp_path: Path) -> None:
    path = _write(tmp_path, {"gt": {"seed": 3, "rotation_deg": 12.0}})
    config = Config.load(path, {"gt": {"seed": 9}})
    assert config.gt.seed == 9
    assert config.gt.rotation_deg == 12.0


def test_config_hash_ignores_runtime() -> None:
    """The same experiment on two machines under two names must carry one hash."""
    base = Config.load()
    renamed = Config.load(overrides={"runtime": {"name": "other", "device": "cuda"}})
    assert renamed.config_hash() == base.config_hash()


def test_config_hash_tracks_the_science() -> None:
    assert Config.load(overrides={"gt": {"seed": 1}}).config_hash() != Config.load().config_hash()
    assert (
        Config.load(overrides={"data": {"split": "train"}}).config_hash()
        != Config.load().config_hash()
    )


def test_snapshot_round_trips(tmp_path: Path) -> None:
    config = Config.load(overrides={"runtime": {"name": "snap"}, "gt": {"seed": 5}})
    written = config.snapshot(tmp_path / "run")
    restored = Config.model_validate(yaml.safe_load(written.read_text()))
    assert restored.config_hash() == config.config_hash()
    assert restored.runtime.name == "snap"


def test_sections_are_frozen() -> None:
    config = Config.load()
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on frozen
        config.gt.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("gt", {"scale_min": 1.5, "scale_max": 1.2}, "scale_min <= scale_max"),
        ("gt", {"rotation_deg": -1.0}, ">= 0"),
        ("data", {"split": "test"}, "train"),
        ("data", {"limit": -1}, ">= 0"),
        ("eval", {"thresholds_px": [5.0, 3.0]}, "ascending"),
        ("eval", {"thresholds_px": []}, "non-empty"),
    ],
)
def test_impossible_values_are_rejected(section: str, payload: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Config.load(overrides={section: payload})


def test_deep_merge_leaves_untouched_keys_alone() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}


def test_the_reference_modality_defaults_to_the_other_side() -> None:
    """`None` means "whichever one `moving` is not". Two callers deriving that separately is
    how a benchmark quietly starts registering an image against itself."""
    from cmreg.config import Modality

    assert Config().gt.reference_modality is Modality.OPTICAL
    assert not Config().gt.is_monomodal
    flipped = Config.model_validate({"gt": {"moving": "optical"}})
    assert flipped.gt.reference_modality is Modality.THERMAL
    assert not flipped.gt.is_monomodal


def test_a_monomodal_config_is_a_different_experiment() -> None:
    """`reference` is scientific, so it is inside `config_hash()`: a control cell and the
    benchmark cell it controls for must not share a fingerprint (TASKS.md P1-1b)."""
    control = Config.model_validate({"gt": {"reference": "thermal", "moving": "thermal"}})
    assert control.gt.is_monomodal
    assert control.config_hash() != Config().config_hash()


def test_composing_a_residual_is_a_different_experiment(tmp_path: Path) -> None:
    """`gt.residual_calibration` is scientific, so it must move the hash.

    A run that composes a rig constant into its ground truth is not the same measurement as one
    that does not -- P3-7's F1 is the whole argument -- and two rows sharing a `config_hash`
    across that boundary would be pooled as if they were replicates.
    """
    plain = Config.model_validate({})
    composed = Config.model_validate({"gt": {"residual_calibration": str(tmp_path / "flir.json")}})
    assert composed.config_hash() != plain.config_hash()


def test_the_residual_calibration_defaults_to_none() -> None:
    """Tier-1 as originally specified. Composition is opt-in per dataset (GRID.md §3): `msrs`
    and `dronevehicle` must never compose, so the default cannot be "whatever file is there"."""
    assert Config.model_validate({}).gt.residual_calibration is None
